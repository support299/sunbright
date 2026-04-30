"""
AI dashboard insights via an OpenAI-compatible chat completions API (same pattern as sunbright-dashboard
`server/_core/llm.ts`: Manus Forge + Gemini by default).
"""
import json
import logging
import os
import re
import time
import hashlib
import urllib.error
import urllib.request
from typing import Any

from django.core.cache import cache

from dashboard.models import InsightConversation, InsightMessage
from dashboard.services.analytics_service import (
    get_clean_deals_bundle,
    get_cx_bundle,
    get_manager_bundle,
    get_performance_bundle,
    get_pipeline_bundle,
    get_retention_bundle,
)
from dashboard.services.project_service import (
    get_category_breakdown,
    get_cancellation_reasons_breakdown,
    get_cancelled_projects,
    get_on_hold_reasons_breakdown,
    get_on_hold_projects,
    get_overview_metrics,
)

logger = logging.getLogger(__name__)


class InsightsLLMError(Exception):
    """Upstream LLM failure or response could not be parsed."""


class InsightsRateLimitError(Exception):
    """Chat rate/quota limits hit."""


class InsightsConversationError(Exception):
    """Conversation lookup or ownership error."""


def _forge_chat_url() -> str:
    base = (os.getenv("BUILT_IN_FORGE_API_URL") or "").strip().rstrip("/")
    if base:
        return f"{base}/v1/chat/completions"
    # forge.manus.im no longer resolves in public DNS; .ai is the working host (same path as sunbright-dashboard intent).
    return "https://forge.manus.ai/v1/chat/completions"


def _forge_api_key() -> str:
    return (os.getenv("BUILT_IN_FORGE_API_KEY") or "").strip()


def _llm_model() -> str:
    return (os.getenv("BUILT_IN_FORGE_MODEL") or "gemini-2.5-flash").strip()


def _llm_request_includes_thinking() -> bool:
    """Forge/Manus accepts a `thinking` block; OpenAI and most providers reject unknown keys."""
    if os.getenv("BUILT_IN_FORGE_SKIP_THINKING", "").strip().lower() in ("1", "true", "yes"):
        return False
    base = (os.getenv("BUILT_IN_FORGE_API_URL") or "").strip().lower()
    if "openai.com" in base:
        return False
    return True


def _max_completion_tokens() -> int:
    """Honor BUILT_IN_FORGE_MAX_TOKENS; otherwise OpenAI caps at 16384 for common models, Forge keeps 32768."""
    raw = os.getenv("BUILT_IN_FORGE_MAX_TOKENS", "").strip()
    if raw:
        try:
            return max(256, min(int(raw), 200_000))
        except ValueError:
            pass
    base = (os.getenv("BUILT_IN_FORGE_API_URL") or "").strip().lower()
    if "openai.com" in base:
        return 16384
    return 32768


def _chat_max_output_tokens() -> int:
    raw = os.getenv("INSIGHTS_CHAT_MAX_OUTPUT_TOKENS", "").strip()
    if raw:
        try:
            return max(128, min(int(raw), 8192))
        except ValueError:
            pass
    return 800


def _chat_max_input_tokens() -> int:
    raw = os.getenv("INSIGHTS_CHAT_MAX_INPUT_TOKENS", "").strip()
    if raw:
        try:
            return max(512, min(int(raw), 24000))
        except ValueError:
            pass
    return 4500


def _chat_rate_limit_per_minute() -> int:
    raw = os.getenv("INSIGHTS_CHAT_MAX_REQUESTS_PER_MINUTE", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), 120))
        except ValueError:
            pass
    return 20


def _chat_daily_quota() -> int:
    raw = os.getenv("INSIGHTS_CHAT_DAILY_QUOTA", "").strip()
    if raw:
        try:
            return max(0, min(int(raw), 10000))
        except ValueError:
            pass
    return 0


INSIGHTS_JSON_SCHEMA: dict[str, Any] = {
    "name": "insights_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "executiveSummary": {
                "type": "string",
                "description": "2-3 paragraph executive summary of overall performance",
            },
            "keyMetrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "value": {"type": "string"},
                        "status": {"type": "string", "enum": ["good", "warning", "critical"]},
                        "insight": {"type": "string"},
                    },
                    "required": ["metric", "value", "status", "insight"],
                    "additionalProperties": False,
                },
            },
            "repInsights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "repName": {"type": "string"},
                        "strength": {"type": "string"},
                        "improvement": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["repName", "strength", "improvement", "recommendation"],
                    "additionalProperties": False,
                },
            },
            "teamInsights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "teamName": {"type": "string"},
                        "strength": {"type": "string"},
                        "improvement": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["teamName", "strength", "improvement", "recommendation"],
                    "additionalProperties": False,
                },
            },
            "retentionInsights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "area": {"type": "string"},
                        "finding": {"type": "string"},
                        "recommendation": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["area", "finding", "recommendation", "priority"],
                    "additionalProperties": False,
                },
            },
            "cxInsights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "area": {"type": "string"},
                        "finding": {"type": "string"},
                        "recommendation": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["area", "finding", "recommendation", "priority"],
                    "additionalProperties": False,
                },
            },
            "actionItems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "owner": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        "expectedImpact": {"type": "string"},
                    },
                    "required": ["action", "owner", "priority", "expectedImpact"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "executiveSummary",
            "keyMetrics",
            "repInsights",
            "teamInsights",
            "retentionInsights",
            "cxInsights",
            "actionItems",
        ],
        "additionalProperties": False,
    },
}


def gather_insights_context(date_from, date_to, user) -> dict[str, Any]:
    perf = get_performance_bundle(date_from, date_to, user)
    cx = get_cx_bundle(date_from, date_to, user)
    retention = get_retention_bundle(date_from, date_to, user)
    clean = get_clean_deals_bundle(date_from, date_to, user)
    return {
        "overview": get_overview_metrics(date_from, date_to, user),
        "repPerformance": (perf.get("reps") or [])[:15],
        "teamPerformance": (perf.get("teams") or [])[:15],
        "cleanDealsByRep": (clean.get("byRep") or [])[:15],
        "retentionByRep": (retention.get("byRep") or [])[:15],
        "retentionByLeadSource": (retention.get("byLeadSource") or [])[:25],
        "cancellationReasons": get_cancellation_reasons_breakdown(date_from, date_to, user)[:25],
        "onHoldReasons": get_on_hold_reasons_breakdown(date_from, date_to, user)[:25],
        "cxOverview": cx.get("overview"),
        "cxByInstaller": (cx.get("byInstaller") or [])[:15],
    }


def _build_user_prompt(ctx: dict[str, Any]) -> str:
    payload = json.dumps(ctx, indent=2, default=str)
    return f"""You are a senior solar industry data analyst for Sunbright Solar USA. Analyze the following dashboard data and provide actionable insights and recommendations for executives, managers, and sales reps.

## Company Data Summary

{payload}

## Instructions
Provide your analysis in the following JSON structure (the API enforces the schema). Be specific with names, numbers, and percentages. Reference actual data points. Each recommendation should be actionable and tied to a specific metric."""


def _invoke_llm_chat(
    messages: list[dict[str, str]],
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
    retry_once: bool = False,
) -> dict[str, Any]:
    key = _forge_api_key()
    if not key:
        raise InsightsLLMError("BUILT_IN_FORGE_API_KEY is not set (internal call path).")

    body: dict[str, Any] = {
        "model": _llm_model(),
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format:
        body["response_format"] = response_format
    if _llm_request_includes_thinking():
        body["thinking"] = {"budget_tokens": 128}

    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        _forge_chat_url(),
        data=raw,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    timeout = float(os.getenv("BUILT_IN_FORGE_TIMEOUT_SECONDS") or "25")
    attempts = 2 if retry_once else 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_error = InsightsLLMError(f"LLM HTTP {e.code}: {detail}")
            if attempt + 1 >= attempts or e.code < 500:
                raise last_error from e
        except urllib.error.URLError as e:
            hint = ""
            err_s = str(e).lower()
            if "getaddrinfo" in err_s or "11001" in err_s or "name or service not known" in err_s:
                hint = (
                    " Could not resolve the API hostname (DNS). If you set BUILT_IN_FORGE_API_URL, check it. "
                    "The default host is https://forge.manus.ai (not .im). "
                    "Or remove BUILT_IN_FORGE_API_KEY to use rule-based insights without the network."
                )
            last_error = InsightsLLMError(f"LLM request failed: {e}.{hint}")
            if attempt + 1 >= attempts:
                raise last_error from e
        except json.JSONDecodeError as e:
            raise InsightsLLMError("LLM returned invalid JSON envelope") from e
    else:
        raise last_error or InsightsLLMError("LLM request failed")

    try:
        choices = parsed["choices"]
        msg = choices[0]["message"]
        content = msg["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise InsightsLLMError("LLM response missing choices[0].message.content") from e

    if not isinstance(content, str) or not content.strip():
        raise InsightsLLMError("LLM returned empty content")
    usage = parsed.get("usage") or {}
    return {
        "content": content,
        "usage": {
            "input": int(usage.get("prompt_tokens") or 0),
            "output": int(usage.get("completion_tokens") or 0),
        },
        "finish_reason": str(choices[0].get("finish_reason") or ""),
    }


def _invoke_llm(system_prompt: str, user_prompt: str) -> str:
    out = _invoke_llm_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=_max_completion_tokens(),
        response_format={"type": "json_schema", "json_schema": INSIGHTS_JSON_SCHEMA},
    )
    return out["content"]


def _metric_status_net_retention(rate: float) -> str:
    if rate >= 82.0:
        return "good"
    if rate >= 68.0:
        return "warning"
    return "critical"


def _metric_status_cancellation(rate: float) -> str:
    if rate <= 8.0:
        return "good"
    if rate <= 18.0:
        return "warning"
    return "critical"


def _metric_status_clean_pct(pct: float) -> str:
    if pct >= 55.0:
        return "good"
    if pct >= 35.0:
        return "warning"
    return "critical"


def _rep_name(row: dict[str, Any]) -> str:
    return str(row.get("salesRep") or row.get("sales_rep") or "Unknown rep").strip() or "Unknown rep"


def _team_name(row: dict[str, Any]) -> str:
    return str(row.get("salesTeam") or row.get("sales_team") or "Unknown team").strip() or "Unknown team"


def _heuristic_insights_from_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Same response shape as the LLM path, but computed locally when no Forge key is configured
    (mirrors how sunbright-dashboard is often used: metrics-only until a host-injected key exists).
    """
    ov = ctx.get("overview") or {}
    total = int(ov.get("totalProjects") or 0)
    active = int(ov.get("activeProjects") or 0)
    cancelled = int(ov.get("cancelledProjects") or 0)
    on_hold = int(ov.get("onHoldProjects") or 0)
    clean = int(ov.get("cleanDeals") or 0)
    clean_pct = float(ov.get("cleanDealPct") or 0.0)
    cancel_rate = float(ov.get("cancellationRate") or 0.0)
    net_ret = float(ov.get("netRetentionRate") or 0.0)
    pipe = float(ov.get("activePipelineValue") or 0.0)
    tcv = float(ov.get("totalContractValue") or 0.0)
    avg_install = ov.get("avgDaysToInstall")

    cx = ctx.get("cxOverview") or {}
    cx_installs = int(cx.get("totalInstalls") or 0)
    review_rate = float(cx.get("reviewCaptureRate") or 0.0)

    if total == 0:
        executive = (
            "There are no projects in the current filter window, so portfolio KPIs cannot be compared. "
            "Widen the date range in the header or run a sync if you expect data."
        )
    else:
        p1 = (
            f"In this period the database shows {total} projects ({active} active). "
            f"Clean deals account for {clean_pct:.1f}% of volume ({clean} clean of {total}), "
            f"with a net retention rate of {net_ret:.1f}% and cancellation rate of {cancel_rate:.1f}%."
        )
        p2 = (
            f"Active pipeline value is about ${pipe:,.0f} on roughly ${tcv:,.0f} total contract value in scope. "
        )
        if avg_install is not None:
            p2 += f"Average days from customer date to install is {avg_install} days. "
        if cx_installs:
            p2 += (
                f"Customer experience rows cover {cx_installs} installs with a "
                f"{review_rate:.1f}% review capture rate."
            )
        executive = p1 + "\n\n" + p2.strip()

    key_metrics: list[dict[str, str]] = []
    if total:
        key_metrics.append(
            {
                "metric": "Net retention",
                "value": f"{net_ret:.1f}%",
                "status": _metric_status_net_retention(net_ret),
                "insight": "Share of projects not cancelled, on hold, or red-flagged in this cohort.",
            }
        )
        key_metrics.append(
            {
                "metric": "Clean deal rate",
                "value": f"{clean_pct:.1f}%",
                "status": _metric_status_clean_pct(clean_pct),
                "insight": "Clean installs as a share of all projects in the filter.",
            }
        )
        key_metrics.append(
            {
                "metric": "Cancellation rate",
                "value": f"{cancel_rate:.1f}%",
                "status": _metric_status_cancellation(cancel_rate),
                "insight": f"{cancelled} cancelled projects in this window.",
            }
        )
        key_metrics.append(
            {
                "metric": "On hold",
                "value": str(on_hold),
                "status": "warning" if on_hold > max(3, total // 25) else "good",
                "insight": "Projects currently paused in workflow.",
            }
        )
    if cx_installs:
        key_metrics.append(
            {
                "metric": "CX review capture",
                "value": f"{review_rate:.1f}%",
                "status": "good" if review_rate >= 40 else "warning" if review_rate >= 20 else "critical",
                "insight": "Installs with a captured review vs installs in CX scope.",
            }
        )

    rep_insights: list[dict[str, str]] = []
    for r in (ctx.get("repPerformance") or [])[:6]:
        t = int(r.get("totalProjects") or 0)
        if t < 1:
            continue
        cp = float(r.get("cleanDealPct") or 0.0)
        nr = float(r.get("netRetentionRate") or 0.0)
        cr = float(r.get("cancellationRate") or 0.0)
        rep_insights.append(
            {
                "repName": _rep_name(r),
                "strength": f"Clean deal rate {cp:.1f}% across {t} projects; net retention {nr:.1f}%.",
                "improvement": f"Cancellations at {cr:.1f}% of rep volume — review stalled or at-risk deals.",
                "recommendation": "Pair weekly pipeline reviews with top loss reasons for this rep.",
            }
        )

    team_insights: list[dict[str, str]] = []
    for r in (ctx.get("teamPerformance") or [])[:5]:
        t = int(r.get("totalProjects") or 0)
        if t < 1:
            continue
        cp = float(r.get("cleanDealPct") or 0.0)
        nr = float(r.get("netRetentionRate") or 0.0)
        team_insights.append(
            {
                "teamName": _team_name(r),
                "strength": f"{t} projects with {cp:.1f}% clean rate and {nr:.1f}% net retention.",
                "improvement": "Compare install-cycle delays vs company median to find friction.",
                "recommendation": "Align team coaching on the lowest cohort metric vs branch average.",
            }
        )

    retention_insights: list[dict[str, Any]] = []
    lead_rows = [x for x in (ctx.get("retentionByLeadSource") or []) if int(x.get("totalProjects") or 0) >= 3]
    lead_rows.sort(key=lambda x: float(x.get("netRetentionRate") or 0.0))
    for row in lead_rows[:3]:
        ls = str(row.get("lead_source") or "Lead source").strip() or "Lead source"
        nr = float(row.get("netRetentionRate") or 0.0)
        tp = int(row.get("totalProjects") or 0)
        retention_insights.append(
            {
                "area": f"Lead source: {ls}",
                "finding": f"Net retention {nr:.1f}% over {tp} projects in this filter.",
                "recommendation": "Validate lead quality, handoff SLAs, and pricing fit for this channel.",
                "priority": "high" if nr < 60 else "medium",
            }
        )
    top_cancel = (ctx.get("cancellationReasons") or [])[:1]
    if top_cancel:
        rc = top_cancel[0]
        retention_insights.append(
            {
                "area": "Cancellations",
                "finding": f"Top reason: \"{rc.get('reason')}\" ({rc.get('count')} projects).",
                "recommendation": "Run a focused win/loss review on this reason with sales and operations.",
                "priority": "high" if int(rc.get("count") or 0) > 5 else "medium",
            }
        )
    if not retention_insights:
        retention_insights.append(
            {
                "area": "Retention",
                "finding": "Not enough segmented retention rows in this filter for a channel-level signal.",
                "recommendation": "Expand the date range or ensure lead source fields are populated on sync.",
                "priority": "low",
            }
        )

    cx_insights: list[dict[str, Any]] = []
    if cx_installs:
        cx_insights.append(
            {
                "area": "Reviews",
                "finding": f"Review capture rate is {review_rate:.1f}% across {cx_installs} installs.",
                "recommendation": "Tighten post-install follow-up so more jobs receive a review request within a week.",
                "priority": "high" if review_rate < 25 else "medium" if review_rate < 45 else "low",
            }
        )
    for row in (ctx.get("cxByInstaller") or [])[:4]:
        inst = str(row.get("installer") or "Installer").strip() or "Installer"
        ti = int(row.get("totalInstalls") or 0)
        rr = float(row.get("reviewCaptureRate") or 0.0)
        if ti < 2:
            continue
        cx_insights.append(
            {
                "area": f"Installer: {inst}",
                "finding": f"{ti} installs, {rr:.1f}% review capture.",
                "recommendation": "Share best-practice install closeouts from higher-capture crews with this partner.",
                "priority": "medium" if rr < 35 else "low",
            }
        )
    if not cx_insights:
        cx_insights.append(
            {
                "area": "Customer experience",
                "finding": "No CX install rows in this date window (CX uses install date).",
                "recommendation": "Adjust the header filter or sync CX data if post-install metrics should appear here.",
                "priority": "low",
            }
        )

    actions: list[dict[str, str]] = []
    if total:
        actions.append(
            {
                "action": f"Review the top cancellation reason with managers ({cancel_rate:.1f}% overall rate).",
                "owner": "Sales leadership",
                "priority": "high" if cancel_rate > 15 else "medium",
                "expectedImpact": "Fewer late-stage losses and clearer coaching targets.",
            }
        )
        actions.append(
            {
                "action": "Reconcile on-hold backlog with project owners and set exit dates.",
                "owner": "Operations",
                "priority": "medium" if on_hold else "low",
                "expectedImpact": "Lower working capital risk and clearer pipeline forecasting.",
            }
        )
    if cx_installs and review_rate < 40:
        actions.append(
            {
                "action": "Launch a 7-day post-install review request campaign by installer tier.",
                "owner": "CX / Marketing",
                "priority": "medium",
                "expectedImpact": "Higher review capture and referral-ready customers.",
            }
        )
    if not actions:
        actions.append(
            {
                "action": "Load dashboard data for a meaningful date range, then re-run insights.",
                "owner": "Admin",
                "priority": "low",
                "expectedImpact": "Actionable metrics for the team.",
            }
        )

    return {
        "insightSource": "heuristic",
        "executiveSummary": executive,
        "keyMetrics": key_metrics,
        "repInsights": rep_insights,
        "teamInsights": team_insights,
        "retentionInsights": retention_insights,
        "cxInsights": cx_insights,
        "actionItems": actions[:8],
    }


CHAT_SYSTEM_PROMPT = """You are an AI assistant for Sunbright analytics dashboard.

Rules:
- Only use provided dashboard data when answering dashboard-related questions.
- Do not hallucinate metrics or values.
- If some data is missing, still provide best-effort analysis from available metrics and clearly label limitations.
- Keep answers concise and business-focused.
- Politely redirect unrelated questions back to Sunbright analytics help."""

CHAT_FALLBACK_MESSAGE = "I’m having trouble generating insights right now. Please try again."

INJECTION_PATTERNS = (
    r"ignore\s+previous\s+instructions",
    r"reveal\s+.*system\s+prompt",
    r"show\s+.*api\s+key",
    r"developer\s+message",
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _detect_safety_flags(message: str) -> dict[str, Any]:
    lowered = message.lower()
    matches = [pattern for pattern in INJECTION_PATTERNS if re.search(pattern, lowered)]
    return {
        "possiblePromptInjection": bool(matches),
        "matchedPatterns": matches,
    }


def _classify_intent(message: str) -> str:
    text = message.lower().strip()
    if not text:
        return InsightMessage.IntentLabel.GREETING_SMALLTALK

    dashboard_signals = (
        # Broad analytics terms
        "dashboard",
        "metric",
        "insight",
        "analysis",
        "trend",
        "kpi",
        # Executive overview
        "overview",
        "revenue",
        "contract",
        "active pipeline",
        # Clean deals
        "clean deal",
        "clean deals",
        "deal quality",
        "realization ratio",
        # Retention / cancellations / on-hold
        "retention",
        "churn",
        "cancel",
        "cancellation",
        "cancelled",
        "on hold",
        "red flagged",
        # Performance
        "revenue",
        "sales",
        "project",
        "pipeline",
        "pipeline velocity",
        "cx",
        "customer experience",
        "installer",
        "rep",
        "team",
        # Manager performance / outcome pending
        "manager",
        "manager performance",
        "sit down",
        "qualified sit down",
        "closing rate",
        "pending outcome",
        "outcome pending",
        "door stats",
        # Sidebar labels
        "executive overview",
        "clean deals",
        "retention",
        "rep performance",
        "team performance",
        "pipeline velocity",
        "on hold details",
        "cancellations",
        "customer experience",
        "manager performance",
        "outcome pending",
    )
    if any(token in text for token in dashboard_signals):
        return InsightMessage.IntentLabel.DASHBOARD_QUERY

    smalltalk_signals = ("hi", "hello", "hey", "good morning", "good evening", "thanks", "thank you")
    if any(text.startswith(token) for token in smalltalk_signals):
        return InsightMessage.IntentLabel.GREETING_SMALLTALK

    return InsightMessage.IntentLabel.OUT_OF_SCOPE


def _detect_dashboard_scope(message: str) -> str:
    text = message.lower()
    if any(token in text for token in ("manager performance", "manager", "sit down", "closing rate", "door stats")):
        return "manager"
    if any(token in text for token in ("outcome pending", "pending outcome", "pending deals", "pending")):
        return "outcome_pending"
    if any(token in text for token in ("on hold details", "on hold", "hold reason")):
        return "on_hold_details"
    if any(token in text for token in ("cancellations", "cancelled projects", "cancellation reasons")):
        return "cancellations"
    if any(token in text for token in ("retention", "lead source", "churn", "cancel", "cancellation")):
        return "retention"
    if any(token in text for token in ("cx", "customer experience", "review", "installer", "testimonial")):
        return "cx"
    if any(token in text for token in ("rep", "team", "sales", "performance", "manager")):
        return "performance"
    if any(token in text for token in ("clean deal", "clean", "deal quality")):
        return "clean_deals"
    if any(token in text for token in ("pipeline", "revenue", "contract", "overview", "summary", "dashboard")):
        return "executive_overview"
    return "executive_overview"


def _detect_metric_hint(message: str, scope: str | None) -> str | None:
    text = (message or "").lower()
    if scope == "executive_overview":
        if "pipeline" in text or "velocity" in text:
            return "pipeline_velocity"
        if "revenue" in text or "contract" in text:
            return "overview_revenue"
        return "executive_overview"
    if scope == "manager":
        if "rep" in text:
            return "manager_rep"
        if "team" in text:
            return "manager_team"
        if "pending" in text:
            return "pending_outcome"
        return "manager_overview"
    if scope == "on_hold_details":
        return "on_hold_details"
    if scope == "cancellations":
        return "cancellations"
    if scope == "outcome_pending":
        return "pending_outcome"
    if scope == "retention":
        if any(term in text for term in ("cancelled projects", "most cancelled", "losing customers", "lost customers")):
            return "cancelled_projects"
        if "cancellation rate" in text or "cancel rate" in text or "%" in text:
            return "cancellation_rate"
        if "retention" in text or "churn" in text:
            return "retention_overview"
    if scope == "performance":
        if "rep" in text:
            return "rep_performance"
        if "team" in text:
            return "team_performance"
    if scope == "cx":
        return "cx_overview"
    return None


def _gather_chat_context(scope: str, date_from, date_to, user, metric_hint: str | None = None) -> dict[str, Any]:
    context: dict[str, Any] = {
        "overview": get_overview_metrics(date_from, date_to, user),
    }
    if scope == "performance":
        perf = get_performance_bundle(date_from, date_to, user)
        context["repPerformance"] = (perf.get("reps") or [])[:12]
        context["teamPerformance"] = (perf.get("teams") or [])[:12]
    elif scope == "retention":
        retention = get_retention_bundle(date_from, date_to, user)
        context["retentionByRep"] = (retention.get("byRep") or [])[:12]
        context["retentionByLeadSource"] = (retention.get("byLeadSource") or [])[:16]
        context["cancellationReasons"] = get_cancellation_reasons_breakdown(date_from, date_to, user)[:12]
        context["onHoldReasons"] = get_on_hold_reasons_breakdown(date_from, date_to, user)[:12]
    elif scope == "cx":
        cx = get_cx_bundle(date_from, date_to, user)
        context["cxOverview"] = cx.get("overview") or {}
        context["cxByInstaller"] = (cx.get("byInstaller") or [])[:12]
    elif scope == "clean_deals":
        clean = get_clean_deals_bundle(date_from, date_to, user)
        context["cleanDealsByRep"] = (clean.get("byRep") or [])[:12]
    elif scope == "executive_overview":
        if metric_hint == "pipeline_velocity":
            pipeline = get_pipeline_bundle(date_from, date_to, user)
            velocity_rows = pipeline.get("velocity") or []
            averages = pipeline.get("averages") or {}
            context["pipelineAverages"] = averages
            context["pipelineVelocitySample"] = velocity_rows[:20]
        context["categoryBreakdown"] = get_category_breakdown(date_from, date_to, user)[:10]
    elif scope == "manager":
        manager = get_manager_bundle(date_from, date_to, user)
        context["managerOverview"] = manager.get("managerOverview") or {}
        context["managerRepPerformance"] = (manager.get("repPerformance") or [])[:12]
        context["managerTeamPerformance"] = (manager.get("teamPerformance") or [])[:10]
        context["doorStats"] = manager.get("doorStats") or {}
        context["dealStageBreakdown"] = (manager.get("dealStageBreakdown") or [])[:10]
        context["pendingOutcome"] = (manager.get("pendingOutcome") or [])[:15]
    elif scope == "on_hold_details":
        on_hold_rows = get_on_hold_projects(date_from, date_to, user).values(
            "first_name", "last_name", "sales_rep", "sales_team", "on_hold_reason", "job_status", "customer_since"
        )[:25]
        context["onHoldReasons"] = get_on_hold_reasons_breakdown(date_from, date_to, user)[:12]
        context["onHoldProjectsSample"] = list(on_hold_rows)
    elif scope == "cancellations":
        cancelled_rows = get_cancelled_projects(date_from, date_to, user).values(
            "first_name", "last_name", "sales_rep", "sales_team", "cancellation_reason", "job_status", "customer_since"
        )[:25]
        context["cancellationReasons"] = get_cancellation_reasons_breakdown(date_from, date_to, user)[:12]
        context["cancelledProjectsSample"] = list(cancelled_rows)
    elif scope == "outcome_pending":
        manager = get_manager_bundle(date_from, date_to, user)
        context["managerOverview"] = manager.get("managerOverview") or {}
        context["pendingOutcome"] = (manager.get("pendingOutcome") or [])[:20]
    return context


def _filter_context_for_llm(shaped_context: dict[str, Any] | None, scope: str | None, metric_hint: str | None) -> dict[str, Any] | None:
    if not shaped_context:
        return None

    filtered: dict[str, Any] = {
        "scope": shaped_context.get("scope"),
        "shape_version": "v3-llm-minimal",
    }

    # Keep a tiny overview block only for high-level percentage references.
    overview = shaped_context.get("overview") or {}
    if overview:
        filtered["overview"] = {
            "cancellation_rate": overview.get("cancellation_rate"),
            "net_retention_rate": overview.get("net_retention_rate"),
            "total_projects": overview.get("total_projects"),
        }

    if scope == "retention":
        reps = shaped_context.get("retention_by_rep") or []
        if metric_hint in ("cancelled_projects", "cancellation_rate", "retention_overview"):
            filtered["retention_by_rep"] = [
                {
                    "sales_rep": _rep_name(row),
                    "cancelledProjects": int(row.get("cancelledProjects") or 0),
                    "cancellationRate": float(row.get("cancellationRate") or 0.0),
                    "netRetentionRate": float(row.get("netRetentionRate") or 0.0),
                }
                for row in reps[:8]
            ]
        if metric_hint in ("retention_overview", None):
            filtered["retention_by_lead_source"] = [
                {
                    "lead_source": str(row.get("lead_source") or row.get("leadSource") or "Unknown"),
                    "netRetentionRate": float(row.get("netRetentionRate") or 0.0),
                    "totalProjects": int(row.get("totalProjects") or 0),
                }
                for row in (shaped_context.get("retention_by_lead_source") or [])[:6]
            ]
        if metric_hint in ("retention_overview", "cancellation_rate", None):
            reasons = [
                {
                    "reason": str(row.get("reason") or ""),
                    "count": int(row.get("count") or 0),
                }
                for row in (shaped_context.get("cancellation_reasons") or [])[:5]
            ]
            filtered["cancellation_reasons"] = reasons
            filtered["top_cancellation_reasons"] = [
                f"{row['reason']} ({row['count']})" for row in reasons if row.get("reason")
            ]
    elif scope == "performance":
        if metric_hint in ("rep_performance", None):
            filtered["rep_breakdown"] = [
                {
                    "sales_rep": _rep_name(row),
                    "totalProjects": int(row.get("totalProjects") or 0),
                    "cleanDealPct": float(row.get("cleanDealPct") or 0.0),
                    "cancellationRate": float(row.get("cancellationRate") or 0.0),
                }
                for row in (shaped_context.get("rep_breakdown") or [])[:8]
            ]
        if metric_hint in ("team_performance", None):
            filtered["team_breakdown"] = [
                {
                    "sales_team": str(row.get("salesTeam") or row.get("sales_team") or "Unknown"),
                    "totalProjects": int(row.get("totalProjects") or 0),
                    "cleanDealPct": float(row.get("cleanDealPct") or 0.0),
                    "cancellationRate": float(row.get("cancellationRate") or 0.0),
                }
                for row in (shaped_context.get("team_breakdown") or [])[:8]
            ]
    elif scope == "cx":
        filtered["cx_overview"] = shaped_context.get("cx_overview") or {}
        filtered["cx_by_installer"] = [
            {
                "installer": str(row.get("installer") or "Unknown"),
                "totalInstalls": int(row.get("totalInstalls") or 0),
                "reviewCaptureRate": float(row.get("reviewCaptureRate") or 0.0),
            }
            for row in (shaped_context.get("cx_by_installer") or [])[:6]
        ]
    elif scope == "clean_deals":
        filtered["clean_deals_by_rep"] = [
            {
                "sales_rep": _rep_name(row),
                "cleanDeals": int(row.get("cleanDeals") or 0),
                "totalProjects": int(row.get("totalProjects") or 0),
                "cleanDealPct": float(row.get("cleanDealPct") or 0.0),
            }
            for row in (shaped_context.get("clean_deals_by_rep") or [])[:8]
        ]
    elif scope == "executive_overview":
        if metric_hint == "pipeline_velocity":
            pipeline_averages = shaped_context.get("pipeline_averages") or {}
            filtered["pipeline_averages"] = {
                "avgDaysToCrc": pipeline_averages.get("avgDaysToCrc"),
                "avgDaysToPermit": pipeline_averages.get("avgDaysToPermit"),
                "avgDaysToInstall": pipeline_averages.get("avgDaysToInstall"),
                "avgDaysInstallToPto": pipeline_averages.get("avgDaysInstallToPto"),
                "avgDaysToPtoSubmitted": pipeline_averages.get("avgDaysToPtoSubmitted"),
            }
            velocity_rows = shaped_context.get("pipeline_velocity_sample") or []
            filtered["pipeline_velocity_sample"] = [
                {
                    "salesRep": row.get("salesRep"),
                    "salesTeam": row.get("salesTeam"),
                    "daysToInstall": row.get("daysToInstall"),
                    "daysToPtoSubmitted": row.get("daysToPtoSubmitted"),
                    "daysToPermit": row.get("daysToPermit"),
                    "projectAgeDays": row.get("projectAgeDays"),
                }
                for row in velocity_rows[:8]
            ]
        filtered["category_breakdown"] = [
            {
                "project_category": row.get("project_category"),
                "count": int(row.get("count") or 0),
            }
            for row in (shaped_context.get("category_breakdown") or [])[:8]
        ]
    elif scope == "manager":
        filtered["manager_overview"] = shaped_context.get("manager_overview") or {}
        if metric_hint in ("manager_rep", "manager_overview", None):
            filtered["manager_rep_performance"] = [
                {
                    "salesRep": row.get("salesRep"),
                    "salesTeam": row.get("salesTeam"),
                    "totalAppointments": row.get("totalAppointments"),
                    "closedDeals": row.get("closedDeals"),
                    "cancelledDeals": row.get("cancelledDeals"),
                    "sitDownRate": row.get("sitDownRate"),
                    "closingRate": row.get("closingRate"),
                }
                for row in (shaped_context.get("manager_rep_performance") or [])[:8]
            ]
        if metric_hint in ("manager_team", "manager_overview", None):
            filtered["manager_team_performance"] = [
                {
                    "salesTeam": row.get("salesTeam"),
                    "repCount": row.get("repCount"),
                    "totalAppointments": row.get("totalAppointments"),
                    "closedDeals": row.get("closedDeals"),
                    "cancelledDeals": row.get("cancelledDeals"),
                    "sitDownRate": row.get("sitDownRate"),
                    "closingRate": row.get("closingRate"),
                }
                for row in (shaped_context.get("manager_team_performance") or [])[:8]
            ]
        filtered["door_stats"] = shaped_context.get("door_stats") or {}
        filtered["deal_stage_breakdown"] = (shaped_context.get("deal_stage_breakdown") or [])[:8]
        if metric_hint in ("pending_outcome", "manager_overview", None):
            filtered["pending_outcome"] = (shaped_context.get("pending_outcome") or [])[:10]
    elif scope == "on_hold_details":
        filtered["on_hold_reasons"] = (shaped_context.get("on_hold_reasons") or [])[:8]
        filtered["on_hold_projects_sample"] = [
            {
                "name": f"{(row.get('first_name') or '').strip()} {(row.get('last_name') or '').strip()}".strip() or "Unknown",
                "salesRep": row.get("sales_rep"),
                "salesTeam": row.get("sales_team"),
                "reason": row.get("on_hold_reason") or row.get("job_status"),
            }
            for row in (shaped_context.get("on_hold_projects_sample") or [])[:10]
        ]
    elif scope == "cancellations":
        filtered["cancellation_reasons"] = (shaped_context.get("cancellation_reasons") or [])[:8]
        filtered["cancelled_projects_sample"] = [
            {
                "name": f"{(row.get('first_name') or '').strip()} {(row.get('last_name') or '').strip()}".strip() or "Unknown",
                "salesRep": row.get("sales_rep"),
                "salesTeam": row.get("sales_team"),
                "reason": row.get("cancellation_reason") or row.get("job_status"),
            }
            for row in (shaped_context.get("cancelled_projects_sample") or [])[:10]
        ]
    elif scope == "outcome_pending":
        filtered["manager_overview"] = shaped_context.get("manager_overview") or {}
        filtered["pending_outcome"] = (shaped_context.get("pending_outcome") or [])[:12]

    return filtered


def _shape_dashboard_context(raw: dict[str, Any], scope: str) -> dict[str, Any]:
    overview = raw.get("overview") or {}
    shaped = {
        "overview": {
            "metric": "portfolio_overview",
            "total_projects": int(overview.get("totalProjects") or 0),
            "active_projects": int(overview.get("activeProjects") or 0),
            "clean_deals": int(overview.get("cleanDeals") or 0),
            "clean_deal_pct": float(overview.get("cleanDealPct") or 0.0),
            "cancellation_rate": float(overview.get("cancellationRate") or 0.0),
            "net_retention_rate": float(overview.get("netRetentionRate") or 0.0),
            "pipeline_value": float(overview.get("activePipelineValue") or 0.0),
        },
        "scope": scope,
        "shape_version": "v2-minimal",
    }
    if raw.get("repPerformance"):
        shaped["rep_breakdown"] = (raw.get("repPerformance") or [])[:10]
    if raw.get("teamPerformance"):
        shaped["team_breakdown"] = (raw.get("teamPerformance") or [])[:10]
    if raw.get("cleanDealsByRep"):
        shaped["clean_deals_by_rep"] = (raw.get("cleanDealsByRep") or [])[:10]
    if raw.get("retentionByRep"):
        shaped["retention_by_rep"] = (raw.get("retentionByRep") or [])[:10]
    if raw.get("retentionByLeadSource"):
        shaped["retention_by_lead_source"] = (raw.get("retentionByLeadSource") or [])[:12]
    if raw.get("cancellationReasons"):
        shaped["cancellation_reasons"] = (raw.get("cancellationReasons") or [])[:10]
    if raw.get("onHoldReasons"):
        shaped["on_hold_reasons"] = (raw.get("onHoldReasons") or [])[:10]
    if raw.get("cxOverview"):
        shaped["cx_overview"] = raw.get("cxOverview") or {}
    if raw.get("cxByInstaller"):
        shaped["cx_by_installer"] = (raw.get("cxByInstaller") or [])[:10]
    if raw.get("pipelineAverages"):
        shaped["pipeline_averages"] = raw.get("pipelineAverages") or {}
    if raw.get("pipelineVelocitySample"):
        shaped["pipeline_velocity_sample"] = (raw.get("pipelineVelocitySample") or [])[:12]
    if raw.get("categoryBreakdown"):
        shaped["category_breakdown"] = (raw.get("categoryBreakdown") or [])[:10]
    if raw.get("managerOverview"):
        shaped["manager_overview"] = raw.get("managerOverview") or {}
    if raw.get("managerRepPerformance"):
        shaped["manager_rep_performance"] = (raw.get("managerRepPerformance") or [])[:10]
    if raw.get("managerTeamPerformance"):
        shaped["manager_team_performance"] = (raw.get("managerTeamPerformance") or [])[:10]
    if raw.get("doorStats"):
        shaped["door_stats"] = raw.get("doorStats") or {}
    if raw.get("dealStageBreakdown"):
        shaped["deal_stage_breakdown"] = (raw.get("dealStageBreakdown") or [])[:10]
    if raw.get("pendingOutcome"):
        shaped["pending_outcome"] = (raw.get("pendingOutcome") or [])[:12]
    if raw.get("onHoldProjectsSample"):
        shaped["on_hold_projects_sample"] = (raw.get("onHoldProjectsSample") or [])[:12]
    if raw.get("cancelledProjectsSample"):
        shaped["cancelled_projects_sample"] = (raw.get("cancelledProjectsSample") or [])[:12]
    return shaped


def _build_context_meta(
    intent_label: str,
    date_from,
    date_to,
    shaped_context: dict[str, Any] | None,
    data_scope: str | None = None,
) -> dict[str, Any]:
    meta = {
        "intent": intent_label,
        "date_range": {
            "date_from": str(date_from) if date_from else None,
            "date_to": str(date_to) if date_to else None,
        },
        "context_type": "minimal",
        "record_count": 0,
    }
    if shaped_context:
        rep_count = len(shaped_context.get("rep_breakdown") or [])
        team_count = len(shaped_context.get("team_breakdown") or [])
        lead_count = len(shaped_context.get("retention_by_lead_source") or [])
        cancel_count = len(shaped_context.get("cancellation_reasons") or [])
        cx_count = len(shaped_context.get("cx_by_installer") or [])
        meta.update(
            {
                "context_type": "dashboard_summary",
                "shape_version": shaped_context.get("shape_version"),
                "record_count": rep_count + team_count + lead_count + cancel_count + cx_count,
                "data_scope": data_scope or shaped_context.get("scope") or "overview",
            }
        )
    return meta


def _history_messages(conversation: InsightConversation) -> list[dict[str, str]]:
    rows = list(conversation.messages.filter(deleted_at__isnull=True).order_by("-created_at")[:5])
    rows.reverse()
    return [{"role": row.role, "content": row.content} for row in rows if row.role in {"user", "assistant", "system"}]


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _requires_explanation(text: str) -> bool:
    lowered = (text or "").lower()
    explanation_terms = (
        "why",
        "trend",
        "analysis",
        "detailed",
        "insight",
        "insights",
        "recommend",
        "improve",
        "reason",
        "root cause",
        "explain",
        "what should",
        "how can",
    )
    return any(term in lowered for term in explanation_terms)


def _explanation_mode_note(text: str) -> str:
    if _requires_explanation(text):
        return (
            "Explanation mode: if direct causal attribution is unavailable, provide grounded business hypotheses "
            "from available aggregate data and clearly mark them as inferred (not confirmed). "
            "Do not stop at 'data unavailable'."
        )
    return ""


def _deterministic_cancellation_rep_plan(text: str, intent_label: str, data_scope: str | None) -> dict[str, Any] | None:
    if intent_label != InsightMessage.IntentLabel.DASHBOARD_QUERY or data_scope != "retention":
        return None

    normalized = _normalize_question(text)
    cancel_terms = (
        "cancel",
        "cancelled",
        "cancellation",
        "cancel rate",
        "losing customers",
        "lost customers",
        "churn",
    )
    has_metric = any(term in normalized for term in cancel_terms)
    has_entity = "rep" in normalized or "sales rep" in normalized
    asks_top = any(term in normalized for term in ("highest", "most", "max", "top"))
    if not (has_metric and has_entity and asks_top):
        return None
    if _requires_explanation(normalized):
        return None

    match = re.search(r"\btop\s+(\d+)\b", normalized)
    top_n = 1
    if match:
        try:
            top_n = max(1, min(int(match.group(1)), 10))
        except ValueError:
            top_n = 1
    operation = "top_n" if top_n > 1 else "max"
    metric = "cancellation_rate"
    if any(term in normalized for term in ("most cancelled", "losing customers", "lost customers", "cancelled projects", "cancelled customers")):
        metric = "cancelled_projects"
    if "rate" in normalized or "%" in normalized:
        metric = "cancellation_rate"
    return {
        "metric": metric,
        "entity": "sales_rep",
        "operation": operation,
        "top_n": top_n,
    }


def _deterministic_rank_plan(
    text: str,
    intent_label: str,
    data_scope: str | None,
    metric_hint: str | None,
) -> dict[str, Any] | None:
    if intent_label != InsightMessage.IntentLabel.DASHBOARD_QUERY:
        return None
    normalized = _normalize_question(text)
    if _requires_explanation(normalized):
        return None
    if not any(term in normalized for term in ("top", "best", "highest", "most", "lowest", "least", "worst")):
        return None

    direction = "max"
    if any(term in normalized for term in ("lowest", "least", "worst")):
        direction = "min"

    match = re.search(r"\btop\s+(\d+)\b", normalized)
    top_n = 1
    rank_position = None
    if match:
        try:
            top_n = max(1, min(int(match.group(1)), 10))
        except ValueError:
            top_n = 1
    ordinal_map = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
        "fifth": 5,
        "5th": 5,
    }
    for token, pos in ordinal_map.items():
        if token in normalized:
            rank_position = pos
            top_n = max(top_n, pos)
            break
    operation = "top_n" if top_n > 1 else direction

    scope = data_scope or "executive_overview"
    if scope == "manager":
        source_key, name_key, entity = "managerRepPerformance", "salesRep", "manager"
        if "team" in normalized:
            source_key, name_key, entity = "managerTeamPerformance", "salesTeam", "team"
        metric_key, metric_label = "closingRate", "closing rate"
        if "sit down" in normalized:
            metric_key, metric_label = "sitDownRate", "sit-down rate"
        elif "cancel" in normalized:
            metric_key, metric_label = "cancelledDeals", "cancelled deals"
        elif "appointment" in normalized:
            metric_key, metric_label = "totalAppointments", "total appointments"
        elif "pending" in normalized:
            metric_key, metric_label = "pendingOutcome", "pending outcome"
        return {
            "scope": scope,
            "source_key": source_key,
            "name_key": name_key,
            "entity": entity,
            "metric_key": metric_key,
            "metric_label": metric_label,
            "operation": operation,
            "top_n": top_n,
            "rank_position": rank_position,
            "metric_hint": metric_hint,
        }

    if scope == "performance":
        source_key, name_key, entity = "repPerformance", "salesRep", "sales rep"
        if "team" in normalized:
            source_key, name_key, entity = "teamPerformance", "salesTeam", "team"
        metric_key, metric_label = "totalProjects", "total projects"
        if "cancel" in normalized:
            metric_key, metric_label = "cancellationRate", "cancellation rate"
        elif "retention" in normalized:
            metric_key, metric_label = "netRetentionRate", "net retention rate"
        elif "clean" in normalized:
            metric_key, metric_label = "cleanDealPct", "clean deal percentage"
        return {
            "scope": scope,
            "source_key": source_key,
            "name_key": name_key,
            "entity": entity,
            "metric_key": metric_key,
            "metric_label": metric_label,
            "operation": operation,
            "top_n": top_n,
            "rank_position": rank_position,
            "metric_hint": metric_hint,
        }

    if scope == "retention":
        source_key, name_key, entity = "retentionByRep", "sales_rep", "sales rep"
        metric_key, metric_label = "cancellationRate", "cancellation rate"
        if "cancelled" in normalized or "losing customers" in normalized or "lost customers" in normalized:
            metric_key, metric_label = "cancelledProjects", "cancelled projects"
        elif "retention" in normalized:
            metric_key, metric_label = "netRetentionRate", "net retention rate"
        elif "reason" in normalized:
            source_key, name_key, entity = "cancellationReasons", "reason", "reason"
            metric_key, metric_label = "count", "count"
        return {
            "scope": scope,
            "source_key": source_key,
            "name_key": name_key,
            "entity": entity,
            "metric_key": metric_key,
            "metric_label": metric_label,
            "operation": operation,
            "top_n": top_n,
            "rank_position": rank_position,
            "metric_hint": metric_hint,
        }

    if scope == "clean_deals":
        return {
            "scope": scope,
            "source_key": "cleanDealsByRep",
            "name_key": "salesRep",
            "entity": "sales rep",
            "metric_key": "cleanDealPct" if "pct" in normalized or "percentage" in normalized else "cleanDeals",
            "metric_label": "clean deal percentage" if "pct" in normalized or "percentage" in normalized else "clean deals",
            "operation": operation,
            "top_n": top_n,
            "rank_position": rank_position,
            "metric_hint": metric_hint,
        }

    if scope == "cx":
        return {
            "scope": scope,
            "source_key": "cxByInstaller",
            "name_key": "installer",
            "entity": "installer",
            "metric_key": "reviewCaptureRate" if "review" in normalized else "totalInstalls",
            "metric_label": "review capture rate" if "review" in normalized else "total installs",
            "operation": operation,
            "top_n": top_n,
            "rank_position": rank_position,
            "metric_hint": metric_hint,
        }

    if scope == "on_hold_details":
        return {
            "scope": scope,
            "source_key": "onHoldReasons",
            "name_key": "reason",
            "entity": "reason",
            "metric_key": "count",
            "metric_label": "count",
            "operation": operation,
            "top_n": top_n,
            "rank_position": rank_position,
            "metric_hint": metric_hint,
        }

    if scope == "cancellations":
        return {
            "scope": scope,
            "source_key": "cancellationReasons",
            "name_key": "reason",
            "entity": "reason",
            "metric_key": "count",
            "metric_label": "count",
            "operation": operation,
            "top_n": top_n,
            "rank_position": rank_position,
            "metric_hint": metric_hint,
        }

    if scope == "outcome_pending":
        return {
            "scope": scope,
            "source_key": "pendingOutcome",
            "name_key": "salesRep",
            "entity": "sales rep",
            "metric_key": "count",
            "metric_label": "pending outcomes",
            "operation": operation,
            "top_n": top_n,
            "rank_position": rank_position,
            "metric_hint": metric_hint,
        }

    if scope == "executive_overview":
        return {
            "scope": scope,
            "source_key": "categoryBreakdown",
            "name_key": "project_category",
            "entity": "project category",
            "metric_key": "count",
            "metric_label": "project count",
            "operation": operation,
            "top_n": top_n,
            "rank_position": rank_position,
            "metric_hint": metric_hint,
        }
    return None


def _compute_cancellation_rep_result(plan: dict[str, Any], date_from, date_to, user) -> tuple[str, dict[str, Any]] | None:
    raw_context = _gather_chat_context("retention", date_from, date_to, user)
    rows = raw_context.get("retentionByRep") or []
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        rate = row.get("cancellationRate")
        cancelled = row.get("cancelledProjects")
        if rate is None:
            continue
        try:
            rate_num = float(rate)
        except (TypeError, ValueError):
            continue
        try:
            cancelled_num = int(cancelled or 0)
        except (TypeError, ValueError):
            cancelled_num = 0
        parsed_rows.append(
            {
                "sales_rep": _rep_name(row),
                "cancellation_rate": round(rate_num, 1),
                "cancelled_projects": cancelled_num,
            }
        )

    if not parsed_rows:
        return None

    metric = str(plan.get("metric") or "cancellation_rate")
    sort_key = "cancelled_projects" if metric == "cancelled_projects" else "cancellation_rate"
    parsed_rows.sort(key=lambda x: x[sort_key], reverse=True)
    top_n = int(plan.get("top_n") or 1)
    winners = parsed_rows[:top_n]
    if top_n == 1:
        winner = winners[0]
        if metric == "cancelled_projects":
            reply = (
                f"The sales rep with the most cancelled projects is "
                f"{winner['sales_rep']} with {winner['cancelled_projects']} cancelled projects."
            )
        else:
            reply = (
                f"The sales rep with the highest cancellation rate is "
                f"{winner['sales_rep']} at {winner['cancellation_rate']:.1f}%."
            )
    else:
        if metric == "cancelled_projects":
            lines = [f"{idx + 1}. {item['sales_rep']} - {item['cancelled_projects']} cancelled" for idx, item in enumerate(winners)]
            reply = "Top reps by cancelled projects:\n" + "\n".join(lines)
        else:
            lines = [f"{idx + 1}. {item['sales_rep']} - {item['cancellation_rate']:.1f}%" for idx, item in enumerate(winners)]
            reply = "Top reps by cancellation rate:\n" + "\n".join(lines)

    payload = {
        "mode": "deterministic",
        "metric": metric,
        "entity": "sales_rep",
        "operation": plan.get("operation"),
        "top_n": top_n,
        "results": winners,
    }
    return reply, payload


def _compute_deterministic_rank_result(plan: dict[str, Any], date_from, date_to, user) -> tuple[str, dict[str, Any]] | None:
    raw_context = _gather_chat_context(
        plan.get("scope") or "executive_overview",
        date_from,
        date_to,
        user,
        metric_hint=plan.get("metric_hint"),
    )
    source_key = str(plan.get("source_key") or "")
    rows = raw_context.get(source_key) or []
    if not isinstance(rows, list) or not rows:
        return None

    metric_key = str(plan.get("metric_key") or "count")
    name_key = str(plan.get("name_key") or "name")
    rank_rows: list[dict[str, Any]] = []
    for row in rows:
        name = ""
        if name_key == "sales_rep":
            name = _rep_name(row)
        else:
            name = str(row.get(name_key) or "").strip()
        if not name:
            name = "Unknown"
        raw_value = row.get(metric_key)
        if raw_value is None and metric_key == "count":
            raw_value = row.get("count")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        rank_rows.append({"name": name, "value": value})
    if not rank_rows:
        return None

    reverse = str(plan.get("operation")) != "min"
    rank_rows.sort(key=lambda x: x["value"], reverse=reverse)
    top_n = int(plan.get("top_n") or 1)
    winners = rank_rows[:top_n]
    rank_position = int(plan.get("rank_position") or 0)
    metric_label = str(plan.get("metric_label") or "metric")
    entity = str(plan.get("entity") or "item")
    op = str(plan.get("operation") or "max")
    descriptor = "highest" if op != "min" else "lowest"

    if rank_position > 0:
        idx = rank_position - 1
        if idx >= len(rank_rows):
            return None
        selected = rank_rows[idx]
        value_txt = f"{selected['value']:.1f}" if metric_label.endswith("rate") or metric_label.endswith("percentage") else f"{int(round(selected['value']))}"
        if metric_label.endswith("rate") or metric_label.endswith("percentage"):
            value_txt += "%"
        suffix = "th"
        if rank_position == 1:
            suffix = "st"
        elif rank_position == 2:
            suffix = "nd"
        elif rank_position == 3:
            suffix = "rd"
        reply = f"The {rank_position}{suffix} {entity} by {metric_label} is {selected['name']} at {value_txt}."
        winners = [selected]
    elif top_n == 1:
        winner = winners[0]
        value_txt = f"{winner['value']:.1f}" if metric_label.endswith("rate") or metric_label.endswith("percentage") else f"{int(round(winner['value']))}"
        if metric_label.endswith("rate") or metric_label.endswith("percentage"):
            value_txt += "%"
        reply = f"The {entity} with the {descriptor} {metric_label} is {winner['name']} at {value_txt}."
    else:
        lines = []
        for idx, item in enumerate(winners):
            if metric_label.endswith("rate") or metric_label.endswith("percentage"):
                lines.append(f"{idx + 1}. {item['name']} - {item['value']:.1f}%")
            else:
                lines.append(f"{idx + 1}. {item['name']} - {int(round(item['value']))}")
        reply = f"Top {entity}s by {metric_label}:\n" + "\n".join(lines)

    payload = {
        "mode": "deterministic",
        "scope": plan.get("scope"),
        "metric": metric_key,
        "metricLabel": metric_label,
        "entity": entity,
        "operation": op,
        "top_n": top_n,
        "results": winners,
    }
    return reply, payload


def _chat_cache_key(normalized_question: str, scope: str | None, date_from, date_to, plan_signature: str) -> str:
    raw_key = (
        f"{normalized_question}|{scope or 'none'}|{date_from or 'none'}|"
        f"{date_to or 'none'}|{plan_signature}"
    )
    digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
    return f"insights_chat:deterministic:{digest}"


def _enforce_rate_limits(user_id: int) -> None:
    per_minute = _chat_rate_limit_per_minute()
    minute_key = f"insights_chat:minute:{user_id}:{int(time.time() // 60)}"
    minute_count = cache.get(minute_key, 0)
    if minute_count >= per_minute:
        raise InsightsRateLimitError("Rate limit exceeded. Please wait a minute and try again.")
    cache.set(minute_key, minute_count + 1, timeout=70)

    daily_quota = _chat_daily_quota()
    if daily_quota > 0:
        day_key = f"insights_chat:day:{user_id}:{time.strftime('%Y%m%d')}"
        day_count = cache.get(day_key, 0)
        if day_count >= daily_quota:
            raise InsightsRateLimitError("Daily chat quota reached for this account.")
        cache.set(day_key, day_count + 1, timeout=24 * 3600 + 60)


def _compose_chat_messages(
    user_message: str,
    history: list[dict[str, str]],
    intent_label: str,
    shaped_context: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], str]:
    explanation_note = _explanation_mode_note(user_message)
    intent_note = (
        "User intent: DASHBOARD_QUERY. Use provided dashboard context for data-backed answers only."
        if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY
        else "User intent: non-dashboard. Keep response short and redirect to dashboard analytics when needed."
    )
    context_note = ""
    if shaped_context:
        context_note = f"\nDashboard context:\n{json.dumps(shaped_context, default=str)}"
    system_content = f"{CHAT_SYSTEM_PROMPT}\n\n{intent_note}\n{explanation_note}{context_note}"
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    max_input_tokens = _chat_max_input_tokens()
    while len(messages) > 2:
        token_count = sum(_estimate_tokens(item.get("content", "")) for item in messages)
        if token_count <= max_input_tokens:
            break
        # Preserve system + latest user message, drop oldest history item first.
        del messages[1]
    rendered_prompt = json.dumps(messages, ensure_ascii=True)
    return messages, rendered_prompt


def _chat_response_for_non_dashboard(intent_label: str) -> str | None:
    if intent_label == InsightMessage.IntentLabel.GREETING_SMALLTALK:
        return "Hi! Ask me about Sunbright dashboard metrics like revenue, retention, clean deal rate, or rep/team performance."
    if intent_label == InsightMessage.IntentLabel.OUT_OF_SCOPE:
        return "I can help with Sunbright analytics questions. Try asking about dashboard performance, trends, retention, or action recommendations."
    return None


def _shape_explanation_fallback(reply: str, user_message: str, shaped_context: dict[str, Any] | None) -> str:
    text = (reply or "").strip()
    if not _requires_explanation(user_message):
        return text
    lowered = text.lower()
    unavailable_markers = ("not available", "do not have", "cannot determine", "insufficient")
    if not any(marker in lowered for marker in unavailable_markers):
        return text

    reasons = []
    if shaped_context:
        reasons = list((shaped_context.get("top_cancellation_reasons") or [])[:3])
    reasons_line = ", ".join(reasons) if reasons else "pricing concerns, buyer hesitation, and lead quality variation"

    if shaped_context and shaped_context.get("pipeline_averages"):
        avg = shaped_context.get("pipeline_averages") or {}
        return (
            "Based on available pipeline velocity data, install-cycle timing appears to be the key bottleneck area. "
            f"Current averages show about {avg.get('avgDaysToInstall')} days to install and "
            f"{avg.get('avgDaysToPtoSubmitted')} days to PTO submission, with roughly "
            f"{avg.get('avgDaysToPermit')} days to permit approval. "
            "Treat this as directional analysis from sampled records; for deeper diagnostics, add stage-level completion coverage and rep-level milestone quality checks."
        )

    return (
        "Based on available dashboard data, common drivers appear to include "
        f"{reasons_line}. While rep-specific root causes are not directly captured, "
        "these aggregate patterns likely influence current outcomes. "
        "Use this as directional guidance and validate with rep-level notes or richer stage-level fields."
    )


def chat_with_insights_assistant(
    *,
    user,
    message: str,
    conversation_id: int | None = None,
    date_from=None,
    date_to=None,
) -> dict[str, Any]:
    text = (message or "").strip()
    print("[insights-chat] incoming request", {"user_id": getattr(user, "id", None), "conversation_id": conversation_id})
    if not text:
        raise InsightsLLMError("Message cannot be empty.")
    if len(text) > 4000:
        raise InsightsLLMError("Message is too long.")
    print("[insights-chat] message accepted", {"length": len(text), "preview": text[:120]})

    _enforce_rate_limits(user.id)
    print("[insights-chat] rate limit check passed", {"user_id": user.id})
    if conversation_id:
        conversation = InsightConversation.objects.filter(id=conversation_id, user=user, deleted_at__isnull=True).first()
        if not conversation:
            raise InsightsConversationError("Conversation not found.")
        print("[insights-chat] using existing conversation", {"conversation_id": conversation.id})
    else:
        conversation = InsightConversation.objects.create(
            user=user,
            title=(text[:80] + "...") if len(text) > 80 else text,
            date_from=date_from,
            date_to=date_to,
            scope_snapshot={"is_staff": bool(user.is_staff)},
        )
        print("[insights-chat] created new conversation", {"conversation_id": conversation.id})

    intent_label = _classify_intent(text)
    safety_flags = _detect_safety_flags(text)
    print(
        "[insights-chat] classified intent",
        {"intent_label": intent_label, "possible_prompt_injection": safety_flags.get("possiblePromptInjection", False)},
    )
    shaped_context = None
    data_scope = None
    metric_hint = None
    normalized_question = _normalize_question(text)
    deterministic_hit = False
    cache_hit = False
    llm_used = False
    if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
        data_scope = _detect_dashboard_scope(text)
        metric_hint = _detect_metric_hint(text, data_scope)
        print("[insights-chat] dashboard scope selected", {"scope": data_scope})
        print("[insights-chat] metric hint", {"metric_hint": metric_hint})
    deterministic_plan = _deterministic_rank_plan(text, intent_label, data_scope, metric_hint)
    if not deterministic_plan:
        deterministic_plan = _deterministic_cancellation_rep_plan(text, intent_label, data_scope)
    if deterministic_plan:
        plan_metric = str(
            deterministic_plan.get("metric")
            or deterministic_plan.get("metric_key")
            or deterministic_plan.get("metricLabel")
            or "metric"
        )
        plan_entity = str(deterministic_plan.get("entity") or "entity")
        plan_operation = str(deterministic_plan.get("operation") or "op")
        plan_top_n = int(deterministic_plan.get("top_n") or 1)
        plan_signature = f"{plan_metric}:{plan_entity}:{plan_operation}:{plan_top_n}"
        cache_key = _chat_cache_key(
            normalized_question,
            data_scope,
            date_from or conversation.date_from,
            date_to or conversation.date_to,
            plan_signature,
        )
        cached_payload = cache.get(cache_key)
        if cached_payload:
            cache_hit = True
            print("[insights-chat] cache_hit", {"cache_hit": cache_hit, "cache_key": cache_key})
            deterministic_hit = True
            context_meta = _build_context_meta(
                intent_label,
                date_from or conversation.date_from,
                date_to or conversation.date_to,
                None,
                data_scope=data_scope,
            )
            context_meta.update({"deterministic_hit": True, "cache_hit": True, "llm_used": False})
            user_row = InsightMessage.objects.create(
                conversation=conversation,
                role=InsightMessage.Role.USER,
                content=text,
                intent_label=intent_label,
                context_meta=context_meta,
                safety_flags=safety_flags,
            )
            print("[insights-chat] user message saved", {"message_id": user_row.id})
            assistant_row = InsightMessage.objects.create(
                conversation=conversation,
                role=InsightMessage.Role.ASSISTANT,
                content=str(cached_payload.get("reply") or ""),
                intent_label=intent_label,
                context_meta=context_meta,
            )
            print("[insights-chat] deterministic_hit", {"deterministic_hit": True, "cache_hit": True, "llm_used": False})
            return {
                "conversationId": conversation.id,
                "messageId": assistant_row.id,
                "reply": assistant_row.content,
                "intentLabel": intent_label,
                "contextMeta": context_meta,
            }
        print("[insights-chat] cache_hit", {"cache_hit": False, "cache_key": cache_key})
        deterministic_result = _compute_deterministic_rank_result(
            deterministic_plan,
            date_from or conversation.date_from,
            date_to or conversation.date_to,
            user,
        )
        if deterministic_result is None and deterministic_plan.get("scope") == "retention":
            deterministic_result = _compute_cancellation_rep_result(
                deterministic_plan,
                date_from or conversation.date_from,
                date_to or conversation.date_to,
                user,
            )
        if deterministic_result:
            deterministic_hit = True
            reply_text, deterministic_payload = deterministic_result
            cache.set(
                cache_key,
                {"reply": reply_text, "payload": deterministic_payload},
                timeout=300,
            )
            context_meta = _build_context_meta(
                intent_label,
                date_from or conversation.date_from,
                date_to or conversation.date_to,
                None,
                data_scope=data_scope,
            )
            context_meta.update(
                {
                    "deterministic_hit": True,
                    "cache_hit": False,
                    "llm_used": False,
                    "deterministic_payload": deterministic_payload,
                }
            )
            user_row = InsightMessage.objects.create(
                conversation=conversation,
                role=InsightMessage.Role.USER,
                content=text,
                intent_label=intent_label,
                context_meta=context_meta,
                safety_flags=safety_flags,
            )
            print("[insights-chat] user message saved", {"message_id": user_row.id})
            assistant_row = InsightMessage.objects.create(
                conversation=conversation,
                role=InsightMessage.Role.ASSISTANT,
                content=reply_text,
                intent_label=intent_label,
                context_meta=context_meta,
            )
            print("[insights-chat] deterministic_hit", {"deterministic_hit": True, "cache_hit": False, "llm_used": False})
            return {
                "conversationId": conversation.id,
                "messageId": assistant_row.id,
                "reply": assistant_row.content,
                "intentLabel": intent_label,
                "contextMeta": context_meta,
            }
    if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
        raw_context = _gather_chat_context(
            data_scope or "executive_overview",
            date_from or conversation.date_from,
            date_to or conversation.date_to,
            user,
            metric_hint=metric_hint,
        )
        broad_context = _shape_dashboard_context(raw_context, data_scope or "executive_overview")
        shaped_context = _filter_context_for_llm(broad_context, data_scope, metric_hint)
        print(
            "[insights-chat] context prepared",
            {
                "scope": data_scope or "executive_overview",
                "shape_version": (shaped_context or {}).get("shape_version"),
                "keys": list((shaped_context or {}).keys()),
            },
        )
    context_meta = _build_context_meta(
        intent_label,
        date_from or conversation.date_from,
        date_to or conversation.date_to,
        shaped_context,
        data_scope=data_scope,
    )
    context_meta.update({"deterministic_hit": deterministic_hit, "cache_hit": cache_hit, "llm_used": llm_used})
    print("[insights-chat] context meta", context_meta)

    user_row = InsightMessage.objects.create(
        conversation=conversation,
        role=InsightMessage.Role.USER,
        content=text,
        intent_label=intent_label,
        context_meta=context_meta,
        safety_flags=safety_flags,
    )
    print("[insights-chat] user message saved", {"message_id": user_row.id})

    static_reply = _chat_response_for_non_dashboard(intent_label)
    if static_reply:
        print("[insights-chat] static response path", {"intent_label": intent_label})
        assistant_row = InsightMessage.objects.create(
            conversation=conversation,
            role=InsightMessage.Role.ASSISTANT,
            content=static_reply,
            intent_label=intent_label,
            context_meta=context_meta,
        )
        return {
            "conversationId": conversation.id,
            "messageId": assistant_row.id,
            "reply": static_reply,
            "intentLabel": intent_label,
            "contextMeta": context_meta,
        }

    history = _history_messages(conversation)
    print("[insights-chat] loaded history", {"history_count": len(history)})
    messages, rendered_prompt = _compose_chat_messages(text, history[:-1], intent_label, shaped_context)
    print("[insights-chat] prompt composed", {"message_count": len(messages), "prompt_chars": len(rendered_prompt)})
    print("[insights-chat] llm prompt messages", messages)
    start = time.perf_counter()
    try:
        print("[insights-chat] calling llm")
        llm_used = True
        context_meta["llm_used"] = True
        llm_output = _invoke_llm_chat(
            messages=messages,
            max_tokens=_chat_max_output_tokens(),
            retry_once=True,
        )
        reply = _shape_explanation_fallback(llm_output["content"], text, shaped_context).strip()
        print("[insights-chat] llm raw response", llm_output["content"])
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(
            "[insights-chat] llm success",
            {
                "latency_ms": latency_ms,
                "token_input": llm_output["usage"]["input"],
                "token_output": llm_output["usage"]["output"],
                "finish_reason": llm_output["finish_reason"],
            },
        )
        print("[insights-chat] flow flags", {"deterministic_hit": deterministic_hit, "cache_hit": cache_hit, "llm_used": llm_used})
        assistant_row = InsightMessage.objects.create(
            conversation=conversation,
            role=InsightMessage.Role.ASSISTANT,
            content=reply or CHAT_FALLBACK_MESSAGE,
            rendered_prompt=rendered_prompt,
            intent_label=intent_label,
            context_meta=context_meta,
            provider=_forge_chat_url(),
            model=_llm_model(),
            latency_ms=latency_ms,
            token_input=llm_output["usage"]["input"],
            token_output=llm_output["usage"]["output"],
            finish_reason=llm_output["finish_reason"],
        )
        return {
            "conversationId": conversation.id,
            "messageId": assistant_row.id,
            "reply": assistant_row.content,
            "intentLabel": intent_label,
            "contextMeta": context_meta,
        }
    except InsightsLLMError as exc:
        logger.exception("Insights chat LLM failure")
        print("[insights-chat] llm failure, returning fallback", {"error": str(exc)})
        print("[insights-chat] flow flags", {"deterministic_hit": deterministic_hit, "cache_hit": cache_hit, "llm_used": llm_used})
        assistant_row = InsightMessage.objects.create(
            conversation=conversation,
            role=InsightMessage.Role.ASSISTANT,
            content=CHAT_FALLBACK_MESSAGE,
            rendered_prompt=rendered_prompt,
            intent_label=intent_label,
            context_meta=context_meta,
            provider=_forge_chat_url(),
            model=_llm_model(),
            error_payload={"error": str(exc)},
        )
        return {
            "conversationId": conversation.id,
            "messageId": assistant_row.id,
            "reply": CHAT_FALLBACK_MESSAGE,
            "intentLabel": intent_label,
            "contextMeta": context_meta,
        }


def list_insight_conversations(user) -> list[InsightConversation]:
    return list(InsightConversation.objects.filter(user=user, deleted_at__isnull=True).order_by("-updated_at"))


def list_insight_messages(user, conversation_id: int) -> list[InsightMessage]:
    conversation = InsightConversation.objects.filter(id=conversation_id, user=user, deleted_at__isnull=True).first()
    if not conversation:
        raise InsightsConversationError("Conversation not found.")
    return list(conversation.messages.filter(deleted_at__isnull=True).order_by("created_at"))


def generate_dashboard_insights(date_from, date_to, user) -> dict[str, Any]:
    ctx = gather_insights_context(date_from, date_to, user)
    if not _forge_api_key():
        return _heuristic_insights_from_context(ctx)

    system = "You are a senior solar industry data analyst. Provide analysis in valid JSON format only."
    user_prompt = _build_user_prompt(ctx)
    content = _invoke_llm(system, user_prompt)
    try:
        out = json.loads(content)
    except json.JSONDecodeError as e:
        raise InsightsLLMError("Failed to parse structured LLM output as JSON") from e
    if not isinstance(out, dict):
        raise InsightsLLMError("LLM returned a non-object JSON root")
    out["insightSource"] = "llm"
    return out
