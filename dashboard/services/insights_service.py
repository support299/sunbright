"""
AI dashboard insights via an OpenAI-compatible chat completions API (same pattern as sunbright-dashboard
`server/_core/llm.ts`: Manus Forge + Gemini by default).
"""
import json
import logging
import os
import re
import time
import difflib
import hashlib
import urllib.error
import urllib.request
from datetime import date, timedelta
from typing import Any

from django.core.cache import cache
from django.utils import timezone

from dashboard.models import InsightConversation, InsightMessage
from dashboard.services.analytics_service import (
    get_clean_deals_bundle,
    get_cx_bundle,
    get_manager_bundle,
    get_performance_bundle,
    get_pipeline_bundle,
    get_retention_bundle,
    get_role_performance_bundle,
)
from dashboard.services.project_service import (
    base_queryset,
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
    return 1200


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


def _chat_dim_sig_for_vocab(
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
) -> str:
    parts = []
    if installer and str(installer).strip():
        parts.append(f"i:{str(installer).strip().lower()}")
    if sales_team and str(sales_team).strip():
        parts.append(f"t:{str(sales_team).strip().lower()}")
    if lead_source and str(lead_source).strip():
        parts.append(f"l:{str(lead_source).strip().lower()}")
    if project_manager and str(project_manager).strip():
        parts.append(f"p:{str(project_manager).strip().lower()}")
    if market and str(market).strip():
        parts.append(f"k:{str(market).strip().lower()}")
    rk = (rep_kind or "").strip().lower() if rep_kind else ""
    rn = (rep_name or "").strip() if rep_name else ""
    if rk in ("sales_rep", "setter") and rn:
        parts.append(f"r:{rk}:{rn.lower()}")
    return "|".join(sorted(parts)) if parts else "all"


def _load_chat_rep_setter_vocab(
    user,
    date_from,
    date_to,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
) -> list[str]:
    """
    Distinct sales_rep / setter names in the current filter window so natural-language
    questions ("what about Eddie Lopez") can be classified as dashboard queries.
    Cached briefly to avoid repeated DISTINCT scans during a chat session.
    """
    dim_sig = _chat_dim_sig_for_vocab(
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
        market=market,
        rep_kind=rep_kind,
        rep_name=rep_name,
    )
    raw_key = f"{getattr(user, 'id', 0)}|{date_from}|{date_to}|{dim_sig}"
    digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
    cache_key = f"insights_chat:vocab:{digest}"
    cached = cache.get(cache_key)
    if isinstance(cached, list):
        return cached

    qs = base_queryset(
        date_from,
        date_to,
        user,
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
        market=market,
        rep_kind=rep_kind,
        rep_name=rep_name,
    )
    names: set[str] = set()
    for field in ("sales_rep", "setter"):
        for val in (
            qs.exclude(**{f"{field}__exact": ""})
            .values_list(field, flat=True)
            .distinct()[:500]
        ):
            n = str(val or "").strip().lower()
            if len(n) >= 4:
                names.add(re.sub(r"\s+", " ", n))
    out = sorted(names)
    cache.set(cache_key, out, timeout=300)
    return out


def _vocab_hits_exact(message: str, vocab: list[str]) -> list[str]:
    if not message or not vocab:
        return []
    normalized = re.sub(r"[^\w\s]", " ", message.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return []
    hits: list[str] = []
    seen: set[str] = set()
    for name in sorted(set(vocab), key=len, reverse=True):
        n = str(name).strip().lower()
        if len(n) < 4:
            continue
        if n in normalized and n not in seen:
            hits.append(n)
            seen.add(n)
    return hits


def _fuzzy_vocab_hits_in_message(message: str, vocab: list[str], *, ratio_min: float = 0.84) -> list[str]:
    """
    Match two-word names with small typos (e.g. 'eddi lopez' vs 'eddie lopez') using sequence similarity.
    """
    if not message or not vocab:
        return []
    normalized = re.sub(r"[^\w\s]", " ", message.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = normalized.split()
    if len(tokens) < 2:
        return []
    hits: list[str] = []
    seen: set[str] = set()
    for i in range(len(tokens) - 1):
        candidate = f"{tokens[i]} {tokens[i + 1]}"
        if len(candidate) < 6:
            continue
        for name in vocab:
            n = str(name).strip().lower()
            if len(n) < 6 or n in seen:
                continue
            if difflib.SequenceMatcher(None, candidate, n).ratio() >= ratio_min:
                hits.append(n)
                seen.add(n)
    return hits


def _message_matches_vocab_name(message: str, vocab: list[str]) -> bool:
    if not message or not vocab:
        return False
    if _vocab_hits_exact(message, vocab):
        return True
    return bool(_fuzzy_vocab_hits_in_message(message, vocab))


def _vocab_hits_in_message(message: str, vocab: list[str]) -> list[str]:
    """Return distinct vocab names (lowercase): exact substring match, then fuzzy two-word match."""
    exact = _vocab_hits_exact(message, vocab)
    if exact:
        return exact
    return _fuzzy_vocab_hits_in_message(message, vocab)


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


def gather_insights_context(
    date_from,
    date_to,
    user,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
) -> dict[str, Any]:
    dim_kw = dict(
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
        market=market,
        rep_kind=rep_kind,
        rep_name=rep_name,
    )
    perf = get_performance_bundle(date_from, date_to, user, **dim_kw)
    cx = get_cx_bundle(date_from, date_to, user, **dim_kw)
    retention = get_retention_bundle(date_from, date_to, user, **dim_kw)
    clean = get_clean_deals_bundle(date_from, date_to, user, **dim_kw)
    return {
        "overview": get_overview_metrics(date_from, date_to, user, **dim_kw),
        "repPerformance": (perf.get("reps") or [])[:15],
        "teamPerformance": (perf.get("teams") or [])[:15],
        "cleanDealsByRep": (clean.get("byRep") or [])[:15],
        "cleanDealPortfolio": _shape_clean_deal_portfolio_for_chat(clean.get("analysis") or []),
        "retentionByRep": (retention.get("byRep") or [])[:15],
        "retentionByLeadSource": (retention.get("byLeadSource") or [])[:25],
        "cancellationReasons": get_cancellation_reasons_breakdown(date_from, date_to, user, **dim_kw)[:25],
        "onHoldReasons": get_on_hold_reasons_breakdown(date_from, date_to, user, **dim_kw)[:25],
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
- You receive a **wide dashboard context pack** (overview, rep/setter samples, retention/cancellation headlines, clean-deal portfolio, optional focus rows, thread previews). `scope_hint` is only a weak hint from keywords—not an exhaustive list of what matters; infer what the user cares about from their message and prior turns.
- **`requested_window`** describes the date range already applied to this payload (including conversational phrases like “this month”). Treat it as authoritative: never say the dashboard “has no month filter” or that you cannot apply a timeframe when `requested_window` is present. Open with a one-line timeframe anchor only when helpful (e.g. “For May 1–14…”).
- If **`comparison_window`** is present, give directional month-over-week or period-vs-period commentary using both overviews; avoid claiming you lack history when that block is populated.
- Only use provided dashboard data when answering analytics questions. Do not invent metrics.
- **Tone:** answer with what the data *does* show first—synthesis, momentum, trade-offs. Do not lead with disclaimers, “I cannot,” or “I do not have access.” If something is genuinely missing, mention it briefly after the substantive answer.
- **Avoid repetitive KPI dumps:** do not restate cancellation %, retention %, and clean-deal stats every turn unless the user asked about them or they are central to the question. Prefer the metrics most relevant to the latest user message and `continuity.active_topic`.
- If `ambiguous_entities` is non-empty, two or more people may match the user's wording: ask **one** short clarifying question (name + role) before picking metrics; do not guess.
- If some slice is missing, still deliver the best partial read, then note the gap in one short phrase.
- Use prior turns plus `thread_hints` and `continuity` for follow-ups (“what about X”, “same for last month”).
- When `rep_focus` or `setter_role_metrics_focus` is non-empty, lead with that person's numbers (doors, contacts, appointments, sit-downs, sitDownRate, etc.). Do not claim you have no data if those arrays are populated.
- If focus arrays are empty and the user named someone, say they may be outside current filters and suggest widening dates/team/manager—do not invent numbers.
- Use `clean_deal_portfolio` for clean vs non-clean cancellation comparisons when present.
- Prefer short paragraphs and bullets with concrete numbers. Avoid robotic canned refusals; stay conversational.
- If the user asks something clearly unrelated to Sunbright analytics, redirect briefly to dashboard topics."""

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


def _classify_intent(message: str, *, safety_flags: dict[str, Any] | None = None) -> str:
    """
    Phase 1: default to analytics conversation for this product surface.
    Only refuse on prompt-injection heuristics; use GREETING for short pleasantries.
    """
    text = (message or "").strip()
    lowered = text.lower()
    if not lowered:
        return InsightMessage.IntentLabel.GREETING_SMALLTALK

    if safety_flags and safety_flags.get("possiblePromptInjection"):
        return InsightMessage.IntentLabel.OUT_OF_SCOPE

    if len(lowered) <= 48 and re.match(
        r"^(hi|hello|hey|good morning|good afternoon|good evening)\b",
        lowered,
    ):
        return InsightMessage.IntentLabel.GREETING_SMALLTALK
    if len(lowered) <= 32 and lowered in ("thanks", "thank you", "thank you!", "ty", "ok", "okay", "bye", "goodbye"):
        return InsightMessage.IntentLabel.GREETING_SMALLTALK

    return InsightMessage.IntentLabel.DASHBOARD_QUERY


def _today_in_app_tz() -> date:
    return timezone.localdate()


def _weekday_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _first_day_of_month(d: date) -> date:
    return date(d.year, d.month, 1)


def _last_calendar_month_range(today: date) -> tuple[date, date]:
    first_this = _first_day_of_month(today)
    end = first_this - timedelta(days=1)
    start = _first_day_of_month(end)
    return start, end


def _quarter_index(month: int) -> int:
    return (month - 1) // 3 + 1


def _quarter_range(year: int, quarter: int) -> tuple[date, date]:
    starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    sm, sd = starts[quarter]
    start = date(year, sm, sd)
    if quarter == 4:
        end = date(year, 12, 31)
    else:
        nsm = starts[quarter + 1][0]
        end = date(year, nsm, 1) - timedelta(days=1)
    return start, end


def _previous_calendar_quarter_range(today: date) -> tuple[date, date]:
    cq = _quarter_index(today.month)
    cy = today.year
    if cq == 1:
        return _quarter_range(cy - 1, 4)
    return _quarter_range(cy, cq - 1)


def _human_date_window(d0: date, d1: date) -> str:
    def month_day(d: date) -> str:
        return f"{d.strftime('%b')} {d.day}, {d.year}"

    if d0 == d1:
        return month_day(d0)
    if d0.year == d1.year and d0.month == d1.month:
        return f"{d0.strftime('%b')} {d0.day}–{d1.day}, {d1.year}"
    if d0.year == d1.year:
        return f"{d0.strftime('%b')} {d0.day}–{month_day(d1)}"
    return f"{month_day(d0)}–{month_day(d1)}"


def _infer_active_topic(message: str) -> str | None:
    low = (message or "").lower()
    if any(t in low for t in ("cancel", "cancellation", "churn", "retention")):
        return "retention_health"
    if any(t in low for t in ("pipeline", "contract value", "active pipeline", "revenue", "booking")):
        return "sales_pipeline"
    if "sales" in low and "sales rep" not in low and "salesrep" not in low.replace(" ", ""):
        return "sales_pipeline"
    if "team" in low and "setter" not in low:
        return "team_performance"
    if any(t in low for t in ("setter", "door", "knock", "appointment", "sit down", "sitdown", "manager")):
        return "field_ops"
    if any(t in low for t in ("clean deal", "deal quality", "portfolio")):
        return "deal_quality"
    if any(t in low for t in ("sales rep", "rep ", " rep", "closer", "performance")):
        return "rep_performance"
    return None


def _mentions_this_and_last_month(low: str) -> bool:
    has_this = any(
        p in low
        for p in (
            "this month",
            "current month",
            "month to date",
            " month-to-date",
            "mtd",
            "so far this month",
        )
    )
    has_last = "last month" in low or "previous month" in low or "prior month" in low
    if has_this and has_last:
        return True
    if has_last and ("compare" in low or " vs " in low or " versus " in low or " against " in low):
        return "month" in low or "mtd" in low
    return False


def _mentions_this_and_last_week(low: str) -> bool:
    has_this = "this week" in low or "week to date" in low or "wtd" in low or "so far this week" in low
    has_last = "last week" in low or "previous week" in low
    if has_this and has_last:
        return True
    if has_last and ("compare" in low or " vs " in low or " versus " in low):
        return "week" in low or "wtd" in low
    return False


def _extract_conversation_filters(
    message: str,
    existing_filters: dict[str, Any],
    session_state: dict[str, Any] | None = None,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """
    Phase 1.5: infer date windows and light topic hints from natural language.
    Does not replace dimension filters (team/installer) from the request UI.
    """
    today = today or _today_in_app_tz()
    low = (message or "").lower().strip()
    session_state = session_state or {}
    out: dict[str, Any] = {
        "date_from": None,
        "date_to": None,
        "comparison": None,
        "active_timeframe": None,
        "active_topic": _infer_active_topic(message),
        "human_window": None,
        "temporal_matched": False,
    }

    def mark(df: date, dt: date, tf_id: str) -> None:
        out["date_from"] = df
        out["date_to"] = dt
        out["active_timeframe"] = tf_id
        out["human_window"] = _human_date_window(df, dt)
        out["temporal_matched"] = True

    # --- Comparisons (check before single-window phrases) ---
    if _mentions_this_and_last_month(low):
        a0, a1 = _first_day_of_month(today), today
        b0, b1 = _last_calendar_month_range(today)
        out["comparison"] = {"date_from": b0, "date_to": b1, "label": "last_month", "human_window": _human_date_window(b0, b1)}
        mark(a0, a1, "this_month_vs_last_month")
        return out

    if _mentions_this_and_last_week(low):
        mon = _weekday_monday(today)
        prev_mon = mon - timedelta(days=7)
        prev_end = mon - timedelta(days=1)
        out["comparison"] = {
            "date_from": prev_mon,
            "date_to": prev_end,
            "label": "last_week",
            "human_window": _human_date_window(prev_mon, prev_end),
        }
        mark(mon, today, "this_week_vs_last_week")
        return out

    # --- Single-window phrases (order: specific → broad) ---
    if re.search(r"\b(yesterday)\b", low):
        mark(today - timedelta(days=1), today - timedelta(days=1), "yesterday")
        return out

    if re.search(r"\b(last|past)\s+7\s+days?\b", low):
        mark(today - timedelta(days=6), today, "last_7_days")
        return out

    if re.search(r"\b(last|past)\s+30\s+days?\b", low) or "rolling 30" in low:
        mark(today - timedelta(days=29), today, "last_30_days")
        return out

    if (
        re.search(r"\b(this week|week to date|wtd|so far this week)\b", low)
        or "week so far" in low
    ):
        mark(_weekday_monday(today), today, "this_week")
        return out

    if "last week" in low or "previous week" in low:
        mon = _weekday_monday(today)
        start = mon - timedelta(days=7)
        end = mon - timedelta(days=1)
        mark(start, end, "last_week")
        return out

    if (
        re.search(r"\b(this month|current month|month to date|mtd|so far this month)\b", low)
        or "month so far" in low
        or ("month" in low and "only" in low and ("this" in low or "current" in low))
        or ("filter" in low and "this month" in low)
    ):
        mark(_first_day_of_month(today), today, "this_month")
        return out

    if "last month" in low or "previous month" in low or "prior month" in low:
        s, e = _last_calendar_month_range(today)
        mark(s, e, "last_month")
        return out

    if re.search(r"\b(year to date|ytd)\b", low):
        mark(date(today.year, 1, 1), today, "ytd")
        return out

    if re.search(r"\b(last quarter|previous quarter)\b", low):
        s, e = _previous_calendar_quarter_range(today)
        mark(s, e, "last_quarter")
        return out

    m = re.search(r"\bq([1-4])\b(?:\s*,?\s*(\d{4}))?", low)
    if m:
        q = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else today.year
        s, e = _quarter_range(year, q)
        if s > today:
            year -= 1
            s, e = _quarter_range(year, q)
        mark(s, e, f"Q{q}_{year}")
        return out

    # Explicit ISO-ish dates in message (light): "from 2025-01-01" — skip for Phase 1.5 complexity

    return out


def _resolve_effective_chat_dates(
    extraction: dict[str, Any],
    session_state: dict[str, Any],
    base_from,
    base_to,
) -> tuple[Any, Any, dict[str, Any]]:
    """Pick primary date window: new utterance beats session sticky beats request/conversation."""
    trace: dict[str, Any] = {"source": "request_or_conversation", "utterance_override": False}
    if extraction.get("temporal_matched") and extraction.get("date_from") and extraction.get("date_to"):
        trace["source"] = "utterance"
        trace["utterance_override"] = True
        return extraction["date_from"], extraction["date_to"], trace
    sf = session_state.get("effective_date_from")
    st = session_state.get("effective_date_to")
    if sf and st:
        try:
            trace["source"] = "session_sticky"
            return date.fromisoformat(str(sf)), date.fromisoformat(str(st)), trace
        except ValueError:
            pass
    return base_from, base_to, trace


def _continuity_block_for_llm(session_state: dict[str, Any]) -> dict[str, Any] | None:
    if not session_state:
        return None
    block: dict[str, Any] = {}
    if session_state.get("active_timeframe"):
        block["prior_active_timeframe"] = session_state.get("active_timeframe")
    if session_state.get("active_topic"):
        block["prior_active_topic"] = session_state.get("active_topic")
    if session_state.get("last_human_window"):
        block["prior_window_human"] = session_state.get("last_human_window")
    return block or None


def _persist_chat_session_state(
    conversation: InsightConversation,
    *,
    extraction: dict[str, Any],
    eff_from,
    eff_to,
    date_trace: dict[str, Any],
) -> None:
    snap: dict[str, Any] = dict(conversation.scope_snapshot or {})
    prev = snap.get("chat_session") if isinstance(snap.get("chat_session"), dict) else {}
    ch: dict[str, Any] = {
        "active_timeframe": extraction.get("active_timeframe") or prev.get("active_timeframe"),
        "active_topic": extraction.get("active_topic") or prev.get("active_topic"),
        "last_human_window": extraction.get("human_window") or prev.get("last_human_window"),
        "last_date_source": date_trace.get("source"),
    }
    if eff_from is not None and eff_to is not None:
        ch["effective_date_from"] = str(eff_from)
        ch["effective_date_to"] = str(eff_to)
    else:
        ch["effective_date_from"] = prev.get("effective_date_from")
        ch["effective_date_to"] = prev.get("effective_date_to")
    if extraction.get("comparison"):
        c = extraction["comparison"]
        ch["last_comparison"] = {
            "label": c.get("label"),
            "date_from": str(c.get("date_from")) if c.get("date_from") else None,
            "date_to": str(c.get("date_to")) if c.get("date_to") else None,
        }
    snap["chat_session"] = ch
    conversation.scope_snapshot = snap
    if eff_from is not None and eff_to is not None:
        conversation.date_from = eff_from
        conversation.date_to = eff_to
    conversation.save(update_fields=["scope_snapshot", "date_from", "date_to", "updated_at"])


def _slim_overview_for_compare(ov: dict[str, Any]) -> dict[str, Any]:
    return {
        "totalProjects": ov.get("totalProjects"),
        "activeProjects": ov.get("activeProjects"),
        "cancelledProjects": ov.get("cancelledProjects"),
        "cleanDealPct": ov.get("cleanDealPct"),
        "cancellationRate": ov.get("cancellationRate"),
        "netRetentionRate": ov.get("netRetentionRate"),
        "activePipelineValue": ov.get("activePipelineValue"),
    }


def _detect_dashboard_scope(message: str) -> str:
    text = message.lower()
    # Clean vs non-clean + cancellation compares portfolio buckets (not generic cancellation list).
    if "clean" in text and any(
        token in text for token in ("cancel", "cancellation", "churn", "versus", " vs ", "ratio", "non-clean", "non clean")
    ):
        return "clean_deals"
    if any(token in text for token in ("manager performance", "door stats")):
        return "manager"
    if re.search(r"\b(managers?)\b", text) and any(
        token in text
        for token in (
            "sit down",
            "qualified sit down",
            "closing rate",
            "performance",
            "appointment",
            "rep",
            "team",
            "show",
        )
    ):
        return "manager"
    if re.search(r"\b(managers?)\b", text):
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
    if any(
        token in text
        for token in (
            "rep",
            "team",
            "sales",
            "performance",
            "setter",
            "closer",
            "sit down",
            "qualified sit down",
            "appointment",
            "doors",
            "knock",
            "closing rate",
        )
    ):
        return "performance"
    if any(token in text for token in ("clean deal", "clean deals", "deal quality")):
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


def _shape_clean_deal_portfolio_for_chat(analysis: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate clean vs not-clean cancellation for LLM context."""
    out: list[dict[str, Any]] = []
    for row in analysis:
        if not isinstance(row, dict):
            continue
        total = int(row.get("total") or 0)
        cancelled = int(row.get("cancelled") or 0)
        is_clean = bool(row.get("isCleanDeal"))
        out.append(
            {
                "bucket": "clean" if is_clean else "not_clean",
                "label": "clean deals" if is_clean else "not clean deals",
                "totalProjects": total,
                "cancelledProjects": cancelled,
                "cancellationRatePct": round(100.0 * cancelled / total, 1) if total else 0.0,
            }
        )
    return out


def _gather_chat_context(
    scope: str,
    date_from,
    date_to,
    user,
    metric_hint: str | None = None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
    message: str | None = None,
    matched_vocab_hits: list[str] | None = None,
) -> dict[str, Any]:
    dim_kw = dict(
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
        market=market,
        rep_kind=rep_kind,
        rep_name=rep_name,
    )
    context: dict[str, Any] = {
        "overview": get_overview_metrics(date_from, date_to, user, **dim_kw),
    }
    if scope == "performance":
        perf = get_performance_bundle(date_from, date_to, user, **dim_kw)
        reps_all = perf.get("reps") or []
        context["repPerformance"] = reps_all[:12]
        context["teamPerformance"] = (perf.get("teams") or [])[:12]
        hits = matched_vocab_hits or []
        msg_l = (message or "").lower()
        if hits:
            hit_set = {h.strip().lower() for h in hits if str(h).strip()}
            focus_reps = [r for r in reps_all if str(r.get("salesRep") or "").strip().lower() in hit_set]
            if not focus_reps:
                focus_reps = [
                    r for r in reps_all if any(h in str(r.get("salesRep") or "").lower() for h in hit_set)
                ]
            if focus_reps:
                context["repPerformanceFocus"] = focus_reps[:6]
        if hits or "setter" in msg_l or "appointment" in msg_l:
            try:
                role = get_role_performance_bundle(date_from, date_to, user, **dim_kw)
                setters = role.get("setters") or []
                if hits:
                    hit_set = {h.strip().lower() for h in hits if str(h).strip()}
                    focus_s = [s for s in setters if str(s.get("agent") or "").strip().lower() in hit_set]
                    if not focus_s:
                        focus_s = [
                            s for s in setters if any(h in str(s.get("agent") or "").lower() for h in hit_set)
                        ]
                    if focus_s:
                        context["setterPerformanceFocus"] = focus_s[:4]
                if not context.get("setterPerformanceFocus") and (
                    "setter" in msg_l or "appointment" in msg_l
                ):
                    context["setterPerformanceSample"] = sorted(
                        setters, key=lambda x: -int(x.get("appointments") or 0)
                    )[:10]
            except Exception as exc:
                logger.warning("role performance bundle for chat failed: %s", exc)
    elif scope == "retention":
        retention = get_retention_bundle(date_from, date_to, user, **dim_kw)
        context["retentionByRep"] = (retention.get("byRep") or [])[:12]
        context["retentionByLeadSource"] = (retention.get("byLeadSource") or [])[:16]
        context["cancellationReasons"] = get_cancellation_reasons_breakdown(date_from, date_to, user, **dim_kw)[:12]
        context["onHoldReasons"] = get_on_hold_reasons_breakdown(date_from, date_to, user, **dim_kw)[:12]
    elif scope == "cx":
        cx = get_cx_bundle(date_from, date_to, user, **dim_kw)
        context["cxOverview"] = cx.get("overview") or {}
        context["cxByInstaller"] = (cx.get("byInstaller") or [])[:12]
    elif scope == "clean_deals":
        clean = get_clean_deals_bundle(date_from, date_to, user, **dim_kw)
        context["cleanDealsByRep"] = (clean.get("byRep") or [])[:12]
        analysis = clean.get("analysis") or []
        context["cleanDealPortfolio"] = _shape_clean_deal_portfolio_for_chat(analysis)
    elif scope == "executive_overview":
        if metric_hint == "pipeline_velocity":
            pipeline = get_pipeline_bundle(date_from, date_to, user, **dim_kw)
            velocity_rows = pipeline.get("velocity") or []
            averages = pipeline.get("averages") or {}
            context["pipelineAverages"] = averages
            context["pipelineVelocitySample"] = velocity_rows[:20]
        context["categoryBreakdown"] = get_category_breakdown(date_from, date_to, user, **dim_kw)[:10]
    elif scope == "manager":
        manager = get_manager_bundle(date_from, date_to, user, **dim_kw)
        context["managerOverview"] = manager.get("managerOverview") or {}
        reps_m = manager.get("repPerformance") or []
        context["managerRepPerformance"] = reps_m[:12]
        context["managerTeamPerformance"] = (manager.get("teamPerformance") or [])[:10]
        context["doorStats"] = manager.get("doorStats") or {}
        context["dealStageBreakdown"] = (manager.get("dealStageBreakdown") or [])[:10]
        context["pendingOutcome"] = (manager.get("pendingOutcome") or [])[:15]
        hits = matched_vocab_hits or []
        msg_l = (message or "").lower()
        if hits:
            hit_set = {h.strip().lower() for h in hits if str(h).strip()}
            mf = [r for r in reps_m if str(r.get("salesRep") or "").strip().lower() in hit_set]
            if not mf:
                mf = [r for r in reps_m if any(h in str(r.get("salesRep") or "").lower() for h in hit_set)]
            if mf:
                context["managerRepPerformanceFocus"] = mf[:6]
        if hits or "setter" in msg_l or "appointment" in msg_l:
            try:
                role = get_role_performance_bundle(date_from, date_to, user, **dim_kw)
                setters = role.get("setters") or []
                if hits:
                    hit_set = {h.strip().lower() for h in hits if str(h).strip()}
                    focus_s = [s for s in setters if str(s.get("agent") or "").strip().lower() in hit_set]
                    if not focus_s:
                        focus_s = [
                            s for s in setters if any(h in str(s.get("agent") or "").lower() for h in hit_set)
                        ]
                    if focus_s:
                        context["setterPerformanceFocus"] = focus_s[:4]
            except Exception as exc:
                logger.warning("role performance bundle for chat (manager scope) failed: %s", exc)
    elif scope == "on_hold_details":
        on_hold_rows = get_on_hold_projects(date_from, date_to, user, **dim_kw).values(
            "first_name", "last_name", "sales_rep", "sales_team", "on_hold_reason", "job_status", "customer_since"
        )[:25]
        context["onHoldReasons"] = get_on_hold_reasons_breakdown(date_from, date_to, user, **dim_kw)[:12]
        context["onHoldProjectsSample"] = list(on_hold_rows)
    elif scope == "cancellations":
        cancelled_rows = get_cancelled_projects(date_from, date_to, user, **dim_kw).values(
            "first_name", "last_name", "sales_rep", "sales_team", "cancellation_reason", "job_status", "customer_since"
        )[:25]
        context["cancellationReasons"] = get_cancellation_reasons_breakdown(date_from, date_to, user, **dim_kw)[:12]
        context["cancelledProjectsSample"] = list(cancelled_rows)
    elif scope == "outcome_pending":
        manager = get_manager_bundle(date_from, date_to, user, **dim_kw)
        context["managerOverview"] = manager.get("managerOverview") or {}
        context["pendingOutcome"] = (manager.get("pendingOutcome") or [])[:20]
    return context


def _scored_fuzzy_name_candidates(
    message: str, vocab: list[str], *, ratio_min: float = 0.82
) -> list[dict[str, Any]]:
    """Best fuzzy match score per canonical vocab name (two-token spans vs full name)."""
    normalized = re.sub(r"[^\w\s]", " ", (message or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = normalized.split()
    best: dict[str, float] = {}
    best_span: dict[str, str] = {}
    if len(tokens) < 2:
        return []
    for i in range(len(tokens) - 1):
        span = f"{tokens[i]} {tokens[i + 1]}"
        if len(span) < 6:
            continue
        for name in set(vocab):
            n = str(name).strip().lower()
            if len(n) < 6:
                continue
            r = difflib.SequenceMatcher(None, span, n).ratio()
            if r < ratio_min:
                continue
            if r > best.get(n, 0.0):
                best[n] = r
                best_span[n] = span
    out = [
        {"canonical_name": name, "score": best[name], "matched_span": best_span[name]}
        for name in sorted(best, key=lambda k: -best[k])
    ]
    return out[:10]


def _ambiguous_entities_decision(
    candidates: list[dict[str, Any]], *, score_gap_max: float = 0.045, min_score: float = 0.82
) -> list[dict[str, Any]] | None:
    if len(candidates) < 2:
        return None
    if float(candidates[1]["score"]) < min_score:
        return None
    if float(candidates[0]["score"]) - float(candidates[1]["score"]) <= score_gap_max:
        return candidates[:4]
    return None


def _thread_hints_for_pack(
    conversation: InsightConversation,
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    msgs = list(
        conversation.messages.filter(deleted_at__isnull=True).order_by("-created_at")[:24]
    )
    last_a = next((m for m in msgs if m.role == InsightMessage.Role.ASSISTANT), None)
    last_u = next((m for m in msgs if m.role == InsightMessage.Role.USER), None)
    out: dict[str, Any] = {
        "last_user_preview": ((last_u.content or "").strip()[:500] if last_u else None),
        "last_assistant_preview": ((last_a.content or "").strip()[:900] if last_a else None),
    }
    ss = session_state or {}
    if ss.get("last_human_window") or ss.get("active_timeframe"):
        out["session_prior_window"] = ss.get("last_human_window")
        out["session_prior_timeframe"] = ss.get("active_timeframe")
    return out


def _build_wide_chat_context_pack(
    *,
    user,
    date_from,
    date_to,
    message: str,
    scope_hint: str | None,
    metric_hint: str | None,
    vocab_hits: list[str],
    ambiguous_entities: list[dict[str, Any]] | None,
    conversation: InsightConversation,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
    requested_window_meta: dict[str, Any] | None = None,
    comparison_window: dict[str, Any] | None = None,
    continuity: dict[str, Any] | None = None,
    prior_session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Bounded multi-slice context for conversational analytics (Phase 1).
    `scope_hint` is metadata only; the model chooses relevance from the full pack.
    """
    dim_kw = dict(
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
        market=market,
        rep_kind=rep_kind,
        rep_name=rep_name,
    )
    rw_meta = requested_window_meta or {}
    requested_window = {
        "date_from": str(date_from) if date_from else None,
        "date_to": str(date_to) if date_to else None,
        "timeframe_id": rw_meta.get("timeframe_id"),
        "human_label": rw_meta.get("human_window") or (date_from and date_to and _human_date_window(date_from, date_to)),
        "source": rw_meta.get("source"),
    }

    comp_block: dict[str, Any] | None = None
    if comparison_window and comparison_window.get("date_from") and comparison_window.get("date_to"):
        c0, c1 = comparison_window["date_from"], comparison_window["date_to"]
        cov = get_overview_metrics(c0, c1, user, **dim_kw)
        comp_block = {
            "label": comparison_window.get("label"),
            "date_from": str(c0),
            "date_to": str(c1),
            "human_label": comparison_window.get("human_window") or _human_date_window(c0, c1),
            "overview": _slim_overview_for_compare(cov),
        }

    ov = get_overview_metrics(date_from, date_to, user, **dim_kw)
    perf = get_performance_bundle(date_from, date_to, user, **dim_kw)
    reps_all = perf.get("reps") or []
    teams_slice = (perf.get("teams") or [])[:8]

    setters_sample: list[dict[str, Any]] = []
    setters_all: list[dict[str, Any]] = []
    try:
        role_bundle = get_role_performance_bundle(date_from, date_to, user, **dim_kw)
        setters_all = list(role_bundle.get("setters") or [])
        setters_sample = sorted(setters_all, key=lambda x: -int(x.get("appointments") or 0))[:15]
    except Exception as exc:
        logger.warning("wide pack: role performance bundle failed: %s", exc)

    clean = get_clean_deals_bundle(date_from, date_to, user, **dim_kw)
    portfolio = _shape_clean_deal_portfolio_for_chat(clean.get("analysis") or [])

    retention = get_retention_bundle(date_from, date_to, user, **dim_kw)
    by_rep_ret = retention.get("byRep") or []
    retention_top: list[dict[str, Any]] = []
    for row in sorted(by_rep_ret, key=lambda r: float(r.get("netRetentionRate") or 0.0), reverse=True)[:6]:
        retention_top.append(
            {
                "sales_rep": str(row.get("sales_rep") or row.get("salesRep") or ""),
                "totalProjects": int(row.get("totalProjects") or 0),
                "cancellationRate": float(row.get("cancellationRate") or 0.0),
                "netRetentionRate": float(row.get("netRetentionRate") or 0.0),
            }
        )

    cancel_reasons = get_cancellation_reasons_breakdown(date_from, date_to, user, **dim_kw)[:8]

    hit_set = {h.strip().lower() for h in (vocab_hits or []) if str(h).strip()}
    rep_focus: list[dict[str, Any]] = []
    setter_focus: list[dict[str, Any]] = []
    if hit_set and not ambiguous_entities:
        rep_focus = [r for r in reps_all if str(r.get("salesRep") or "").strip().lower() in hit_set]
        if not rep_focus:
            rep_focus = [
                r for r in reps_all if any(h in str(r.get("salesRep") or "").lower() for h in hit_set)
            ]
        rep_focus = rep_focus[:6]
        for s in setters_all:
            ag = str(s.get("agent") or "").strip().lower()
            if ag in hit_set or any(h in ag for h in hit_set):
                setter_focus.append(s)
        if not setter_focus:
            setter_focus = [
                s for s in setters_all if any(h in str(s.get("agent") or "").lower() for h in hit_set)
            ]
        setter_focus = setter_focus[:5]

    hints = _thread_hints_for_pack(conversation, session_state=prior_session_state)

    pack: dict[str, Any] = {
        "pack_version": "wide-v2",
        "scope_hint": scope_hint or "executive_overview",
        "metric_hint": metric_hint,
        "requested_window": requested_window,
        "overview": ov,
        "rep_performance_sample": reps_all[:18],
        "team_performance_sample": teams_slice,
        "setter_role_metrics_sample": setters_sample,
        "rep_focus": rep_focus,
        "setter_role_metrics_focus": setter_focus,
        "clean_deal_portfolio": portfolio,
        "retention_headline": {"top_reps_by_net_retention": retention_top},
        "cancellation_summary": {
            "overview_cancellation_rate_pct": float(ov.get("cancellationRate") or 0.0),
            "top_cancellation_reasons": [
                {"reason": str(r.get("reason") or ""), "count": int(r.get("count") or 0)} for r in cancel_reasons
            ],
        },
        "ambiguous_entities": ambiguous_entities,
        "thread_hints": hints,
    }
    if comp_block:
        pack["comparison_window"] = comp_block
    if continuity:
        pack["continuity"] = continuity
    return pack


def _filter_wide_pack_for_llm(pack: dict[str, Any]) -> dict[str, Any]:
    """Slim wide pack for token control while keeping cross-domain signals."""
    out: dict[str, Any] = {
        "shape_version": "v4-wide-llm",
        "pack_version": pack.get("pack_version"),
        "scope_hint": pack.get("scope_hint"),
        "metric_hint": pack.get("metric_hint"),
    }
    ov = pack.get("overview") or {}
    out["overview"] = {
        "totalProjects": ov.get("totalProjects"),
        "activeProjects": ov.get("activeProjects"),
        "cancelledProjects": ov.get("cancelledProjects"),
        "onHoldProjects": ov.get("onHoldProjects"),
        "cleanDeals": ov.get("cleanDeals"),
        "cleanDealPct": ov.get("cleanDealPct"),
        "cancellationRate": ov.get("cancellationRate"),
        "netRetentionRate": ov.get("netRetentionRate"),
        "activePipelineValue": ov.get("activePipelineValue"),
    }
    out["rep_performance_sample"] = [
        {
            "salesRep": r.get("salesRep"),
            "salesTeam": r.get("salesTeam"),
            "totalProjects": int(r.get("totalProjects") or 0),
            "cleanDealPct": float(r.get("cleanDealPct") or 0.0),
            "cancellationRate": float(r.get("cancellationRate") or 0.0),
            "netRetentionRate": float(r.get("netRetentionRate") or 0.0),
        }
        for r in (pack.get("rep_performance_sample") or [])[:14]
    ]
    out["team_performance_sample"] = [
        {
            "salesTeam": r.get("salesTeam"),
            "totalProjects": int(r.get("totalProjects") or 0),
            "cleanDealPct": float(r.get("cleanDealPct") or 0.0),
            "cancellationRate": float(r.get("cancellationRate") or 0.0),
        }
        for r in (pack.get("team_performance_sample") or [])[:8]
    ]
    setter_rows = pack.get("setter_role_metrics_sample") or []
    out["setter_role_metrics_sample"] = [
        {
            "agent": r.get("agent"),
            "allDoors": int(r.get("allDoors") or 0),
            "contactsMade": int(r.get("contactsMade") or 0),
            "appointments": int(r.get("appointments") or 0),
            "sitdowns": int(r.get("sitdowns") or 0),
            "qfdSitdowns": int(r.get("qfdSitdowns") or 0),
            "contactRate": float(r.get("contactRate") or 0.0),
            "apptSchedRatio": float(r.get("apptSchedRatio") or 0.0),
            "sitDownRate": float(r.get("sitDownRate") or 0.0),
            "cancellationRate": float(r.get("cancellationRate") or 0.0),
        }
        for r in setter_rows[:12]
    ]
    rep_f = pack.get("rep_focus") or []
    out["rep_focus"] = [
        {
            "salesRep": r.get("salesRep"),
            "salesTeam": r.get("salesTeam"),
            "totalProjects": int(r.get("totalProjects") or 0),
            "cleanDealPct": float(r.get("cleanDealPct") or 0.0),
            "cancellationRate": float(r.get("cancellationRate") or 0.0),
            "netRetentionRate": float(r.get("netRetentionRate") or 0.0),
        }
        for r in rep_f[:5]
    ]
    set_f = pack.get("setter_role_metrics_focus") or []
    out["setter_role_metrics_focus"] = [
        {
            "agent": r.get("agent"),
            "allDoors": int(r.get("allDoors") or 0),
            "contactsMade": int(r.get("contactsMade") or 0),
            "appointments": int(r.get("appointments") or 0),
            "sitdowns": int(r.get("sitdowns") or 0),
            "qfdSitdowns": int(r.get("qfdSitdowns") or 0),
            "contactRate": float(r.get("contactRate") or 0.0),
            "apptSchedRatio": float(r.get("apptSchedRatio") or 0.0),
            "sitDownRate": float(r.get("sitDownRate") or 0.0),
            "cancellationRate": float(r.get("cancellationRate") or 0.0),
        }
        for r in set_f[:5]
    ]
    out["clean_deal_portfolio"] = pack.get("clean_deal_portfolio") or []
    rh = pack.get("retention_headline") or {}
    out["retention_headline"] = {
        "top_reps_by_net_retention": (rh.get("top_reps_by_net_retention") or [])[:6],
    }
    cs = pack.get("cancellation_summary") or {}
    out["cancellation_summary"] = {
        "overview_cancellation_rate_pct": cs.get("overview_cancellation_rate_pct"),
        "top_cancellation_reasons": (cs.get("top_cancellation_reasons") or [])[:6],
    }
    amb = pack.get("ambiguous_entities")
    if amb:
        out["ambiguous_entities"] = amb
    rw = pack.get("requested_window")
    if rw:
        out["requested_window"] = rw
    cw = pack.get("comparison_window")
    if cw:
        out["comparison_window"] = cw
    cont = pack.get("continuity")
    if cont:
        out["continuity"] = cont
    th = pack.get("thread_hints") or {}
    th_out: dict[str, Any] = {}
    if th.get("last_user_preview") or th.get("last_assistant_preview"):
        th_out["last_user_preview"] = th.get("last_user_preview")
        th_out["last_assistant_preview"] = th.get("last_assistant_preview")
    if th.get("session_prior_window") or th.get("session_prior_timeframe"):
        th_out["session_prior_window"] = th.get("session_prior_window")
        th_out["session_prior_timeframe"] = th.get("session_prior_timeframe")
    if th_out:
        out["thread_hints"] = th_out
    return out


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
        focus_reps = shaped_context.get("rep_performance_focus") or []
        if focus_reps:
            filtered["rep_focus"] = [
                {
                    "sales_rep": _rep_name(row),
                    "totalProjects": int(row.get("totalProjects") or 0),
                    "cleanDealPct": float(row.get("cleanDealPct") or 0.0),
                    "cancellationRate": float(row.get("cancellationRate") or 0.0),
                    "netRetentionRate": float(row.get("netRetentionRate") or 0.0),
                }
                for row in focus_reps[:4]
            ]
        setter_focus = shaped_context.get("setter_performance_focus") or []
        setter_sample = shaped_context.get("setter_performance_sample") or []
        setter_rows = setter_focus if setter_focus else setter_sample
        if setter_rows:
            filtered["setter_role_metrics"] = [
                {
                    "agent": row.get("agent"),
                    "allDoors": int(row.get("allDoors") or 0),
                    "contactsMade": int(row.get("contactsMade") or 0),
                    "appointments": int(row.get("appointments") or 0),
                    "sitdowns": int(row.get("sitdowns") or 0),
                    "qfdSitdowns": int(row.get("qfdSitdowns") or 0),
                    "contactRate": float(row.get("contactRate") or 0.0),
                    "apptSchedRatio": float(row.get("apptSchedRatio") or 0.0),
                    "sitDownRate": float(row.get("sitDownRate") or 0.0),
                    "cancellationRate": float(row.get("cancellationRate") or 0.0),
                }
                for row in setter_rows[:8]
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
        filtered["clean_deal_portfolio"] = shaped_context.get("clean_deal_portfolio") or []
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
        mgr_focus = shaped_context.get("manager_rep_performance_focus") or []
        if mgr_focus:
            filtered["manager_rep_focus"] = [
                {
                    "salesRep": row.get("salesRep"),
                    "salesTeam": row.get("salesTeam"),
                    "totalAppointments": row.get("totalAppointments"),
                    "closedDeals": row.get("closedDeals"),
                    "cancelledDeals": row.get("cancelledDeals"),
                    "sitDownRate": row.get("sitDownRate"),
                    "closingRate": row.get("closingRate"),
                }
                for row in mgr_focus[:6]
            ]
        setter_focus_m = shaped_context.get("setter_performance_focus") or []
        if setter_focus_m:
            filtered["setter_role_metrics"] = [
                {
                    "agent": row.get("agent"),
                    "allDoors": int(row.get("allDoors") or 0),
                    "contactsMade": int(row.get("contactsMade") or 0),
                    "appointments": int(row.get("appointments") or 0),
                    "sitdowns": int(row.get("sitdowns") or 0),
                    "qfdSitdowns": int(row.get("qfdSitdowns") or 0),
                    "contactRate": float(row.get("contactRate") or 0.0),
                    "apptSchedRatio": float(row.get("apptSchedRatio") or 0.0),
                    "sitDownRate": float(row.get("sitDownRate") or 0.0),
                    "cancellationRate": float(row.get("cancellationRate") or 0.0),
                }
                for row in setter_focus_m[:6]
            ]
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
    if raw.get("repPerformanceFocus"):
        shaped["rep_performance_focus"] = (raw.get("repPerformanceFocus") or [])[:6]
    if raw.get("setterPerformanceFocus"):
        shaped["setter_performance_focus"] = (raw.get("setterPerformanceFocus") or [])[:6]
    if raw.get("setterPerformanceSample"):
        shaped["setter_performance_sample"] = (raw.get("setterPerformanceSample") or [])[:12]
    if raw.get("teamPerformance"):
        shaped["team_breakdown"] = (raw.get("teamPerformance") or [])[:10]
    if raw.get("cleanDealsByRep"):
        shaped["clean_deals_by_rep"] = (raw.get("cleanDealsByRep") or [])[:10]
    if raw.get("cleanDealPortfolio"):
        shaped["clean_deal_portfolio"] = raw.get("cleanDealPortfolio") or []
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
    if raw.get("managerRepPerformanceFocus"):
        shaped["manager_rep_performance_focus"] = (raw.get("managerRepPerformanceFocus") or [])[:6]
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
        if shaped_context.get("shape_version") == "v4-wide-llm":
            n = 0
            for k in (
                "rep_performance_sample",
                "setter_role_metrics_sample",
                "rep_focus",
                "setter_role_metrics_focus",
            ):
                v = shaped_context.get(k)
                if isinstance(v, list):
                    n += len(v)
            rh = shaped_context.get("retention_headline") or {}
            n += len(rh.get("top_reps_by_net_retention") or [])
            cs = shaped_context.get("cancellation_summary") or {}
            n += len(cs.get("top_cancellation_reasons") or [])
            if shaped_context.get("ambiguous_entities"):
                n += len(shaped_context["ambiguous_entities"])
            meta.update(
                {
                    "context_type": "wide_pack",
                    "shape_version": shaped_context.get("shape_version"),
                    "pack_version": shaped_context.get("pack_version"),
                    "record_count": n,
                    "data_scope": data_scope or shaped_context.get("scope_hint") or "wide",
                }
            )
        else:
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
    rows = list(conversation.messages.filter(deleted_at__isnull=True).order_by("-created_at")[:10])
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
    if _message_prefers_llm_narrative(text):
        return None
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


def _message_prefers_llm_narrative(text: str) -> bool:
    """
    Long coaching / summary prompts should hit the LLM with dashboard context,
    not the deterministic 'top rep' shortcut (which misfires on phrases like
    'which is the most important').
    """
    lowered = (text or "").lower()
    if len(lowered) > 900:
        return True
    cues = (
        "summarize",
        "summary",
        "help me",
        "ideas",
        "explain",
        "what matters",
        "when it comes to",
        "walk me through",
        "kind of",
        "appointment setter",
        "things that he",
        "things she can",
    )
    if any(c in lowered for c in cues):
        return True
    bulletish = lowered.count("\n-") + lowered.count("\n*") + lowered.count(" - ")
    return bulletish >= 2 and len(lowered) > 200


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
    if _message_prefers_llm_narrative(text):
        return None
    # Avoid treating prose like "which is the most important" as a ranking question.
    has_superlative = any(term in normalized for term in ("top", "best", "highest", "lowest", "least", "worst"))
    has_most_ranking = bool(
        re.search(
            r"\b(most cancelled|most deals|most appointments|most projects|has the most|with the most|"
            r"most sit|most closes|highest number|lowest number|"
            r"who\s+has\s+the\s+most|which\s+\w+\s+has\s+the\s+most)\b",
            normalized,
        )
    )
    if not (has_superlative or has_most_ranking):
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


def _compute_cancellation_rep_result(
    plan: dict[str, Any],
    date_from,
    date_to,
    user,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
) -> tuple[str, dict[str, Any]] | None:
    raw_context = _gather_chat_context(
        "retention",
        date_from,
        date_to,
        user,
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
        market=market,
        rep_kind=rep_kind,
        rep_name=rep_name,
    )
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


def _compute_deterministic_rank_result(
    plan: dict[str, Any],
    date_from,
    date_to,
    user,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
) -> tuple[str, dict[str, Any]] | None:
    raw_context = _gather_chat_context(
        plan.get("scope") or "executive_overview",
        date_from,
        date_to,
        user,
        metric_hint=plan.get("metric_hint"),
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
        market=market,
        rep_kind=rep_kind,
        rep_name=rep_name,
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


def _chat_cache_key(
    normalized_question: str,
    scope: str | None,
    date_from,
    date_to,
    plan_signature: str,
    dim_sig: str = "",
) -> str:
    raw_key = (
        f"{normalized_question}|{scope or 'none'}|{date_from or 'none'}|"
        f"{date_to or 'none'}|{plan_signature}|{dim_sig or 'nodims'}"
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


def _chat_response_for_non_dashboard(
    intent_label: str, *, safety_flags: dict[str, Any] | None = None
) -> str | None:
    if intent_label == InsightMessage.IntentLabel.GREETING_SMALLTALK:
        return "Hi! Ask me about Sunbright dashboard metrics like revenue, retention, clean deal rate, or rep/team performance."
    if intent_label == InsightMessage.IntentLabel.OUT_OF_SCOPE:
        if safety_flags and safety_flags.get("possiblePromptInjection"):
            return (
                "I can't follow requests like that. Ask about your Sunbright dashboard metrics, "
                "teams, retention, or performance instead."
            )
        return "I focus on Sunbright analytics here. Ask about performance, retention, clean deals, or your KPIs."
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
        if not reasons and shaped_context.get("cancellation_summary"):
            tr = (shaped_context.get("cancellation_summary") or {}).get("top_cancellation_reasons") or []
            reasons = [str(r.get("reason") or "") for r in tr[:3] if r.get("reason")]
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
    sales_team=None,
    installer=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
) -> dict[str, Any]:
    text = (message or "").strip()
    dim_sig_parts = []
    if installer and str(installer).strip():
        dim_sig_parts.append(f"i:{str(installer).strip().lower()}")
    if sales_team and str(sales_team).strip():
        dim_sig_parts.append(f"t:{str(sales_team).strip().lower()}")
    if lead_source and str(lead_source).strip():
        dim_sig_parts.append(f"l:{str(lead_source).strip().lower()}")
    if project_manager and str(project_manager).strip():
        dim_sig_parts.append(f"p:{str(project_manager).strip().lower()}")
    if market and str(market).strip():
        dim_sig_parts.append(f"k:{str(market).strip().lower()}")
    rk = (rep_kind or "").strip().lower() if rep_kind else ""
    rn = (rep_name or "").strip() if rep_name else ""
    if rk in ("sales_rep", "setter") and rn:
        dim_sig_parts.append(f"r:{rk}:{rn.lower()}")
    chat_dim_sig = "|".join(sorted(dim_sig_parts)) if dim_sig_parts else ""
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

    safety_flags = _detect_safety_flags(text)
    intent_label = _classify_intent(text, safety_flags=safety_flags)
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
    session_state0: dict[str, Any] = {}
    extraction: dict[str, Any] = {}
    date_trace: dict[str, Any] = {}
    eff_from = date_from if date_from is not None else conversation.date_from
    eff_to = date_to if date_to is not None else conversation.date_to
    continuity = None

    if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
        snap0 = conversation.scope_snapshot or {}
        session_state0 = snap0.get("chat_session") if isinstance(snap0.get("chat_session"), dict) else {}
        base_from = date_from if date_from is not None else conversation.date_from
        base_to = date_to if date_to is not None else conversation.date_to
        extraction = _extract_conversation_filters(
            text,
            {"date_from": base_from, "date_to": base_to},
            session_state0,
        )
        eff_from, eff_to, date_trace = _resolve_effective_chat_dates(extraction, session_state0, base_from, base_to)
        continuity = _continuity_block_for_llm(session_state0)
        data_scope = _detect_dashboard_scope(text)
        metric_hint = _detect_metric_hint(text, data_scope)
        if data_scope == "executive_overview":
            try:
                v2 = _load_chat_rep_setter_vocab(
                    user,
                    eff_from,
                    eff_to,
                    installer=installer,
                    sales_team=sales_team,
                    lead_source=lead_source,
                    project_manager=project_manager,
                    market=market,
                    rep_kind=rep_kind,
                    rep_name=rep_name,
                )
                if _message_matches_vocab_name(text, v2):
                    data_scope = "performance"
                    metric_hint = metric_hint or "rep_performance"
            except Exception as exc:
                logger.warning("insights chat scope assist vocab failed: %s", exc)
        print("[insights-chat] dashboard scope selected", {"scope": data_scope})
        print(
            "[insights-chat] metric hint + dates",
            {
                "metric_hint": metric_hint,
                "eff_from": str(eff_from) if eff_from else None,
                "eff_to": str(eff_to) if eff_to else None,
                "date_window_source": date_trace.get("source"),
            },
        )
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
            eff_from,
            eff_to,
            plan_signature,
            chat_dim_sig,
        )
        cached_payload = cache.get(cache_key)
        if cached_payload:
            cache_hit = True
            print("[insights-chat] cache_hit", {"cache_hit": cache_hit, "cache_key": cache_key})
            deterministic_hit = True
            context_meta = _build_context_meta(
                intent_label,
                eff_from,
                eff_to,
                None,
                data_scope=data_scope,
            )
            context_meta.update({"deterministic_hit": True, "cache_hit": True, "llm_used": False})
            if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
                context_meta.update(
                    {
                        "date_window_source": date_trace.get("source"),
                        "active_timeframe": extraction.get("active_timeframe") or session_state0.get("active_timeframe"),
                        "active_topic": extraction.get("active_topic") or session_state0.get("active_topic"),
                        "utterance_temporal_hit": extraction.get("temporal_matched"),
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
                content=str(cached_payload.get("reply") or ""),
                intent_label=intent_label,
                context_meta=context_meta,
            )
            print("[insights-chat] deterministic_hit", {"deterministic_hit": True, "cache_hit": True, "llm_used": False})
            if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
                _persist_chat_session_state(
                    conversation,
                    extraction=extraction,
                    eff_from=eff_from,
                    eff_to=eff_to,
                    date_trace=date_trace,
                )
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
            eff_from,
            eff_to,
            user,
            installer=installer,
            sales_team=sales_team,
            lead_source=lead_source,
            project_manager=project_manager,
            market=market,
            rep_kind=rep_kind,
            rep_name=rep_name,
        )
        if deterministic_result is None and deterministic_plan.get("scope") == "retention":
            deterministic_result = _compute_cancellation_rep_result(
                deterministic_plan,
                eff_from,
                eff_to,
                user,
                installer=installer,
                sales_team=sales_team,
                lead_source=lead_source,
                project_manager=project_manager,
                market=market,
                rep_kind=rep_kind,
                rep_name=rep_name,
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
                eff_from,
                eff_to,
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
            if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
                context_meta.update(
                    {
                        "date_window_source": date_trace.get("source"),
                        "active_timeframe": extraction.get("active_timeframe") or session_state0.get("active_timeframe"),
                        "active_topic": extraction.get("active_topic") or session_state0.get("active_topic"),
                        "utterance_temporal_hit": extraction.get("temporal_matched"),
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
            if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
                _persist_chat_session_state(
                    conversation,
                    extraction=extraction,
                    eff_from=eff_from,
                    eff_to=eff_to,
                    date_trace=date_trace,
                )
            return {
                "conversationId": conversation.id,
                "messageId": assistant_row.id,
                "reply": assistant_row.content,
                "intentLabel": intent_label,
                "contextMeta": context_meta,
            }
    if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
        vocab_bundle = _load_chat_rep_setter_vocab(
            user,
            eff_from,
            eff_to,
            installer=installer,
            sales_team=sales_team,
            lead_source=lead_source,
            project_manager=project_manager,
            market=market,
            rep_kind=rep_kind,
            rep_name=rep_name,
        )
        scored_names = _scored_fuzzy_name_candidates(text, vocab_bundle)
        ambiguous_entities = _ambiguous_entities_decision(scored_names)
        vocab_hits = [] if ambiguous_entities else _vocab_hits_in_message(text, vocab_bundle)
        human_w = extraction.get("human_window")
        tf_id = extraction.get("active_timeframe") or session_state0.get("active_timeframe")
        if not human_w and eff_from and eff_to:
            human_w = _human_date_window(eff_from, eff_to)
        req_win_meta = {
            "timeframe_id": tf_id,
            "human_window": human_w,
            "source": date_trace.get("source"),
        }
        wide_pack = _build_wide_chat_context_pack(
            user=user,
            date_from=eff_from,
            date_to=eff_to,
            message=text,
            scope_hint=data_scope or "executive_overview",
            metric_hint=metric_hint,
            vocab_hits=vocab_hits,
            ambiguous_entities=ambiguous_entities,
            conversation=conversation,
            installer=installer,
            sales_team=sales_team,
            lead_source=lead_source,
            project_manager=project_manager,
            market=market,
            rep_kind=rep_kind,
            rep_name=rep_name,
            requested_window_meta=req_win_meta,
            comparison_window=extraction.get("comparison"),
            continuity=continuity,
            prior_session_state=session_state0,
        )
        shaped_context = _filter_wide_pack_for_llm(wide_pack)
        print(
            "[insights-chat] context prepared",
            {
                "scope_hint": (shaped_context or {}).get("scope_hint"),
                "shape_version": (shaped_context or {}).get("shape_version"),
                "ambiguous_entities": bool((shaped_context or {}).get("ambiguous_entities")),
                "keys": list((shaped_context or {}).keys()),
            },
        )
    context_meta = _build_context_meta(
        intent_label,
        eff_from,
        eff_to,
        shaped_context,
        data_scope=data_scope,
    )
    context_meta.update({"deterministic_hit": deterministic_hit, "cache_hit": cache_hit, "llm_used": llm_used})
    if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
        context_meta.update(
            {
                "date_window_source": date_trace.get("source"),
                "active_timeframe": extraction.get("active_timeframe") or session_state0.get("active_timeframe"),
                "active_topic": extraction.get("active_topic") or session_state0.get("active_topic"),
                "utterance_temporal_hit": extraction.get("temporal_matched"),
            }
        )
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

    static_reply = _chat_response_for_non_dashboard(intent_label, safety_flags=safety_flags)
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
        if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
            _persist_chat_session_state(
                conversation,
                extraction=extraction,
                eff_from=eff_from,
                eff_to=eff_to,
                date_trace=date_trace,
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
        if intent_label == InsightMessage.IntentLabel.DASHBOARD_QUERY:
            _persist_chat_session_state(
                conversation,
                extraction=extraction,
                eff_from=eff_from,
                eff_to=eff_to,
                date_trace=date_trace,
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


def generate_dashboard_insights(
    date_from,
    date_to,
    user,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
) -> dict[str, Any]:
    ctx = gather_insights_context(
        date_from,
        date_to,
        user,
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
        market=market,
        rep_kind=rep_kind,
        rep_name=rep_name,
    )
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
