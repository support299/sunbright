"""
Aggregations for dashboard API responses (mirrors sunbright-dashboard/db.ts logic where possible).
Uses fields present on synced Django models.
"""
from collections import defaultdict
from functools import reduce
from operator import or_

from django.db.models import (
    Avg,
    Count,
    DecimalField,
    Max,
    Min,
    Q,
    Sum,
    Value,
)
from django.db.models.functions import Coalesce, Lower
from django.utils import timezone

from dashboard.models import (
    Appointment,
    CxProject,
    DashboardDataScope,
    Door,
    SunbaseUser,
)
from dashboard.scope import appointment_scope_q, cx_scope_q, door_scope_q
from dashboard.services.project_service import base_queryset

_ZERO_MONEY = Value(0, output_field=DecimalField(max_digits=15, decimal_places=2))


def _cx_qs(date_from=None, date_to=None, user=None):
    qs = CxProject.objects.filter(deleted_at__isnull=True).filter(cx_scope_q(user))
    if date_from:
        qs = qs.filter(install_date__gte=date_from)
    if date_to:
        qs = qs.filter(install_date__lte=date_to)
    return qs


def _door_qs(date_from=None, date_to=None, user=None):
    qs = Door.objects.filter(deleted_at__isnull=True).filter(door_scope_q(user))
    if date_from:
        qs = qs.filter(create_time__date__gte=date_from)
    if date_to:
        qs = qs.filter(create_time__date__lte=date_to)
    return qs


def _appt_qs(date_from=None, date_to=None, user=None):
    qs = Appointment.objects.filter(deleted_at__isnull=True).filter(appointment_scope_q(user))
    if date_from:
        qs = qs.filter(appointment_datetime__date__gte=date_from)
    if date_to:
        qs = qs.filter(appointment_datetime__date__lte=date_to)
    return qs


_ACTIVE_CLOSED_STAGES = (
    "Deal - Pending Review",
    "Sold",
    "Sold (CRC)",
    "Sold/ Installed",
)
_CANCEL_STAGES = ("Deal Cancelled", "Canceled", "Closed - Not Complete")
_SHOW_STAGES = ("show", "qualified_show", "closed")


def _pct(num, den):
    if not den:
        return 0.0
    return round(100.0 * float(num) / float(den), 1)


def _round_avg(val):
    if val is None:
        return None
    return round(float(val), 1)


def _days_customer_to_install(customer_since, install_date):
    if not customer_since or not install_date or install_date < customer_since:
        return None
    return (install_date - customer_since).days


def _days_span(start, end):
    """Inclusive day delta when both dates exist and end >= start (pipeline milestones)."""
    if not start or not end or end < start:
        return None
    return (end - start).days


def _install_milestone_end(install_completed, install_date):
    """
    Sunbase often has Install Date filled but Install Completed empty; use install date as the
    install milestone so CRC→Install and Install→PTO can still be computed when other dates exist.
    """
    return install_completed or install_date


def _mean_rounded(values):
    vals = [v for v in values if v is not None]
    return _round_avg(sum(vals) / len(vals)) if vals else None


def _avg_install_metrics_by_group(rows, group_field_names):
    """
    Per group: avg days customer_since → install_date (sunbright-dashboard daysToInstall).
    Also clean / non-clean splits for rep-level fields.
    """
    buckets = defaultdict(
        lambda: {"all_s": 0, "all_n": 0, "c_s": 0, "c_n": 0, "nc_s": 0, "nc_n": 0}
    )
    for row in rows:
        key_parts = []
        for f in group_field_names:
            v = row.get(f)
            if isinstance(v, str):
                v = v.strip()
            else:
                v = v if v is not None else ""
            key_parts.append(v)
        key = tuple(key_parts)
        if not key or not key[0]:
            continue
        d = _days_customer_to_install(row.get("customer_since"), row.get("install_date"))
        if d is None:
            continue
        b = buckets[key]
        b["all_s"] += d
        b["all_n"] += 1
        if row.get("is_clean_deal"):
            b["c_s"] += d
            b["c_n"] += 1
        else:
            b["nc_s"] += d
            b["nc_n"] += 1

    out = {}
    for key, b in buckets.items():
        out[key] = {
            "avgDaysToInstall": _round_avg(b["all_s"] / b["all_n"]) if b["all_n"] else None,
            "avgInstallClean": _round_avg(b["c_s"] / b["c_n"]) if b["c_n"] else None,
            "avgInstallNotClean": _round_avg(b["nc_s"] / b["nc_n"]) if b["nc_n"] else None,
        }
    return out


def _clean_deal_analysis_row(qs, is_clean, today):
    sub = qs.filter(is_clean_deal=is_clean)
    total = sub.count()
    installed = sub.filter(install_date__isnull=False).count()
    cancelled = sub.filter(project_category="Cancelled").count()
    active = sub.filter(project_category="Active").count()

    deltas = []
    for cs, ins in sub.filter(install_date__isnull=False, customer_since__isnull=False).values_list(
        "customer_since", "install_date"
    ):
        d = _days_customer_to_install(cs, ins)
        if d is not None:
            deltas.append(d)
    avg_days_install = _round_avg(sum(deltas) / len(deltas)) if deltas else None

    cancel_ages = []
    for cs in sub.filter(project_category="Cancelled", customer_since__isnull=False).values_list(
        "customer_since", flat=True
    ):
        if cs:
            cancel_ages.append((today - cs).days)
    avg_days_cancel = _round_avg(sum(cancel_ages) / len(cancel_ages)) if cancel_ages else None

    return {
        "isCleanDeal": 1 if is_clean else 0,
        "total": total,
        "installed": installed,
        "cancelled": cancelled,
        "active": active,
        "avgDaysToInstall": avg_days_install,
        "avgDaysToCrc": None,
        "avgProjectAge": None,
        "avgDaysToCancel": avg_days_cancel,
    }


def _aggregate_clean_deal_dimension(rows, group_field_names):
    """Mirror sunbright-dashboard getCleanDealByRep/Team/Installer SQL using in-memory aggregation."""
    buckets = defaultdict(
        lambda: {
            "totalProjects": 0,
            "cleanDeals": 0,
            "cleanInstalled": 0,
            "sum_days_clean": 0,
            "cnt_days_clean": 0,
            "sum_days_not_clean": 0,
            "cnt_days_not_clean": 0,
        }
    )

    for row in rows:
        key_parts = []
        skip = False
        for f in group_field_names:
            v = row.get(f) or ""
            if isinstance(v, str):
                v = v.strip()
            if not v:
                skip = True
                break
            key_parts.append(v)
        if skip:
            continue
        key = tuple(key_parts)
        b = buckets[key]
        b["totalProjects"] += 1
        is_clean = bool(row.get("is_clean_deal"))
        if is_clean:
            b["cleanDeals"] += 1
        cs, ins = row.get("customer_since"), row.get("install_date")
        day_inst = _days_customer_to_install(cs, ins)
        if is_clean and ins:
            b["cleanInstalled"] += 1
        if day_inst is not None:
            if is_clean:
                b["sum_days_clean"] += day_inst
                b["cnt_days_clean"] += 1
            else:
                b["sum_days_not_clean"] += day_inst
                b["cnt_days_not_clean"] += 1

    out = []
    camel_map = {"sales_rep": "salesRep", "sales_team": "salesTeam", "installer": "installer"}
    for key, b in buckets.items():
        t = b["totalProjects"]
        c = b["cleanDeals"]
        ci = b["cleanInstalled"]
        item = {
            "totalProjects": t,
            "cleanDeals": c,
            "cleanDealPct": _pct(c, t),
            "cleanInstalled": ci,
            "realizationRatio": _pct(ci, c) if c else 0.0,
            "avgInstallClean": _round_avg(b["sum_days_clean"] / b["cnt_days_clean"])
            if b["cnt_days_clean"]
            else None,
            "avgInstallNotClean": _round_avg(b["sum_days_not_clean"] / b["cnt_days_not_clean"])
            if b["cnt_days_not_clean"]
            else None,
        }
        for i, fname in enumerate(group_field_names):
            item[camel_map.get(fname, fname)] = key[i]
        out.append(item)

    out.sort(key=lambda x: -x["totalProjects"])
    return out[:200]


def get_clean_deals_bundle(date_from=None, date_to=None, user=None):
    qs = base_queryset(date_from, date_to, user)
    today = timezone.now().date()

    analysis = [
        _clean_deal_analysis_row(qs, True, today),
        _clean_deal_analysis_row(qs, False, today),
    ]

    base_cols = ("is_clean_deal", "install_date", "customer_since", "project_category")

    rep_rows = list(qs.exclude(sales_rep="").values("sales_rep", "sales_team", *base_cols))
    by_rep = _aggregate_clean_deal_dimension(rep_rows, ("sales_rep", "sales_team"))

    team_rows = list(qs.exclude(sales_team="").values("sales_team", *base_cols))
    by_team = _aggregate_clean_deal_dimension(team_rows, ("sales_team",))

    inst_rows = list(qs.exclude(installer="").values("installer", *base_cols))
    by_installer = _aggregate_clean_deal_dimension(inst_rows, ("installer",))

    return {"analysis": analysis, "byRep": by_rep, "byTeam": by_team, "byInstaller": by_installer}


def _retention_row(base_qs, group_fields):
    rows = (
        base_qs.values(*group_fields)
        .annotate(
            totalProjects=Count("id"),
            activeProjects=Count("id", filter=Q(project_category="Active")),
            cancelledProjects=Count("id", filter=Q(project_category="Cancelled")),
            onHoldProjects=Count("id", filter=Q(project_category="On Hold")),
            redFlaggedProjects=Count("id", filter=Q(project_category="Red Flagged")),
        )
        .order_by("-totalProjects")[:200]
    )
    out = []
    for r in rows:
        t = r["totalProjects"]
        bad = r["cancelledProjects"] + r["onHoldProjects"] + r["redFlaggedProjects"]
        out.append(
            {
                **{k: r[k] for k in group_fields},
                "totalProjects": t,
                "activeProjects": r["activeProjects"],
                "cancelledProjects": r["cancelledProjects"],
                "onHoldProjects": r["onHoldProjects"],
                "redFlaggedProjects": r["redFlaggedProjects"],
                "cancellationRate": _pct(r["cancelledProjects"], t),
                "onHoldRate": _pct(r["onHoldProjects"], t),
                "netRetentionRate": _pct(t - bad, t) if t else 0.0,
                "avgDaysToCancel": None,
            }
        )
    return out


def get_retention_bundle(date_from=None, date_to=None, user=None):
    qs = base_queryset(date_from, date_to, user)
    return {
        "byRep": _retention_row(qs.exclude(sales_rep=""), ["sales_rep", "sales_team"]),
        "byTeam": _retention_row(qs.exclude(sales_team=""), ["sales_team"]),
        "byInstaller": _retention_row(qs.exclude(installer=""), ["installer"]),
        "byLeadSource": _retention_row(qs.exclude(lead_source=""), ["lead_source"]),
    }


def get_performance_bundle(date_from=None, date_to=None, user=None):
    qs = base_queryset(date_from, date_to, user)
    perf_cols = ("customer_since", "install_date", "is_clean_deal")
    rep_install_rows = list(qs.exclude(sales_rep="").values("sales_rep", "sales_team", *perf_cols))
    rep_install = _avg_install_metrics_by_group(rep_install_rows, ("sales_rep", "sales_team"))
    team_install_rows = list(qs.exclude(sales_team="").values("sales_team", *perf_cols))
    team_install = _avg_install_metrics_by_group(team_install_rows, ("sales_team",))
    inst_install_rows = list(qs.exclude(installer="").values("installer", *perf_cols))
    inst_install = _avg_install_metrics_by_group(inst_install_rows, ("installer",))

    reps = (
        qs.exclude(sales_rep="")
        .values("sales_rep", "sales_team")
        .annotate(
            totalProjects=Count("id"),
            activeProjects=Count("id", filter=Q(project_category="Active")),
            cancelledProjects=Count("id", filter=Q(project_category="Cancelled")),
            onHoldProjects=Count("id", filter=Q(project_category="On Hold")),
            redFlaggedProjects=Count("id", filter=Q(project_category="Red Flagged")),
            disqualifiedProjects=Count("id", filter=Q(project_category="Disqualified")),
            cleanDeals=Count("id", filter=Q(is_clean_deal=True)),
            activePipelineValue=Coalesce(
                Sum("contract_amount", filter=Q(project_category="Active")),
                _ZERO_MONEY,
            ),
            totalContractValue=Coalesce(Sum("contract_amount"), _ZERO_MONEY),
        )
        .order_by("-totalProjects")[:200]
    )
    rep_list = []
    for r in reps:
        t = r["totalProjects"]
        bad = r["cancelledProjects"] + r["onHoldProjects"] + r["redFlaggedProjects"]
        c = r["cleanDeals"]
        ikey = (r["sales_rep"], (r["sales_team"] or "").strip() if r["sales_team"] else "")
        im = rep_install.get(ikey, {})
        rep_list.append(
            {
                "salesRep": r["sales_rep"],
                "salesTeam": r["sales_team"] or "",
                "totalProjects": t,
                "activeProjects": r["activeProjects"],
                "cancelledProjects": r["cancelledProjects"],
                "onHoldProjects": r["onHoldProjects"],
                "redFlaggedProjects": r["redFlaggedProjects"],
                "disqualifiedProjects": r["disqualifiedProjects"],
                "cleanDeals": c,
                "cleanDealPct": _pct(c, t),
                "cancellationRate": _pct(r["cancelledProjects"], t),
                "netRetentionRate": _pct(t - bad, t) if t else 0.0,
                "avgInstallClean": im.get("avgInstallClean"),
                "avgInstallNotClean": im.get("avgInstallNotClean"),
                "avgDaysToInstall": im.get("avgDaysToInstall"),
                "activePipelineValue": float(r["activePipelineValue"] or 0),
                "totalContractValue": float(r["totalContractValue"] or 0),
            }
        )

    teams = (
        qs.exclude(sales_team="")
        .values("sales_team")
        .annotate(
            totalProjects=Count("id"),
            activeProjects=Count("id", filter=Q(project_category="Active")),
            cancelledProjects=Count("id", filter=Q(project_category="Cancelled")),
            onHoldProjects=Count("id", filter=Q(project_category="On Hold")),
            redFlaggedProjects=Count("id", filter=Q(project_category="Red Flagged")),
            cleanDeals=Count("id", filter=Q(is_clean_deal=True)),
            activePipelineValue=Coalesce(
                Sum("contract_amount", filter=Q(project_category="Active")),
                _ZERO_MONEY,
            ),
            totalContractValue=Coalesce(Sum("contract_amount"), _ZERO_MONEY),
        )
        .order_by("-totalProjects")[:200]
    )
    team_list = []
    for r in teams:
        t = r["totalProjects"]
        bad = r["cancelledProjects"] + r["onHoldProjects"] + r["redFlaggedProjects"]
        c = r["cleanDeals"]
        tk = ((r["sales_team"] or "").strip(),)
        tm = team_install.get(tk, {})
        team_list.append(
            {
                "salesTeam": r["sales_team"],
                "totalProjects": t,
                "activeProjects": r["activeProjects"],
                "cancelledProjects": r["cancelledProjects"],
                "onHoldProjects": r["onHoldProjects"],
                "redFlaggedProjects": r["redFlaggedProjects"],
                "cleanDeals": c,
                "cleanDealPct": _pct(c, t),
                "cancellationRate": _pct(r["cancelledProjects"], t),
                "netRetentionRate": _pct(t - bad, t) if t else 0.0,
                "avgDaysToInstall": tm.get("avgDaysToInstall"),
                "activePipelineValue": float(r["activePipelineValue"] or 0),
                "totalContractValue": float(r["totalContractValue"] or 0),
            }
        )

    inst_list = []
    for r in (
        qs.exclude(installer="")
        .values("installer")
        .annotate(
            totalProjects=Count("id"),
            activeProjects=Count("id", filter=Q(project_category="Active")),
            cancelledProjects=Count("id", filter=Q(project_category="Cancelled")),
            onHoldProjects=Count("id", filter=Q(project_category="On Hold")),
            redFlaggedProjects=Count("id", filter=Q(project_category="Red Flagged")),
            cleanDeals=Count("id", filter=Q(is_clean_deal=True)),
            totalContractValue=Coalesce(Sum("contract_amount"), _ZERO_MONEY),
        )
        .order_by("-totalProjects")[:200]
    ):
        t = r["totalProjects"]
        bad = r["cancelledProjects"] + r["onHoldProjects"] + r["redFlaggedProjects"]
        c = r["cleanDeals"]
        ik = ((r["installer"] or "").strip(),)
        im_i = inst_install.get(ik, {})
        inst_list.append(
            {
                "installer": r["installer"],
                "totalProjects": t,
                "activeProjects": r["activeProjects"],
                "cancelledProjects": r["cancelledProjects"],
                "cleanDeals": c,
                "cleanDealPct": _pct(c, t),
                "cancellationRate": _pct(r["cancelledProjects"], t),
                "netRetentionRate": _pct(t - bad, t) if t else 0.0,
                "avgDaysToInstall": im_i.get("avgDaysToInstall"),
                "totalContractValue": float(r["totalContractValue"] or 0),
            }
        )

    return {"reps": rep_list, "teams": team_list, "installers": inst_list}


_PIPELINE_FUNNEL_ORDER = (
    "Sold Projects",
    "Site Survey",
    "Engineering",
    "Permitting",
    "Ready for Install",
    "Install",
    "Inspection",
    "PTO",
    "Completed",
    "Cancelled",
    "On Hold",
)


def get_pipeline_bundle(date_from=None, date_to=None, user=None):
    """
    Pipeline visualisation bundle.

    Returns velocity rows (one per project) + canonical funnel buckets,
    CRC analytics, deals-pipeline counters, and the quick-install rollup.
    """
    qs = base_queryset(date_from, date_to, user)
    today = timezone.now().date()
    fields = (
        "id",
        "first_name",
        "last_name",
        "sales_rep",
        "sales_team",
        "installer",
        "project_manager",
        "project_category",
        "stage_bucket",
        "is_clean_deal",
        "is_quick_install",
        "job_status",
        "customer_since",
        "install_date",
        "site_survey_scheduled",
        "site_survey_results",
        "site_survey_submitted",
        "site_survey_approved",
        "crc_date",
        "permit_approved",
        "install_completed",
        "pto_submitted",
        "pto_approved",
    )
    rows = list(qs.order_by("customer_since").values(*fields)[:5000])

    crc_l, ss_crc_l, permit_l, crc_inst_l = [], [], [], []
    sign_inst_l, inst_pto_l, sign_pto_l = [], [], []
    clean_inst_l, notclean_inst_l = [], []
    cust_ss_l, ss_ssr_l, ssr_crc_l = [], [], []
    quick_install_rows = []

    velocity = []
    for r in rows:
        cs = r["customer_since"]
        crc = r["crc_date"]
        ss = r["site_survey_scheduled"]
        ssr = r["site_survey_results"]
        permit = r["permit_approved"]
        inst_d = r["install_date"]
        inst_c = r["install_completed"]
        inst_done = _install_milestone_end(inst_c, inst_d)
        pto = r["pto_submitted"]

        d_crc = _days_span(cs, crc)
        if d_crc is not None:
            crc_l.append(d_crc)
        d_ss_crc = _days_span(ss, crc)
        if d_ss_crc is not None:
            ss_crc_l.append(d_ss_crc)
        d_perm = _days_span(cs, permit)
        if d_perm is not None:
            permit_l.append(d_perm)
        d_crc_inst = _days_span(crc, inst_done)
        if d_crc_inst is not None:
            crc_inst_l.append(d_crc_inst)
        d_sign_inst = _days_span(cs, inst_d)
        if d_sign_inst is not None:
            sign_inst_l.append(d_sign_inst)
            if r["is_clean_deal"]:
                clean_inst_l.append(d_sign_inst)
            else:
                notclean_inst_l.append(d_sign_inst)
        d_inst_pto = _days_span(inst_done, pto)
        if d_inst_pto is not None:
            inst_pto_l.append(d_inst_pto)
        d_sign_pto = _days_span(cs, pto)
        if d_sign_pto is not None:
            sign_pto_l.append(d_sign_pto)
        d_cust_ss = _days_span(cs, ss)
        if d_cust_ss is not None:
            cust_ss_l.append(d_cust_ss)
        d_ss_ssr = _days_span(ss, ssr)
        if d_ss_ssr is not None:
            ss_ssr_l.append(d_ss_ssr)
        d_ssr_crc = _days_span(ssr, crc)
        if d_ssr_crc is not None:
            ssr_crc_l.append(d_ssr_crc)

        project_age = (today - cs).days if cs else None
        velocity_row = {
            "id": r["id"],
            "firstName": (r["first_name"] or "").strip() or None,
            "lastName": (r["last_name"] or "").strip() or None,
            "salesRep": r["sales_rep"],
            "salesTeam": r["sales_team"],
            "installer": r["installer"],
            "projectManager": r["project_manager"] or "",
            "projectCategory": r["project_category"],
            "stageBucket": r["stage_bucket"] or "",
            "isCleanDeal": 1 if r["is_clean_deal"] else 0,
            "isQuickInstall": 1 if r["is_quick_install"] else 0,
            "jobStatus": r["job_status"],
            "customerSince": cs.isoformat() if cs else None,
            "daysToCrc": d_crc,
            "daysSsToCrc": d_ss_crc,
            "daysCustomerToSs": d_cust_ss,
            "daysSsToSsr": d_ss_ssr,
            "daysSsrToCrc": d_ssr_crc,
            "daysToPermit": d_perm,
            "daysToInstall": d_sign_inst,
            "daysCrcToInstall": d_crc_inst,
            "daysInstallToPto": d_inst_pto,
            "daysToPtoSubmitted": d_sign_pto,
            "projectAgeDays": project_age,
        }
        velocity.append(velocity_row)
        if r["is_quick_install"] and d_sign_inst is not None:
            quick_install_rows.append(
                {
                    **velocity_row,
                    "installDate": (inst_done.isoformat() if inst_done else None),
                }
            )

    averages = {
        "avgDaysToCrc": _mean_rounded(crc_l),
        "avgDaysSsToCrc": _mean_rounded(ss_crc_l),
        "avgDaysCustomerToSs": _mean_rounded(cust_ss_l),
        "avgDaysSsToSsr": _mean_rounded(ss_ssr_l),
        "avgDaysSsrToCrc": _mean_rounded(ssr_crc_l),
        "avgDaysToPermit": _mean_rounded(permit_l),
        "avgDaysCrcToInstall": _mean_rounded(crc_inst_l),
        "avgDaysToInstall": _mean_rounded(sign_inst_l),
        "avgDaysInstallToPto": _mean_rounded(inst_pto_l),
        "avgDaysToPtoSubmitted": _mean_rounded(sign_pto_l),
        "avgInstallClean": _mean_rounded(clean_inst_l),
        "avgInstallNotClean": _mean_rounded(notclean_inst_l),
    }

    funnel = _build_pipeline_funnel(qs)
    crc_analytics = _build_crc_analytics(averages, crc_l)
    deals_pipeline = _build_deals_pipeline(qs)
    quick_installs = _build_quick_installs(qs, quick_install_rows)

    return {
        "velocity": velocity,
        "averages": averages,
        "funnel": funnel,
        "crcAnalytics": crc_analytics,
        "dealsPipeline": deals_pipeline,
        "quickInstalls": quick_installs,
    }


def _build_pipeline_funnel(qs):
    """
    Stage-bucket counts in canonical funnel order. Each step also reports
    `pctOfPrev` (conversion vs the previous step) so the UI can render a
    proper funnel without duplicating logic.
    """
    raw = dict(
        qs.exclude(stage_bucket="")
        .values_list("stage_bucket")
        .annotate(c=Count("id"))
    )
    out = []
    prev = None
    for stage in _PIPELINE_FUNNEL_ORDER:
        count = int(raw.get(stage) or raw.get((stage,), 0) or 0)
        if not count:
            for k, v in raw.items():
                if isinstance(k, (tuple, list)) and k and k[0] == stage:
                    count = int(v)
                    break
        out.append(
            {
                "stage": stage,
                "count": count,
                "pctOfPrev": _pct(count, prev) if prev else None,
            }
        )
        prev = count
    return out


def _build_crc_analytics(averages, crc_l):
    series = [
        {"label": "Customer → SS", "days": averages.get("avgDaysCustomerToSs"), "n": None},
        {"label": "SS → SSR", "days": averages.get("avgDaysSsToSsr"), "n": None},
        {"label": "SSR → CRC", "days": averages.get("avgDaysSsrToCrc"), "n": None},
        {"label": "Customer → CRC", "days": averages.get("avgDaysToCrc"), "n": len(crc_l)},
    ]
    return {
        "kpis": {
            "avgDaysCustomerToSs": averages.get("avgDaysCustomerToSs"),
            "avgDaysSsToSsr": averages.get("avgDaysSsToSsr"),
            "avgDaysSsrToCrc": averages.get("avgDaysSsrToCrc"),
            "avgDaysToCrc": averages.get("avgDaysToCrc"),
        },
        "series": series,
    }


def _build_deals_pipeline(qs):
    """
    Ordered list of milestone counters used by the 'Deals Pipeline' bar chart.
    The first row ('Sold Projects') is treated by the UI as the denominator
    when computing per-stage percentages.
    """
    stages = [
        ("Sold Projects", qs.exclude(customer_since__isnull=True).count()),
        ("Site Survey", qs.exclude(site_survey_scheduled__isnull=True).count()),
        ("Site Survey Results", qs.exclude(site_survey_results__isnull=True).count()),
        ("CRC", qs.exclude(crc_date__isnull=True).count()),
        ("Permit Approved", qs.exclude(permit_approved__isnull=True).count()),
        ("Install Scheduled", qs.exclude(install_date__isnull=True).count()),
        ("Install Completed", qs.exclude(install_completed__isnull=True).count()),
        ("PTO Submitted", qs.exclude(pto_submitted__isnull=True).count()),
        ("PTO Approved", qs.exclude(pto_approved__isnull=True).count()),
    ]
    return [{"label": label, "count": int(count)} for label, count in stages]


def _build_quick_installs(qs, quick_install_rows):
    total_installed = qs.exclude(install_date__isnull=True).count()
    quick_count = qs.filter(is_quick_install=True).count()
    return {
        "count": quick_count,
        "totalInstalled": total_installed,
        "quickInstallRate": _pct(quick_count, total_installed),
        "items": quick_install_rows[:200],
    }


def get_cx_bundle(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
):
    cx = _cx_qs(date_from, date_to, user)

    inst_clean = (installer or "").strip() if installer else ""
    if inst_clean:
        cx = cx.filter(installer__iexact=inst_clean)

    extra_dims = any(
        v and str(v).strip() for v in (sales_team, lead_source, project_manager)
    )
    if extra_dims:
        proj_qs = base_queryset(
            date_from,
            date_to,
            user,
            installer=installer,
            sales_team=sales_team,
            lead_source=lead_source,
            project_manager=project_manager,
        )
        uuid_set = list(
            proj_qs.exclude(sunbase_job_uuid="")
            .values_list("sunbase_job_uuid", flat=True)
            .distinct()
        )
        cx = cx.filter(sunbase_job_uuid__in=uuid_set) if uuid_set else cx.none()

    total = cx.count()
    reviews = cx.filter(has_review=True).count()
    inspection_passed = cx.filter(inspection_passed__isnull=False).count()
    pto_submitted = cx.filter(pto_submitted__isnull=False).count()
    pto_approved = cx.filter(pto_approved__isnull=False).count()
    inspection_scheduled = cx.filter(inspection_scheduled__isnull=False).count()
    goal_reviews = (total + 1) // 2 if total else 0
    overview = {
        "totalInstalls": total,
        "reviewsCaptured": reviews,
        "reviewCaptureRate": _pct(reviews, total) if total else 0.0,
        "testimonialPotentials": cx.filter(testimonial_potential=True).count(),
        "testimonialsCompleted": cx.filter(testimonial_done=True).count(),
        "modelHomeCount": cx.filter(model_home_program=True).count(),
        "avgInstallToInspection": _round_avg(cx.aggregate(v=Avg("days_install_to_inspection_passed"))["v"]),
        "avgInstallToPtoSubmitted": _round_avg(cx.aggregate(v=Avg("days_install_to_pto_submitted"))["v"]),
        "avgInstallToPtoApproved": _round_avg(cx.aggregate(v=Avg("days_install_to_pto_approved"))["v"]),
        "avgInstallToReview": _round_avg(cx.aggregate(v=Avg("days_install_to_review"))["v"]),
        "avgInspectionScheduledToPassed": _round_avg(cx.aggregate(v=Avg("days_inspection_scheduled_to_passed"))["v"]),
        "inspectionPassedCount": inspection_passed,
        "ptoSubmittedCount": pto_submitted,
        "ptoApprovedCount": pto_approved,
        "inspectionScheduledCount": inspection_scheduled,
        "goalReviews": goal_reviews,
        "reviewGap": max(0, goal_reviews - reviews),
        "inspectionPassedPct": _pct(inspection_passed, total) if total else 0.0,
        "ptoSubmittedPct": _pct(pto_submitted, total) if total else 0.0,
        "ptoApprovedPct": _pct(pto_approved, total) if total else 0.0,
        "installsWithoutInspection": max(0, total - inspection_passed),
        "installsWithoutPto": max(0, total - pto_approved),
        "installsWithoutReview": max(0, total - reviews),
    }

    by_installer = []
    for r in (
        cx.exclude(installer="")
        .values("installer")
        .annotate(
            totalInstalls=Count("id"),
            reviewsCaptured=Count("id", filter=Q(has_review=True)),
            testimonialPotentials=Count("id", filter=Q(testimonial_potential=True)),
            testimonialsCompleted=Count("id", filter=Q(testimonial_done=True)),
            modelHomeCount=Count("id", filter=Q(model_home_program=True)),
            inspectionPassedCount=Count("id", filter=Q(inspection_passed__isnull=False)),
            ptoSubmittedCount=Count("id", filter=Q(pto_submitted__isnull=False)),
            avgInstallToInspection=Avg("days_install_to_inspection_passed"),
            avgInstallToPtoSubmitted=Avg("days_install_to_pto_submitted"),
            avgInstallToPtoApproved=Avg("days_install_to_pto_approved"),
            avgInstallToReview=Avg("days_install_to_review"),
        )
        .order_by("-totalInstalls")[:200]
    ):
        ti = r["totalInstalls"]
        rc = r["reviewsCaptured"]
        by_installer.append(
            {
                "installer": r["installer"],
                "totalInstalls": ti,
                "reviewsCaptured": rc,
                "reviewCaptureRate": _pct(rc, ti),
                "testimonialPotentials": r["testimonialPotentials"],
                "testimonialsCompleted": r["testimonialsCompleted"],
                "modelHomeCount": r["modelHomeCount"],
                "avgInstallToInspection": _round_avg(r.get("avgInstallToInspection")),
                "avgInstallToPtoSubmitted": _round_avg(r.get("avgInstallToPtoSubmitted")),
                "avgInstallToPtoApproved": _round_avg(r.get("avgInstallToPtoApproved")),
                "avgInstallToReview": _round_avg(r.get("avgInstallToReview")),
                "inspectionPassedCount": r["inspectionPassedCount"],
                "ptoSubmittedCount": r["ptoSubmittedCount"],
                "inspectionFailedCount": 0,
            }
        )

    status_breakdown = list(
        cx.values("job_status")
        .annotate(count=Count("id"))
        .order_by("-count")[:100]
    )

    project_list = []
    for row in cx.order_by("-install_date")[:500]:
        project_list.append(
            {
                "id": row.id,
                "firstName": row.first_name,
                "lastName": row.last_name,
                "jobStatus": row.job_status,
                "installer": row.installer,
                "installDate": row.install_date.isoformat() if row.install_date else None,
                "inspectionPassed": row.inspection_passed.isoformat() if row.inspection_passed else None,
                "ptoSubmitted": row.pto_submitted.isoformat() if row.pto_submitted else None,
                "ptoApproved": row.pto_approved.isoformat() if row.pto_approved else None,
                "hasReview": row.has_review,
                "daysInstallToInspectionPassed": row.days_install_to_inspection_passed,
                "daysInstallToPtoApproved": row.days_install_to_pto_approved,
            }
        )

    review_timing = cx.filter(has_review=True, days_install_to_review__isnull=False).aggregate(
        avgDaysToReview=Avg("days_install_to_review"),
        minDaysToReview=Min("days_install_to_review"),
        maxDaysToReview=Max("days_install_to_review"),
    )
    review_timing_out = {
        "avgDaysToReview": _round_avg(review_timing.get("avgDaysToReview")),
        "minDaysToReview": review_timing.get("minDaysToReview"),
        "maxDaysToReview": review_timing.get("maxDaysToReview"),
        "reviewsWithin1Day": cx.filter(days_install_to_review__lte=1).count(),
        "reviewsWithin3Days": cx.filter(days_install_to_review__lte=3).count(),
        "reviewsWithin7Days": cx.filter(days_install_to_review__lte=7).count(),
        "totalReviews": cx.filter(has_review=True).count(),
    }

    timeline_by_installer = [
        {
            "installer": row["installer"],
            "avgInstallToInspection": row["avgInstallToInspection"],
            "avgInstallToPtoSubmitted": row["avgInstallToPtoSubmitted"],
            "avgInstallToPtoApproved": row["avgInstallToPtoApproved"],
            "avgInstallToReview": row["avgInstallToReview"],
        }
        for row in by_installer
    ]

    review_stages = [
        {
            "stage": "Early Post-Install",
            "count": cx.filter(has_review=True, days_install_to_review__lte=1).count(),
        },
        {
            "stage": "Post-Inspection",
            "count": cx.filter(has_review=True, days_install_to_review__gt=1, days_install_to_review__lte=3).count(),
        },
        {
            "stage": "PTO Submit to PTO Approved",
            "count": cx.filter(has_review=True, days_install_to_review__gt=3, days_install_to_review__lte=7).count(),
        },
        {
            "stage": "Inspection to PTO Submit",
            "count": cx.filter(has_review=True, days_install_to_review__gt=7).count(),
        },
    ]
    review_stages = [r for r in review_stages if r["count"] > 0]

    review_details = []
    for row in (
        cx.filter(has_review=True, days_install_to_review__isnull=False)
        .order_by("days_install_to_review", "-review_captured_date")[:500]
    ):
        d = row.days_install_to_review
        if d is None:
            stage = "Unknown"
        elif d <= 1:
            stage = "Early Post-Install"
        elif d <= 3:
            stage = "Post-Inspection"
        elif d <= 7:
            stage = "PTO Submit to PTO Approved"
        else:
            stage = "Inspection to PTO Submit"
        review_details.append(
            {
                "id": row.id,
                "firstName": row.first_name,
                "lastName": row.last_name,
                "installer": row.installer,
                "jobStatus": row.job_status,
                "installDate": row.install_date.isoformat() if row.install_date else None,
                "reviewCapturedDate": row.review_captured_date.isoformat() if row.review_captured_date else None,
                "daysInstallToReview": d,
                "reviewStage": stage,
            }
        )

    return {
        "overview": overview,
        "byInstaller": by_installer,
        "statusBreakdown": status_breakdown,
        "projectList": project_list,
        "timelineByInstaller": timeline_by_installer,
        "reviewTiming": review_timing_out,
        "reviewStages": review_stages,
        "reviewDetails": review_details,
    }


def _manager_overview_counts(appts, doors):
    """Aligns with sunbright-dashboard getManagerOverview (appointment + door rollup)."""
    ta = appts.count()
    sd = appts.filter(stage_category__in=_SHOW_STAGES).count()
    qsd = appts.filter(stage_category__in=("qualified_show", "closed")).count()
    closed = appts.filter(stage_category="closed").count()
    active_closed = appts.filter(deal_stage__in=_ACTIVE_CLOSED_STAGES).count()
    pending = appts.filter(stage_category="pending").count()
    self_set = appts.filter(is_self_set=True).count()
    total_reps = appts.exclude(sales_rep="").values("sales_rep").distinct().count()
    total_teams = appts.exclude(sales_team="").values("sales_team").distinct().count()

    td = doors.count()
    tc = doors.filter(is_contact=True).count()
    canvassers = doors.exclude(canvasser="").values("canvasser").distinct().count()

    return {
        "totalAppointments": ta,
        "totalSitDowns": sd,
        "totalQualifiedSitDowns": qsd,
        "totalClosedDeals": closed,
        "totalActiveClosedDeals": active_closed,
        "totalPendingOutcome": pending,
        "totalSelfSet": self_set,
        "totalReps": total_reps,
        "totalTeams": total_teams,
        "overallSitDownRate": _pct(sd, ta),
        "overallQualifiedSitDownRate": _pct(qsd, ta),
        "overallClosingRate": _pct(closed, sd) if sd else 0.0,
        "overallQualifiedClosingRate": _pct(active_closed, closed) if closed else 0.0,
        "totalDoors": td,
        "totalContacts": tc,
        "overallContactRate": _pct(tc, td) if td else 0.0,
        "totalCanvassers": canvassers,
    }


def _month_key(d):
    return f"{d.year:04d}-{d.month:02d}" if d else None


def _fill_monthly_series(date_from, date_to, mapping):
    """Return [{month, value}] in chronological order, padding zeros across the range."""
    if not mapping and not (date_from and date_to):
        return []
    keys = set(mapping.keys())
    if date_from and date_to:
        y, m = date_from.year, date_from.month
        end_y, end_m = date_to.year, date_to.month
        cursor = []
        while (y, m) <= (end_y, end_m):
            cursor.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                m = 1
                y += 1
        ordered = cursor
    else:
        ordered = sorted(keys)
    return [{"month": k, "value": int(mapping.get(k, 0))} for k in ordered]


def _get_pm_performance_bundle(pq, date_from=None, date_to=None):
    """
    Aggregate operational KPIs by Sunbase Project Manager (job-level Project rows).

    Returns a tuple (rows, kpis) where:
      rows  — per-PM dicts including monthly install / clean-deal series
      kpis  — overall aggregate KPIs across the filtered project queryset
    """
    rows = list(
        pq.exclude(project_manager="").values(
            "project_manager",
            "project_category",
            "is_clean_deal",
            "customer_since",
            "crc_date",
            "site_survey_scheduled",
            "site_survey_results",
            "install_date",
            "install_completed",
        )
    )

    buckets = defaultdict(
        lambda: {
            "totalProjects": 0,
            "activeProjects": 0,
            "cancelledProjects": 0,
            "onHoldProjects": 0,
            "redFlaggedProjects": 0,
            "disqualifiedProjects": 0,
            "cleanDeals": 0,
            "crcReached": 0,
            "installScheduled": 0,
            "installCompleted": 0,
            "crc_days": [],
            "ss_ssr_days": [],
            "install_days": [],
            "installs_by_month": defaultdict(int),
            "clean_deals_by_month": defaultdict(int),
        }
    )

    totals = {
        "totalProjects": 0,
        "activeProjects": 0,
        "cancelledProjects": 0,
        "onHoldProjects": 0,
        "redFlaggedProjects": 0,
        "disqualifiedProjects": 0,
        "cleanDeals": 0,
        "installScheduled": 0,
        "installCompleted": 0,
        "crcReached": 0,
    }

    for r in rows:
        pm = (r.get("project_manager") or "").strip()
        if not pm:
            continue
        b = buckets[pm]
        b["totalProjects"] += 1
        totals["totalProjects"] += 1
        cat = r.get("project_category") or ""
        if cat == "Active":
            b["activeProjects"] += 1
            totals["activeProjects"] += 1
        elif cat == "Cancelled":
            b["cancelledProjects"] += 1
            totals["cancelledProjects"] += 1
        elif cat == "On Hold":
            b["onHoldProjects"] += 1
            totals["onHoldProjects"] += 1
        elif cat == "Red Flagged":
            b["redFlaggedProjects"] += 1
            totals["redFlaggedProjects"] += 1
        elif cat == "Disqualified":
            b["disqualifiedProjects"] += 1
            totals["disqualifiedProjects"] += 1
        if r.get("is_clean_deal"):
            b["cleanDeals"] += 1
            totals["cleanDeals"] += 1

        cs = r.get("customer_since")
        crc = r.get("crc_date")
        if crc:
            b["crcReached"] += 1
            totals["crcReached"] += 1
        if cs and crc and crc >= cs:
            b["crc_days"].append((crc - cs).days)

        ss = r.get("site_survey_scheduled")
        ssr = r.get("site_survey_results")
        if ss and ssr and ssr >= ss:
            b["ss_ssr_days"].append((ssr - ss).days)

        ins = r.get("install_date")
        ins_done = r.get("install_completed")
        if ins:
            b["installScheduled"] += 1
            totals["installScheduled"] += 1
        if ins_done:
            b["installCompleted"] += 1
            totals["installCompleted"] += 1
        if cs and ins and ins >= cs:
            b["install_days"].append((ins - cs).days)

        install_month_src = ins_done or ins
        if install_month_src:
            mk = _month_key(install_month_src)
            if mk:
                b["installs_by_month"][mk] += 1
        if r.get("is_clean_deal") and cs:
            mk = _month_key(cs)
            if mk:
                b["clean_deals_by_month"][mk] += 1

    out = []
    for pm, b in buckets.items():
        t = b["totalProjects"]
        bad = b["cancelledProjects"] + b["onHoldProjects"] + b["redFlaggedProjects"]
        out.append(
            {
                "projectManager": pm,
                "totalProjects": t,
                "activeProjects": b["activeProjects"],
                "cancelledProjects": b["cancelledProjects"],
                "onHoldProjects": b["onHoldProjects"],
                "redFlaggedProjects": b["redFlaggedProjects"],
                "disqualifiedProjects": b["disqualifiedProjects"],
                "cleanDeals": b["cleanDeals"],
                "cleanDealPct": _pct(b["cleanDeals"], t),
                "cancellationRate": _pct(b["cancelledProjects"], t),
                "netRetentionRate": _pct(t - bad, t) if t else 0.0,
                "crcReachedCount": b["crcReached"],
                "installScheduledCount": b["installScheduled"],
                "installCompletedCount": b["installCompleted"],
                "avgDaysToCrc": _mean_rounded(b["crc_days"]),
                "avgDaysSsToSsr": _mean_rounded(b["ss_ssr_days"]),
                "avgDaysToInstall": _mean_rounded(b["install_days"]),
                "monthlyInstalls": _fill_monthly_series(date_from, date_to, b["installs_by_month"]),
                "monthlyCleanDeals": _fill_monthly_series(date_from, date_to, b["clean_deals_by_month"]),
            }
        )

    out.sort(key=lambda x: -x["totalProjects"])
    out = out[:200]

    t = totals["totalProjects"]
    kpis = {
        "totalProjects": t,
        "activeDeals": totals["activeProjects"],
        "cancelled": totals["cancelledProjects"],
        "cancelledPct": _pct(totals["cancelledProjects"], t),
        "onHold": totals["onHoldProjects"],
        "onHoldPct": _pct(totals["onHoldProjects"], t),
        "redFlagged": totals["redFlaggedProjects"],
        "redFlaggedPct": _pct(totals["redFlaggedProjects"], t),
        "cleanDeals": totals["cleanDeals"],
        "cleanDealsPct": _pct(totals["cleanDeals"], t),
        "installScheduled": totals["installScheduled"],
        "installScheduledPct": _pct(totals["installScheduled"], t),
        "installCompleted": totals["installCompleted"],
        "installCompletedPct": _pct(totals["installCompleted"], t),
        "crcReached": totals["crcReached"],
        "crcReachedPct": _pct(totals["crcReached"], t),
    }
    return out, kpis


def get_manager_bundle(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
):
    appts = _appt_qs(date_from, date_to, user)
    if sales_team and str(sales_team).strip():
        appts = appts.filter(sales_team__iexact=str(sales_team).strip())
    doors = _door_qs(date_from, date_to, user)

    rep_rows = (
        appts.exclude(sales_rep="")
        .values("sales_rep", "sales_team")
        .annotate(
            totalAppointments=Count("id"),
            sitDowns=Count("id", filter=Q(stage_category__in=_SHOW_STAGES)),
            qualifiedSitDowns=Count(
                "id", filter=Q(stage_category__in=("qualified_show", "closed"))
            ),
            closedDeals=Count("id", filter=Q(stage_category="closed")),
            activeClosedDeals=Count(
                "id",
                filter=Q(deal_stage__in=_ACTIVE_CLOSED_STAGES),
            ),
            cancelledDeals=Count("id", filter=Q(deal_stage__in=_CANCEL_STAGES)),
            selfSetCount=Count("id", filter=Q(is_self_set=True)),
            assignedCount=Count("id", filter=Q(is_self_set=False)),
            pendingOutcome=Count("id", filter=Q(stage_category="pending")),
        )
        .order_by("-closedDeals", "-totalAppointments")[:200]
    )
    rep_performance = []
    for r in rep_rows:
        ta = r["totalAppointments"]
        sd = r["sitDowns"]
        cl = r["closedDeals"]
        acl = r["activeClosedDeals"]
        rep_performance.append(
            {
                "salesRep": r["sales_rep"],
                "salesTeam": r["sales_team"] or "",
                "totalAppointments": ta,
                "sitDowns": sd,
                "qualifiedSitDowns": r["qualifiedSitDowns"],
                "closedDeals": cl,
                "activeClosedDeals": acl,
                "cancelledDeals": r["cancelledDeals"],
                "selfSetCount": r["selfSetCount"],
                "assignedCount": r["assignedCount"],
                "pendingOutcome": r["pendingOutcome"],
                "sitDownRate": _pct(sd, ta),
                "qualifiedSitDownRate": _pct(r["qualifiedSitDowns"], ta),
                "closingRate": _pct(cl, sd) if sd else 0.0,
                "qualifiedClosingRate": _pct(acl, cl) if cl else 0.0,
                "salesTeamDisplay": r["sales_team"] or "",
            }
        )

    team_rows = (
        appts.exclude(sales_team="")
        .values("sales_team")
        .annotate(
            repCount=Count("sales_rep", distinct=True),
            totalAppointments=Count("id"),
            sitDowns=Count("id", filter=Q(stage_category__in=_SHOW_STAGES)),
            qualifiedSitDowns=Count(
                "id", filter=Q(stage_category__in=("qualified_show", "closed"))
            ),
            closedDeals=Count("id", filter=Q(stage_category="closed")),
            activeClosedDeals=Count("id", filter=Q(deal_stage__in=_ACTIVE_CLOSED_STAGES)),
            cancelledDeals=Count("id", filter=Q(deal_stage__in=_CANCEL_STAGES)),
            selfSetCount=Count("id", filter=Q(is_self_set=True)),
            assignedCount=Count("id", filter=Q(is_self_set=False)),
            pendingOutcome=Count("id", filter=Q(stage_category="pending")),
        )
        .order_by("-closedDeals", "-totalAppointments")[:100]
    )
    team_performance = []
    for r in team_rows:
        ta = r["totalAppointments"]
        sd = r["sitDowns"]
        cl = r["closedDeals"]
        acl = r["activeClosedDeals"]
        team_performance.append(
            {
                "salesTeam": r["sales_team"],
                "salesTeamDisplay": r["sales_team"],
                "repCount": r["repCount"],
                "totalAppointments": ta,
                "sitDowns": sd,
                "qualifiedSitDowns": r["qualifiedSitDowns"],
                "closedDeals": cl,
                "activeClosedDeals": acl,
                "cancelledDeals": r["cancelledDeals"],
                "selfSetCount": r["selfSetCount"],
                "assignedCount": r["assignedCount"],
                "pendingOutcome": r["pendingOutcome"],
                "sitDownRate": _pct(sd, ta),
                "qualifiedSitDownRate": _pct(r["qualifiedSitDowns"], ta),
                "closingRate": _pct(cl, sd) if sd else 0.0,
                "qualifiedClosingRate": _pct(acl, cl) if cl else 0.0,
            }
        )

    door_stats = {
        "totalDoors": doors.count(),
        "contacts": doors.filter(is_contact=True).count(),
        "contactRate": _pct(doors.filter(is_contact=True).count(), doors.count())
        if doors.count()
        else 0.0,
    }

    deal_stage_breakdown = list(
        appts.exclude(deal_stage="")
        .values("deal_stage")
        .annotate(count=Count("id"))
        .order_by("-count")[:50]
    )

    pending = []
    for a in appts.filter(stage_category="pending").order_by("appointment_datetime")[:500]:
        pending.append(
            {
                "firstName": a.first_name,
                "lastName": a.last_name,
                "salesRep": a.sales_rep,
                "salesTeam": a.sales_team,
                "dealStage": a.deal_stage,
                "appointmentDateTime": a.appointment_datetime.isoformat()
                if a.appointment_datetime
                else None,
            }
        )

    pq = base_queryset(
        date_from,
        date_to,
        user,
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
    )
    overview = {
        "totalProjects": pq.count(),
        "activeProjects": pq.filter(project_category="Active").count(),
        "cancelledProjects": pq.filter(project_category="Cancelled").count(),
    }

    manager_overview = _manager_overview_counts(appts, doors)
    pm_performance, pm_kpis = _get_pm_performance_bundle(pq, date_from, date_to)

    return {
        "overview": overview,
        "managerOverview": manager_overview,
        "repPerformance": rep_performance,
        "teamPerformance": team_performance,
        "doorStats": door_stats,
        "dealStageBreakdown": deal_stage_breakdown,
        "pendingOutcome": pending,
        "pmPerformance": pm_performance,
        "pmKpis": pm_kpis,
    }


def _scoped_sunbase_users(user):
    """Sunbase directory rows visible under the same dashboard data scope as facts."""
    qs = SunbaseUser.objects.all()
    if user is None or getattr(user, "is_staff", False) or not getattr(user, "is_authenticated", False):
        return qs
    try:
        ds = user.dashboard_scope
    except DashboardDataScope.DoesNotExist:
        return SunbaseUser.objects.none()

    kind = ds.scope_kind
    if kind == DashboardDataScope.ScopeKind.TEAM:
        if not (ds.sales_team or "").strip():
            return SunbaseUser.objects.none()
        t = ds.sales_team.strip()
        return qs.filter(Q(crew_name__iexact=t) | Q(team__name__iexact=t))

    if kind == DashboardDataScope.ScopeKind.TEAMS:
        names = [n.strip() for n in (ds.sales_teams or []) if isinstance(n, str) and n.strip()]
        if not names:
            return SunbaseUser.objects.none()
        team_q = reduce(or_, (Q(crew_name__iexact=n) | Q(team__name__iexact=n) for n in names))
        return qs.filter(team_q)

    if kind == DashboardDataScope.ScopeKind.REP:
        if not (ds.sales_rep or "").strip():
            return SunbaseUser.objects.none()
        return qs.filter(full_name__iexact=ds.sales_rep.strip())

    return SunbaseUser.objects.none()


def _total_numeric_rows(rows, keys):
    out = {k: 0 for k in keys}
    for r in rows:
        for k in keys:
            out[k] += int(r.get(k) or 0)
    return out


def _name_key(name):
    return (name or "").strip().lower()


def get_role_performance_bundle(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
):
    """
    Role-based agent tables aligned with Sunbase Users (role, crew).
    - Setters: Sunbase role contains "setter"; activity matched on Door.canvasser, Appointment.setter, Project.setter.
    - Closers: Sunbase role is Sales or contains "closer"; activity on Appointment.sales_rep (self vs assigned split).
    Name matching is case-insensitive exact on full name strings (aggregated with Lower() for performance).

    Optional dimension filters mirror the Manager Performance / Pipeline endpoints
    so the global filter bar can narrow these tables consistently.
    """
    doors = _door_qs(date_from, date_to, user)
    appts = _appt_qs(date_from, date_to, user)
    projects = base_queryset(
        date_from,
        date_to,
        user,
        installer=installer,
        sales_team=sales_team,
        lead_source=lead_source,
        project_manager=project_manager,
    )

    sales_team_clean = (sales_team or "").strip() if sales_team else ""
    if sales_team_clean:
        appts = appts.filter(sales_team__iexact=sales_team_clean)
    lead_source_clean = (lead_source or "").strip() if lead_source else ""
    if lead_source_clean:
        appts = appts.filter(lead_source__iexact=lead_source_clean)

    appt_doorish_q = (
        Q(lead_source__icontains="door")
        | Q(lead_source__icontains="d2d")
        | Q(lead_source__icontains="canvass")
        | Q(lead_source__icontains="field")
    )

    door_by_name = {}
    for row in (
        doors.exclude(canvasser="")
        .annotate(_lk=Lower("canvasser"))
        .values("_lk")
        .annotate(all_doors=Count("id"), contacts_made=Count("id", filter=Q(is_contact=True)))
    ):
        door_by_name[row["_lk"]] = row

    appt_by_setter = {}
    for row in (
        appts.exclude(setter="")
        .annotate(_lk=Lower("setter"))
        .values("_lk")
        .annotate(
            appointments=Count("id"),
            appointment_doors_heuristic=Count("id", filter=appt_doorish_q),
            sitdowns=Count("id", filter=Q(stage_category__in=_SHOW_STAGES)),
            qfd_sits=Count("id", filter=Q(stage_category__in=("qualified_show", "closed"))),
        )
    ):
        appt_by_setter[row["_lk"]] = row

    proj_by_setter = {}
    for row in (
        projects.exclude(setter="")
        .annotate(_lk=Lower("setter"))
        .values("_lk")
        .annotate(
            all_jobs=Count("id"),
            jobs_cancelled=Count("id", filter=Q(project_category="Cancelled")),
            active_deals=Count("id", filter=Q(project_category="Active")),
            jobs_on_hold=Count("id", filter=Q(project_category="On Hold")),
            jobs_red_flagged=Count("id", filter=Q(project_category="Red Flagged")),
            clean_deals=Count("id", filter=Q(is_clean_deal=True)),
            total_install=Count(
                "id",
                filter=Q(install_date__isnull=False) | Q(install_completed__isnull=False),
            ),
        )
    ):
        proj_by_setter[row["_lk"]] = row

    proj_by_rep = {}
    for row in (
        projects.exclude(sales_rep="")
        .annotate(_lk=Lower("sales_rep"))
        .values("_lk")
        .annotate(
            all_jobs=Count("id"),
            jobs_cancelled=Count("id", filter=Q(project_category="Cancelled")),
            active_deals=Count("id", filter=Q(project_category="Active")),
            jobs_on_hold=Count("id", filter=Q(project_category="On Hold")),
            jobs_red_flagged=Count("id", filter=Q(project_category="Red Flagged")),
            clean_deals=Count("id", filter=Q(is_clean_deal=True)),
            total_install=Count(
                "id",
                filter=Q(install_date__isnull=False) | Q(install_completed__isnull=False),
            ),
        )
    ):
        proj_by_rep[row["_lk"]] = row

    appt_by_rep = {}
    for row in (
        appts.exclude(sales_rep="")
        .annotate(_lk=Lower("sales_rep"))
        .values("_lk")
        .annotate(
            appointments_self_gen=Count("id", filter=Q(is_self_set=True)),
            appointments_lead_gen=Count("id", filter=Q(is_self_set=False)),
            total_appointments=Count("id"),
            sits_self_gen=Count("id", filter=Q(is_self_set=True, stage_category__in=_SHOW_STAGES)),
            sits_lead_gen=Count("id", filter=Q(is_self_set=False, stage_category__in=_SHOW_STAGES)),
            total_sits=Count("id", filter=Q(stage_category__in=_SHOW_STAGES)),
            qfd_self=Count("id", filter=Q(is_self_set=True, stage_category__in=("qualified_show", "closed"))),
            qfd_lead=Count("id", filter=Q(is_self_set=False, stage_category__in=("qualified_show", "closed"))),
            total_qfd=Count("id", filter=Q(stage_category__in=("qualified_show", "closed"))),
            deals_self=Count("id", filter=Q(is_self_set=True, stage_category="closed")),
            deals_lead=Count("id", filter=Q(is_self_set=False, stage_category="closed")),
        )
    ):
        appt_by_rep[row["_lk"]] = row

    sunbase_qs = _scoped_sunbase_users(user).exclude(full_name="").exclude(role__iexact="admin")

    setter_users = sunbase_qs.filter(role__icontains="setter")
    setter_rows = []
    for su in setter_users.order_by("full_name")[:400]:
        name = (su.full_name or "").strip()
        if not name:
            continue
        lk = _name_key(name)
        dd = door_by_name.get(lk) or {}
        ad = appt_by_setter.get(lk) or {}
        pd = proj_by_setter.get(lk) or {}

        all_doors = int(dd.get("all_doors") or 0)
        contacts_made = int(dd.get("contacts_made") or 0)
        appointments = int(ad.get("appointments") or 0)
        appts_doors_heuristic = int(ad.get("appointment_doors_heuristic") or 0)
        sitdowns = int(ad.get("sitdowns") or 0)
        qfd_sits = int(ad.get("qfd_sits") or 0)
        all_jobs = int(pd.get("all_jobs") or 0)
        jobs_cancelled = int(pd.get("jobs_cancelled") or 0)
        active_deals = int(pd.get("active_deals") or 0)
        jobs_on_hold = int(pd.get("jobs_on_hold") or 0)
        jobs_red_flagged = int(pd.get("jobs_red_flagged") or 0)
        clean_deals = int(pd.get("clean_deals") or 0)
        total_install = int(pd.get("total_install") or 0)

        contact_rate = _pct(contacts_made, all_doors)
        appt_sched_ratio = _pct(appointments, contacts_made) if contacts_made else _pct(appointments, all_doors)
        sit_down_rate = _pct(sitdowns, appointments) if appointments else 0.0
        clean_pct = _pct(clean_deals, all_jobs) if all_jobs else 0.0
        cancel_pct = _pct(jobs_cancelled, all_jobs) if all_jobs else 0.0
        retention_pct = (
            _pct(all_jobs - (jobs_cancelled + jobs_on_hold + jobs_red_flagged), all_jobs)
            if all_jobs
            else 0.0
        )

        setter_rows.append(
            {
                "agent": name,
                "role": su.role or "",
                "crew": su.crew_name or "",
                "allDoors": all_doors,
                "contactsMade": contacts_made,
                "appointmentDoorsHeuristic": appts_doors_heuristic,
                "appointments": appointments,
                "sitdowns": sitdowns,
                "qfdSitdowns": qfd_sits,
                "allJobs": all_jobs,
                "jobsCancelled": jobs_cancelled,
                "activeDeals": active_deals,
                "jobsOnHold": jobs_on_hold,
                "jobsRedFlagged": jobs_red_flagged,
                "cleanDeals": clean_deals,
                "totalInstall": total_install,
                "contactRate": contact_rate,
                "apptSchedRatio": appt_sched_ratio,
                "sitDownRate": sit_down_rate,
                "cleanPct": clean_pct,
                "cancellationRate": cancel_pct,
                "netRetentionRate": retention_pct,
            }
        )

    closer_users = sunbase_qs.filter(Q(role__iexact="Sales") | Q(role__icontains="closer")).exclude(
        role__icontains="setter"
    )
    closer_rows = []
    for su in closer_users.order_by("full_name")[:400]:
        name = (su.full_name or "").strip()
        if not name:
            continue
        lk = _name_key(name)
        dd = door_by_name.get(lk) or {}
        rd = appt_by_rep.get(lk) or {}
        pr = proj_by_rep.get(lk) or {}

        all_doors = int(dd.get("all_doors") or 0)
        contacts_made = int(dd.get("contacts_made") or 0)
        appts_self = int(rd.get("appointments_self_gen") or 0)
        appts_lead = int(rd.get("appointments_lead_gen") or 0)
        total_appts = int(rd.get("total_appointments") or 0)
        sits_self = int(rd.get("sits_self_gen") or 0)
        sits_lead = int(rd.get("sits_lead_gen") or 0)
        total_sits = int(rd.get("total_sits") or 0)
        qfd_self = int(rd.get("qfd_self") or 0)
        qfd_lead = int(rd.get("qfd_lead") or 0)
        total_qfd = int(rd.get("total_qfd") or 0)
        deals_self = int(rd.get("deals_self") or 0)
        deals_lead = int(rd.get("deals_lead") or 0)
        total_deals = deals_self + deals_lead

        all_jobs = int(pr.get("all_jobs") or 0)
        jobs_cancelled = int(pr.get("jobs_cancelled") or 0)
        active_deals = int(pr.get("active_deals") or 0)
        jobs_on_hold = int(pr.get("jobs_on_hold") or 0)
        jobs_red_flagged = int(pr.get("jobs_red_flagged") or 0)
        clean_deals = int(pr.get("clean_deals") or 0)
        total_install = int(pr.get("total_install") or 0)

        sit_down_rate = _pct(total_sits, total_appts) if total_appts else 0.0
        closing_rate = _pct(total_deals, total_sits) if total_sits else 0.0
        clean_pct = _pct(clean_deals, all_jobs) if all_jobs else 0.0
        cancel_pct = _pct(jobs_cancelled, all_jobs) if all_jobs else 0.0
        retention_pct = (
            _pct(all_jobs - (jobs_cancelled + jobs_on_hold + jobs_red_flagged), all_jobs)
            if all_jobs
            else 0.0
        )

        closer_rows.append(
            {
                "agent": name,
                "role": su.role or "",
                "crew": su.crew_name or "",
                "allDoors": all_doors,
                "contactsMade": contacts_made,
                "appointmentsSelfGen": appts_self,
                "appointmentsLeadGen": appts_lead,
                "totalAppointments": total_appts,
                "sitsSelfGen": sits_self,
                "sitsLeadGen": sits_lead,
                "totalSits": total_sits,
                "qfdSitsSelfGen": qfd_self,
                "qfdSitsLeadGen": qfd_lead,
                "totalQfdSits": total_qfd,
                "dealsSelfGen": deals_self,
                "dealsLeadGen": deals_lead,
                "totalDeals": total_deals,
                "allJobs": all_jobs,
                "jobsCancelled": jobs_cancelled,
                "activeDeals": active_deals,
                "jobsOnHold": jobs_on_hold,
                "jobsRedFlagged": jobs_red_flagged,
                "cleanDeals": clean_deals,
                "totalInstall": total_install,
                "sitDownRate": sit_down_rate,
                "closingRate": closing_rate,
                "cleanPct": clean_pct,
                "cancellationRate": cancel_pct,
                "netRetentionRate": retention_pct,
            }
        )

    setter_num_keys = [
        "allDoors",
        "contactsMade",
        "appointmentDoorsHeuristic",
        "appointments",
        "sitdowns",
        "qfdSitdowns",
        "allJobs",
        "jobsCancelled",
        "activeDeals",
        "jobsOnHold",
        "jobsRedFlagged",
        "cleanDeals",
        "totalInstall",
    ]
    closer_num_keys = [
        "allDoors",
        "contactsMade",
        "appointmentsSelfGen",
        "appointmentsLeadGen",
        "totalAppointments",
        "sitsSelfGen",
        "sitsLeadGen",
        "totalSits",
        "qfdSitsSelfGen",
        "qfdSitsLeadGen",
        "totalQfdSits",
        "dealsSelfGen",
        "dealsLeadGen",
        "totalDeals",
        "allJobs",
        "jobsCancelled",
        "activeDeals",
        "jobsOnHold",
        "jobsRedFlagged",
        "cleanDeals",
        "totalInstall",
    ]

    setter_totals = _total_numeric_rows(setter_rows, setter_num_keys)
    closer_totals = _total_numeric_rows(closer_rows, closer_num_keys)

    setter_totals_row = {"agent": "Total", **setter_totals}
    setter_totals_row["contactRate"] = _pct(setter_totals["contactsMade"], setter_totals["allDoors"])
    setter_totals_row["apptSchedRatio"] = (
        _pct(setter_totals["appointments"], setter_totals["contactsMade"])
        if setter_totals["contactsMade"]
        else _pct(setter_totals["appointments"], setter_totals["allDoors"])
    )
    setter_totals_row["sitDownRate"] = (
        _pct(setter_totals["sitdowns"], setter_totals["appointments"]) if setter_totals["appointments"] else 0.0
    )
    s_all_jobs = setter_totals["allJobs"]
    setter_totals_row["cleanPct"] = _pct(setter_totals["cleanDeals"], s_all_jobs) if s_all_jobs else 0.0
    setter_totals_row["cancellationRate"] = _pct(setter_totals["jobsCancelled"], s_all_jobs) if s_all_jobs else 0.0
    setter_totals_row["netRetentionRate"] = (
        _pct(
            s_all_jobs
            - (
                setter_totals["jobsCancelled"]
                + setter_totals["jobsOnHold"]
                + setter_totals["jobsRedFlagged"]
            ),
            s_all_jobs,
        )
        if s_all_jobs
        else 0.0
    )
    setter_totals_row["role"] = ""
    setter_totals_row["crew"] = ""

    closer_totals_row = {"agent": "Total", **closer_totals}
    c_all_jobs = closer_totals["allJobs"]
    c_total_appts = closer_totals["totalAppointments"]
    c_total_sits = closer_totals["totalSits"]
    closer_totals_row["sitDownRate"] = _pct(c_total_sits, c_total_appts) if c_total_appts else 0.0
    closer_totals_row["closingRate"] = _pct(closer_totals["totalDeals"], c_total_sits) if c_total_sits else 0.0
    closer_totals_row["cleanPct"] = _pct(closer_totals["cleanDeals"], c_all_jobs) if c_all_jobs else 0.0
    closer_totals_row["cancellationRate"] = _pct(closer_totals["jobsCancelled"], c_all_jobs) if c_all_jobs else 0.0
    closer_totals_row["netRetentionRate"] = (
        _pct(
            c_all_jobs
            - (
                closer_totals["jobsCancelled"]
                + closer_totals["jobsOnHold"]
                + closer_totals["jobsRedFlagged"]
            ),
            c_all_jobs,
        )
        if c_all_jobs
        else 0.0
    )
    closer_totals_row["role"] = ""
    closer_totals_row["crew"] = ""

    return {
        "setters": setter_rows,
        "setterTotals": setter_totals_row,
        "closers": closer_rows,
        "closerTotals": closer_totals_row,
        "notes": [
            'Rows come from Sunbase Users (General - Users Report) filtered by role: setters contain "setter"; closers are Sales or role contains "closer" (excluding setters).',
            "Facts are matched on Sunbase user's Fullname ↔ Door canvasser / Appointment.setter or sales_rep / Project.setter.",
            "appointmentDoorsHeuristic counts setter appointments whose lead_source text suggests door/canvass; refine if you add explicit flags from Sunbase.",
            "Refresh Job List sync after deploying Project.setter so job columns populate.",
        ],
    }
