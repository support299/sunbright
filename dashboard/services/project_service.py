from django.db.models import Case, Count, F, Q, Sum, When

from dashboard.models import Project, SunbaseUser
from dashboard.scope import project_scope_q


def base_queryset(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
):
    qs = Project.objects.filter(deleted_at__isnull=True).filter(project_scope_q(user))
    if date_from:
        qs = qs.filter(customer_since__gte=date_from)
    if date_to:
        qs = qs.filter(customer_since__lte=date_to)
    if installer:
        t = str(installer).strip()
        if t:
            q_inst = Q(installer__iexact=t)
            q_inst |= Q(installer_ref__name__iexact=t)
            qs = qs.filter(q_inst)
    if sales_team:
        t = str(sales_team).strip()
        if t:
            qs = qs.filter(sales_team__iexact=t)
    if lead_source:
        t = str(lead_source).strip()
        if t:
            qs = qs.filter(lead_source__iexact=t)
    if project_manager:
        t = str(project_manager).strip()
        if t:
            mgr_q = Q(project_manager__iexact=t)
            for n in (
                SunbaseUser.objects.filter(manager_name__iexact=t)
                .exclude(full_name="")
                .values_list("full_name", flat=True)
                .distinct()
            ):
                nn = (n or "").strip()
                if nn:
                    mgr_q |= Q(sales_rep__iexact=nn) | Q(setter__iexact=nn)
            qs = qs.filter(mgr_q)
    if market and hasattr(Project, "market"):
        t = str(market).strip()
        if t:
            qs = qs.filter(market__iexact=t)
    rk = (rep_kind or "").strip().lower() if rep_kind else ""
    rn = (rep_name or "").strip() if rep_name else ""
    if rk == "sales_rep" and rn:
        qs = qs.filter(sales_rep__iexact=rn)
    elif rk == "setter" and rn:
        qs = qs.filter(setter__iexact=rn)
    return qs


def get_overview_metrics(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
):
    """KPI block aligned with sunbright-dashboard `getOverviewStats` (fields we can compute today)."""
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
    total = qs.count()
    active = qs.filter(project_category="Active").count()
    cancelled = qs.filter(project_category="Cancelled").count()
    on_hold = qs.filter(project_category="On Hold").count()
    red_flagged = qs.filter(project_category="Red Flagged").count()
    disqualified = qs.filter(project_category="Disqualified").count()
    clean = qs.filter(is_clean_deal=True).count()
    not_clean = max(total - clean, 0)
    clean_deal_pct = round(100.0 * clean / total, 1) if total else 0.0
    cancellation_rate = round(100.0 * cancelled / total, 1) if total else 0.0
    bad_for_retention = cancelled + red_flagged + on_hold
    net_retention_rate = round(100.0 * (total - bad_for_retention) / total, 1) if total else 0.0

    active_pipeline = qs.filter(project_category="Active").aggregate(s=Sum("contract_amount"))["s"]
    total_contract = qs.aggregate(s=Sum("contract_amount"))["s"]
    active_pipeline_value = float(active_pipeline or 0)
    total_contract_value = float(total_contract or 0)

    with_install = qs.filter(install_date__isnull=False, customer_since__isnull=False).values_list(
        "customer_since", "install_date"
    )
    deltas = []
    for customer_since, install_date in with_install:
        if customer_since and install_date and install_date >= customer_since:
            deltas.append((install_date - customer_since).days)
    installed_projects = len(deltas)
    avg_days_to_install = round(sum(deltas) / len(deltas), 1) if deltas else None

    return {
        "totalProjects": total,
        "activeProjects": active,
        "cancelledProjects": cancelled,
        "onHoldProjects": on_hold,
        "redFlaggedProjects": red_flagged,
        "disqualifiedProjects": disqualified,
        "cleanDeals": clean,
        "notCleanDeals": not_clean,
        "cleanDealPct": clean_deal_pct,
        "cancellationRate": cancellation_rate,
        "netRetentionRate": net_retention_rate,
        "activePipelineValue": active_pipeline_value,
        "totalContractValue": total_contract_value,
        "avgDaysToInstall": avg_days_to_install,
        "installedProjects": installed_projects,
    }


def get_category_breakdown(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
):
    return list(
        base_queryset(
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
        .values("project_category")
        .annotate(count=Count("id"))
        .order_by("-count")
    )


def get_on_hold_projects(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
):
    return base_queryset(
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
    ).filter(project_category="On Hold")


def get_cancelled_projects(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
):
    return base_queryset(
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
    ).filter(project_category="Cancelled")


def get_cancellation_reasons_breakdown(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
):
    """Group cancelled projects by parsed cancellation_reason, or full job_status if reason not extracted."""
    rows = (
        base_queryset(
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
        .filter(project_category="Cancelled")
        .annotate(
            reason=Case(
                When(~Q(cancellation_reason=""), then=F("cancellation_reason")),
                default=F("job_status"),
            )
        )
        .values("reason")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    return [{"reason": r["reason"] or "Unknown", "count": r["count"]} for r in rows]


def get_on_hold_reasons_breakdown(
    date_from=None,
    date_to=None,
    user=None,
    *,
    installer=None,
    sales_team=None,
    lead_source=None,
    project_manager=None,
    market=None,
    rep_kind=None,
    rep_name=None,
):
    """Group on-hold projects by parsed on_hold_reason, or full job_status if reason not extracted."""
    rows = (
        base_queryset(
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
        .filter(project_category="On Hold")
        .annotate(
            reason=Case(
                When(~Q(on_hold_reason=""), then=F("on_hold_reason")),
                default=F("job_status"),
            )
        )
        .values("reason")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    return [{"reason": r["reason"] or "Unknown", "count": r["count"]} for r in rows]
