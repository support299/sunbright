"""
Row-level data scope for non-admin users (sales team / rep).
Staff users always see organization-wide data; scope profile is ignored for them.
"""
from functools import reduce
from operator import or_

from django.db.models import Q

from dashboard.models import DashboardDataScope, SunbaseUser


def _or_team_q(field: str, names: list[str]) -> Q:
    cleaned = [n.strip() for n in names if isinstance(n, str) and n.strip()]
    if not cleaned:
        return Q(pk__in=[])
    return reduce(or_, (Q(**{f"{field}__iexact": n}) for n in cleaned))


def project_scope_q(user) -> Q:
    """Extra filter for Project querysets (combined with date/deleted filters)."""
    if user is None or getattr(user, "is_staff", False) or not getattr(user, "is_authenticated", False):
        return Q()
    try:
        ds = user.dashboard_scope
    except DashboardDataScope.DoesNotExist:
        return Q(pk__in=[])

    kind = ds.scope_kind
    if kind == DashboardDataScope.ScopeKind.TEAM:
        if not (ds.sales_team or "").strip():
            return Q(pk__in=[])
        return Q(sales_team__iexact=ds.sales_team.strip())

    if kind == DashboardDataScope.ScopeKind.TEAMS:
        return _or_team_q("sales_team", list(ds.sales_teams or []))

    if kind == DashboardDataScope.ScopeKind.REP:
        if not (ds.sales_rep or "").strip():
            return Q(pk__in=[])
        return Q(sales_rep__iexact=ds.sales_rep.strip())

    return Q(pk__in=[])


def appointment_scope_q(user) -> Q:
    """Appointments: team filters on sales_team; rep on sales_rep OR setter."""
    if user is None or getattr(user, "is_staff", False) or not getattr(user, "is_authenticated", False):
        return Q()
    try:
        ds = user.dashboard_scope
    except DashboardDataScope.DoesNotExist:
        return Q(pk__in=[])

    kind = ds.scope_kind
    if kind == DashboardDataScope.ScopeKind.TEAM:
        if not (ds.sales_team or "").strip():
            return Q(pk__in=[])
        t = ds.sales_team.strip()
        return Q(sales_team__iexact=t)

    if kind == DashboardDataScope.ScopeKind.TEAMS:
        cleaned = [n.strip() for n in (ds.sales_teams or []) if isinstance(n, str) and n.strip()]
        if not cleaned:
            return Q(pk__in=[])
        return reduce(or_, (Q(sales_team__iexact=n) for n in cleaned))

    if kind == DashboardDataScope.ScopeKind.REP:
        if not (ds.sales_rep or "").strip():
            return Q(pk__in=[])
        r = ds.sales_rep.strip()
        return Q(sales_rep__iexact=r) | Q(setter__iexact=r)

    return Q(pk__in=[])


def door_scope_q(user) -> Q:
    """Doors only support rep-style scope via canvasser name; team scope has no column — no rows."""
    if user is None or getattr(user, "is_staff", False) or not getattr(user, "is_authenticated", False):
        return Q()
    try:
        ds = user.dashboard_scope
    except DashboardDataScope.DoesNotExist:
        return Q(pk__in=[])

    if ds.scope_kind == DashboardDataScope.ScopeKind.REP:
        if not (ds.sales_rep or "").strip():
            return Q(pk__in=[])
        return Q(canvasser__iexact=ds.sales_rep.strip())

    return Q(pk__in=[])


def cx_scope_q(user) -> Q:
    """CxProject has no sales_team/sales_rep; non-staff users get no CX rows until the model is extended."""
    if user is None or getattr(user, "is_staff", False) or not getattr(user, "is_authenticated", False):
        return Q()
    return Q(pk__in=[])


def scoped_sunbase_users(user):
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
