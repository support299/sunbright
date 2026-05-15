"""Distinct-value lookups used to populate the global dashboard filter dropdowns.

Each list is bounded to keep the response small and is sorted alphabetically.
Empty strings are filtered out so the UI does not render blank options.
"""

from dashboard.models import Appointment, Installer, Project, SunbaseUser
from dashboard.scope import appointment_scope_q, project_scope_q

_PROJECT_LIMIT = 500
_APPT_LIMIT = 500


def _project_qs(user):
    return Project.objects.filter(deleted_at__isnull=True).filter(project_scope_q(user))


def _appt_qs(user):
    return Appointment.objects.filter(deleted_at__isnull=True).filter(appointment_scope_q(user))


def _distinct(qs, field, limit):
    return [
        v
        for v in qs.exclude(**{f"{field}__exact": ""})
        .values_list(field, flat=True)
        .distinct()
        .order_by(field)[:limit]
        if v
    ]


def get_filter_options(user=None):
    pq = _project_qs(user)
    aq = _appt_qs(user)

    teams = sorted(set(_distinct(pq, "sales_team", _PROJECT_LIMIT)) | set(_distinct(aq, "sales_team", _APPT_LIMIT)))
    installers_text = _distinct(pq, "installer", _PROJECT_LIMIT)
    ref_installers = [n for n in Installer.objects.values_list("name", flat=True).order_by("name")[:_PROJECT_LIMIT] if n]
    installers = sorted(set(installers_text) | set(ref_installers))
    sales_reps = sorted(set(_distinct(pq, "sales_rep", _PROJECT_LIMIT)) | set(_distinct(aq, "sales_rep", _APPT_LIMIT)))
    setters_list = sorted(set(_distinct(pq, "setter", _PROJECT_LIMIT)) | set(_distinct(aq, "setter", _APPT_LIMIT)))
    lead_sources = sorted(
        set(_distinct(pq, "lead_source", _PROJECT_LIMIT)) | set(_distinct(aq, "lead_source", _APPT_LIMIT))
    )

    project_managers: list[str] = []
    markets: list[str] = []
    # `project_manager` and `market` are Phase-1 schema additions; expose them
    # only if/when they exist so the UI can render the dropdowns gracefully.
    if hasattr(Project, "project_manager"):
        project_managers = _distinct(pq, "project_manager", _PROJECT_LIMIT)
    sunbase_mgr = sorted(
        {
            (m or "").strip()
            for m in SunbaseUser.objects.exclude(manager_name="")
            .values_list("manager_name", flat=True)
            .distinct()
            if m and str(m).strip()
        }
    )[:400]
    project_managers = sorted(set(project_managers) | set(sunbase_mgr))
    if hasattr(Project, "market"):
        markets = _distinct(pq, "market", _PROJECT_LIMIT)

    return {
        "salesTeams": teams,
        "installers": installers,
        "leadSources": lead_sources,
        "projectManagers": project_managers,
        "markets": markets,
        "salesReps": sales_reps,
        "setters": setters_list,
    }
