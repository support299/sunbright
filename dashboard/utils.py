from django.utils.dateparse import parse_date


def _first_scalar(val):
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val


def _collect_filter_values(request):
    """Merged dict of query params + JSON/form body, with list values reduced to first scalar."""
    merged = {}
    qp = getattr(request, "query_params", None)
    if qp is not None:
        merged.update({k: _first_scalar(v) for k, v in qp.items()})
    elif hasattr(request, "GET"):
        merged.update({k: _first_scalar(v) for k, v in request.GET.items()})

    data = getattr(request, "data", None)
    if data is not None:
        keys = (
            "date_from",
            "date_to",
            "dateFrom",
            "dateTo",
            "installer",
            "sales_team",
            "salesTeam",
            "lead_source",
            "leadSource",
            "manager",
            "projectManager",
            "project_manager",
            "market",
            "rep_kind",
            "repKind",
            "rep_name",
            "repName",
        )
        for key in keys:
            if hasattr(data, "get"):
                val = data.get(key)
            elif isinstance(data, dict):
                val = data.get(key)
            else:
                val = None
            val = _first_scalar(val)
            if val not in (None, ""):
                merged[key] = val
    return merged


def _resolve_filter(merged, candidates):
    for key in candidates:
        val = merged.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            return text
    return None


def normalize_rep_kind(raw):
    """API / UI → internal rep dimension for project filtering."""
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("-", "_")
    if s in ("sales_rep", "salesrep", "rep", "closer", "sales", "closer_rep"):
        return "sales_rep"
    if s in ("setter", "appointment_setter"):
        return "setter"
    return None


def parse_dashboard_date_range(request):
    """
    Same semantics as sunbright-dashboard `dateWhere` / `filterParams`:
    filter projects by customer_since (CX uses install_date — see analytics).
    Accepts date_from / date_to (REST) or dateFrom / dateTo (camelCase).
    Merges query string with JSON/form body so POST endpoints (e.g. insights) can pass the same range.
    """
    merged = _collect_filter_values(request)
    raw_from = merged.get("date_from") or merged.get("dateFrom")
    raw_to = merged.get("date_to") or merged.get("dateTo")
    date_from = parse_date(str(raw_from)) if raw_from else None
    date_to = parse_date(str(raw_to)) if raw_to else None
    return date_from, date_to


def parse_dashboard_filters_with_dimensions(request):
    """
    Subset helper kept for callers that only need date + the two most common slicers.
    The full multi-dimensional filter set lives in :func:`parse_dashboard_filters_full`;
    callers that need lead_source / manager / market should switch to that helper.
    """
    date_from, date_to = parse_dashboard_date_range(request)
    merged = _collect_filter_values(request)
    installer = _resolve_filter(merged, ("installer",))
    sales_team = _resolve_filter(merged, ("sales_team", "salesTeam"))
    return date_from, date_to, installer, sales_team


def parse_dashboard_filters_full(request):
    """
    All supported analytics slicers as a single dict.

    Returns
    -------
    dict with keys:
        date_from, date_to, installer, sales_team,
        lead_source, manager, market, rep_kind, rep_name
    """
    date_from, date_to = parse_dashboard_date_range(request)
    merged = _collect_filter_values(request)
    rk = normalize_rep_kind(_resolve_filter(merged, ("rep_kind", "repKind")))
    rn = _resolve_filter(merged, ("rep_name", "repName"))
    return {
        "date_from": date_from,
        "date_to": date_to,
        "installer": _resolve_filter(merged, ("installer",)),
        "sales_team": _resolve_filter(merged, ("sales_team", "salesTeam")),
        "lead_source": _resolve_filter(merged, ("lead_source", "leadSource")),
        "manager": _resolve_filter(merged, ("manager", "projectManager", "project_manager")),
        "market": _resolve_filter(merged, ("market",)),
        "rep_kind": rk,
        "rep_name": rn if rn else None,
    }


def success_response(data):
    return {"success": True, "data": data, "errors": []}


def error_response(errors):
    return {"success": False, "data": None, "errors": errors}
