"""Single source of truth for analytics metrics.

Every KPI rendered in the React dashboards or referenced by AI Insights
should have a matching entry in :mod:`dashboard.metrics.registry`.
"""

from dashboard.metrics.registry import (
    METRIC_REGISTRY,
    MetricDefinition,
    MetricStatus,
    get_metric,
    list_metrics,
    metrics_by_status,
)

__all__ = [
    "METRIC_REGISTRY",
    "MetricDefinition",
    "MetricStatus",
    "get_metric",
    "list_metrics",
    "metrics_by_status",
]
