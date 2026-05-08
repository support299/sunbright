"""
Metrics registry.

This module is the **source of truth** for every analytics metric exposed
through the dashboard API.  When a Power BI number does not match the React
dashboard, look up the metric here, compare ``formula`` to the DAX, and fix
the discrepancy in exactly one place.

Conventions
-----------
* ``key`` — the camelCase identifier returned in the JSON payload
  (e.g. ``avgDaysToCrc`` inside ``pipeline.averages``).
* ``business_label`` — the human-readable label rendered in the UI.
* ``power_bi_label`` — the original label in the client's Power BI report;
  may differ slightly from ``business_label``.
* ``source`` — Django model / table the metric is computed from.
* ``fields`` — model fields actually consumed by the formula.
* ``formula`` — short pseudo-code description of the calculation. Keep it
  declarative; the implementation lives in ``services/analytics_service.py``
  and ``services/project_service.py``.
* ``filters`` — additional WHERE-clause-style filters applied beyond the
  global scope (e.g. ``project_category = 'Active'``).
* ``unit`` — display unit (``"days"``, ``"percent"``, ``"count"``, ``"currency"``).
* ``status`` — :class:`MetricStatus`; mark ``CONFIRMED`` only after the
  client signs off on the formula.
* ``notes`` — anything that helps debugging (edge cases, exclusions, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class MetricStatus(str, Enum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    business_label: str
    source: str
    formula: str
    unit: str = "count"
    fields: tuple[str, ...] = field(default_factory=tuple)
    filters: str = ""
    power_bi_label: str = ""
    status: MetricStatus = MetricStatus.DRAFT
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "businessLabel": self.business_label,
            "source": self.source,
            "formula": self.formula,
            "unit": self.unit,
            "fields": list(self.fields),
            "filters": self.filters,
            "powerBiLabel": self.power_bi_label,
            "status": self.status.value,
            "notes": self.notes,
        }


def _m(*args, **kwargs) -> MetricDefinition:
    return MetricDefinition(*args, **kwargs)


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    # ──────────────────────────── Overview / KPIs ────────────────────────────
    "totalProjects": _m(
        key="totalProjects",
        business_label="Total Projects",
        source="Project",
        formula="COUNT(Project) WHERE deleted_at IS NULL AND in date range",
        unit="count",
        power_bi_label="Deals",
        status=MetricStatus.CONFIRMED,
    ),
    "activeProjects": _m(
        key="activeProjects",
        business_label="Active Projects",
        source="Project",
        formula="COUNT(*) WHERE project_category = 'Active'",
        filters="project_category = 'Active'",
        unit="count",
        power_bi_label="Active Deals",
        status=MetricStatus.CONFIRMED,
    ),
    "cancelledProjects": _m(
        key="cancelledProjects",
        business_label="Cancelled Projects",
        source="Project",
        formula="COUNT(*) WHERE project_category = 'Cancelled'",
        filters="project_category = 'Cancelled'",
        unit="count",
        power_bi_label="Cancelled",
        status=MetricStatus.CONFIRMED,
    ),
    "onHoldProjects": _m(
        key="onHoldProjects",
        business_label="On Hold Projects",
        source="Project",
        formula="COUNT(*) WHERE project_category = 'On Hold'",
        filters="project_category = 'On Hold'",
        unit="count",
        power_bi_label="On Hold",
        status=MetricStatus.CONFIRMED,
    ),
    "redFlaggedProjects": _m(
        key="redFlaggedProjects",
        business_label="Red Flagged Projects",
        source="Project",
        formula="COUNT(*) WHERE project_category = 'Red Flagged'",
        filters="project_category = 'Red Flagged'",
        unit="count",
        status=MetricStatus.CONFIRMED,
    ),
    "cleanDeals": _m(
        key="cleanDeals",
        business_label="Clean Deals",
        source="Project",
        formula="COUNT(*) WHERE is_clean_deal = TRUE",
        fields=("is_clean_deal",),
        filters="is_clean_deal = TRUE",
        unit="count",
        power_bi_label="Clean Deals",
        status=MetricStatus.DRAFT,
        notes=(
            "is_clean_deal is set during sync from the 'Clean Deal' column on the Job List CSV. "
            "Confirm with client what makes a deal 'clean' vs 'not clean'."
        ),
    ),
    "cleanDealPct": _m(
        key="cleanDealPct",
        business_label="Clean Deal %",
        source="Project",
        formula="100 * cleanDeals / totalProjects",
        unit="percent",
        power_bi_label="Clean %",
        status=MetricStatus.DRAFT,
    ),
    "cancellationRate": _m(
        key="cancellationRate",
        business_label="Cancellation Rate",
        source="Project",
        formula="100 * cancelledProjects / totalProjects",
        unit="percent",
        power_bi_label="Cancel %",
        status=MetricStatus.CONFIRMED,
    ),
    "netRetentionRate": _m(
        key="netRetentionRate",
        business_label="Net Retention Rate",
        source="Project",
        formula=(
            "100 * (totalProjects - cancelled - onHold - redFlagged) / totalProjects"
        ),
        unit="percent",
        power_bi_label="Retention Rate",
        status=MetricStatus.DRAFT,
        notes="Power BI definition may differ — confirm whether On Hold counts as retained.",
    ),
    "activePipelineValue": _m(
        key="activePipelineValue",
        business_label="Active Pipeline $",
        source="Project",
        formula="SUM(contract_amount) WHERE project_category = 'Active'",
        fields=("contract_amount",),
        filters="project_category = 'Active'",
        unit="currency",
        status=MetricStatus.CONFIRMED,
    ),
    "totalContractValue": _m(
        key="totalContractValue",
        business_label="Total Contract $",
        source="Project",
        formula="SUM(contract_amount)",
        fields=("contract_amount",),
        unit="currency",
        status=MetricStatus.CONFIRMED,
    ),
    # ────────────────────────── Pipeline / velocity ──────────────────────────
    "avgDaysToInstall": _m(
        key="avgDaysToInstall",
        business_label="Sign → Install (days)",
        source="Project",
        formula="AVG(install_date - customer_since) WHERE both dates present",
        fields=("customer_since", "install_date"),
        filters="install_date IS NOT NULL AND customer_since IS NOT NULL",
        unit="days",
        power_bi_label="Installations Days",
        status=MetricStatus.CONFIRMED,
    ),
    "avgDaysToCrc": _m(
        key="avgDaysToCrc",
        business_label="Sign → CRC (days)",
        source="Project",
        formula="AVG(crc_date - customer_since)",
        fields=("customer_since", "crc_date"),
        filters="crc_date IS NOT NULL AND customer_since IS NOT NULL",
        unit="days",
        power_bi_label="CRC Days",
        status=MetricStatus.DRAFT,
        notes="Power BI may average per-month then per-year (avg-of-avg). Verify methodology.",
    ),
    "avgDaysSsToCrc": _m(
        key="avgDaysSsToCrc",
        business_label="Site Survey → CRC (days)",
        source="Project",
        formula="AVG(crc_date - site_survey_scheduled)",
        fields=("site_survey_scheduled", "crc_date"),
        unit="days",
        status=MetricStatus.DRAFT,
    ),
    "avgDaysCrcToInstall": _m(
        key="avgDaysCrcToInstall",
        business_label="CRC → Install (days)",
        source="Project",
        formula="AVG(install_completed_or_install_date - crc_date)",
        fields=("crc_date", "install_completed", "install_date"),
        unit="days",
        status=MetricStatus.DRAFT,
    ),
    "avgDaysToPermit": _m(
        key="avgDaysToPermit",
        business_label="Sign → Permit (days)",
        source="Project",
        formula="AVG(permit_approved - customer_since)",
        fields=("customer_since", "permit_approved"),
        unit="days",
        status=MetricStatus.DRAFT,
    ),
    "avgDaysInstallToPto": _m(
        key="avgDaysInstallToPto",
        business_label="Install → PTO (days)",
        source="Project",
        formula="AVG(pto_submitted - install_completed_or_install_date)",
        fields=("install_completed", "install_date", "pto_submitted"),
        unit="days",
        status=MetricStatus.DRAFT,
    ),
    "avgInstallClean": _m(
        key="avgInstallClean",
        business_label="Sign → Install (Clean Deals)",
        source="Project",
        formula="AVG(install_date - customer_since) WHERE is_clean_deal = TRUE",
        fields=("customer_since", "install_date", "is_clean_deal"),
        filters="is_clean_deal = TRUE",
        unit="days",
        status=MetricStatus.DRAFT,
    ),
    "avgInstallNotClean": _m(
        key="avgInstallNotClean",
        business_label="Sign → Install (Not Clean)",
        source="Project",
        formula="AVG(install_date - customer_since) WHERE is_clean_deal = FALSE",
        fields=("customer_since", "install_date", "is_clean_deal"),
        filters="is_clean_deal = FALSE",
        unit="days",
        status=MetricStatus.DRAFT,
    ),
    # ─────────── Pipeline V2 metrics (Phase 1) ───────────
    "avgDaysSsToSsr": _m(
        key="avgDaysSsToSsr",
        business_label="Site Survey → SSR (days)",
        source="Project",
        formula="AVG(site_survey_results - site_survey_scheduled)",
        fields=("site_survey_scheduled", "site_survey_results"),
        unit="days",
        power_bi_label="SS - SSR Days",
        status=MetricStatus.DRAFT,
        notes="Sourced from Job List 'Site Survey Results' column.",
    ),
    "avgDaysSsrToCrc": _m(
        key="avgDaysSsrToCrc",
        business_label="SSR → CRC (days)",
        source="Project",
        formula="AVG(crc_date - site_survey_results)",
        fields=("site_survey_results", "crc_date"),
        unit="days",
        power_bi_label="SSR - CRC",
        status=MetricStatus.DRAFT,
    ),
    "ssrRate": _m(
        key="ssrRate",
        business_label="Site Survey Result Rate",
        source="Project",
        formula="100 * COUNT(site_survey_results IS NOT NULL) / COUNT(site_survey_scheduled IS NOT NULL)",
        fields=("site_survey_scheduled", "site_survey_results"),
        unit="percent",
        power_bi_label="Site Survey Result Rate",
        status=MetricStatus.DRAFT,
    ),
    "quickInstalls": _m(
        key="quickInstalls",
        business_label="Quick Installs",
        source="Project",
        formula="COUNT(*) WHERE is_quick_install = TRUE",
        fields=("is_quick_install",),
        filters="is_quick_install = TRUE",
        unit="count",
        power_bi_label="Quick Installs",
        status=MetricStatus.DRAFT,
        notes=(
            "Default rule: install_completed (or install_date when missing) "
            "minus customer_since <= 30 days. Threshold lives in "
            "sunbase_sync_service.QUICK_INSTALL_DAYS — change it there if "
            "the client confirms a different cutoff."
        ),
    ),
    "avgDaysToPtoApproved": _m(
        key="avgDaysToPtoApproved",
        business_label="Sign → PTO Approved (days)",
        source="Project",
        formula="AVG(pto_approved - customer_since)",
        fields=("customer_since", "pto_approved"),
        unit="days",
        status=MetricStatus.DRAFT,
    ),
    "avgDaysInstallToInspection": _m(
        key="avgDaysInstallToInspection",
        business_label="Install → Inspection Passed (days)",
        source="Project",
        formula="AVG(inspection_passed - install_completed)",
        fields=("install_completed", "inspection_passed"),
        unit="days",
        status=MetricStatus.DRAFT,
    ),
    "avgDaysInspectionToPto": _m(
        key="avgDaysInspectionToPto",
        business_label="Inspection → PTO Submitted (days)",
        source="Project",
        formula="AVG(pto_submitted - inspection_passed)",
        fields=("inspection_passed", "pto_submitted"),
        unit="days",
        status=MetricStatus.DRAFT,
    ),
    "stageBucketCount": _m(
        key="stageBucketCount",
        business_label="Projects per stage",
        source="Project",
        formula="COUNT(*) GROUP BY stage_bucket",
        fields=("stage_bucket",),
        unit="count",
        status=MetricStatus.DRAFT,
        notes=(
            "stage_bucket is heuristically derived from job_status during sync; "
            "see _classify_stage_bucket() in sunbase_sync_service.py for the mapping."
        ),
    ),
    "byProjectManager": _m(
        key="byProjectManager",
        business_label="Projects by Project Manager",
        source="Project",
        formula="GROUP BY project_manager, aggregate counts and milestone averages",
        fields=("project_manager",),
        unit="count",
        power_bi_label="PM's Dashboard",
        status=MetricStatus.DRAFT,
    ),
    "cancelledWithDate": _m(
        key="cancelledWithDate",
        business_label="Cancelled (with date)",
        source="Project",
        formula="COUNT(*) WHERE cancelled_date IS NOT NULL",
        fields=("cancelled_date",),
        filters="cancelled_date IS NOT NULL",
        unit="count",
        status=MetricStatus.DRAFT,
        notes="Sourced from Retention Cancellation KPIs report; back-filled onto Project rows.",
    ),
    # ──────────────────── Customer experience (CX) ────────────────────
    "reviewCaptureRate": _m(
        key="reviewCaptureRate",
        business_label="Review Capture Rate",
        source="CxProject",
        formula="100 * COUNT(has_review = TRUE) / COUNT(*)",
        fields=("has_review",),
        unit="percent",
        status=MetricStatus.CONFIRMED,
    ),
    "avgInstallToInspection": _m(
        key="avgInstallToInspection",
        business_label="Install → Inspection (days)",
        source="CxProject",
        formula="AVG(days_install_to_inspection_passed)",
        fields=("days_install_to_inspection_passed",),
        unit="days",
        status=MetricStatus.CONFIRMED,
    ),
    "avgInstallToPtoSubmitted": _m(
        key="avgInstallToPtoSubmitted",
        business_label="Install → PTO Submitted (days)",
        source="CxProject",
        formula="AVG(days_install_to_pto_submitted)",
        fields=("days_install_to_pto_submitted",),
        unit="days",
        status=MetricStatus.CONFIRMED,
    ),
    "avgInstallToReview": _m(
        key="avgInstallToReview",
        business_label="Install → Review (days)",
        source="CxProject",
        formula="AVG(days_install_to_review)",
        fields=("days_install_to_review",),
        unit="days",
        status=MetricStatus.CONFIRMED,
    ),
    # ─────────────── Manager / appointment-level metrics ───────────────
    "totalAppointments": _m(
        key="totalAppointments",
        business_label="Total Appointments",
        source="Appointment",
        formula="COUNT(Appointment)",
        unit="count",
        status=MetricStatus.CONFIRMED,
    ),
    "sitDownRate": _m(
        key="sitDownRate",
        business_label="Sit-Down Rate",
        source="Appointment",
        formula="100 * sitDowns / totalAppointments",
        unit="percent",
        status=MetricStatus.CONFIRMED,
    ),
    "closingRate": _m(
        key="closingRate",
        business_label="Closing Rate",
        source="Appointment",
        formula="100 * closedDeals / sitDowns",
        unit="percent",
        status=MetricStatus.CONFIRMED,
    ),
    "qualifiedClosingRate": _m(
        key="qualifiedClosingRate",
        business_label="Qualified Closing Rate",
        source="Appointment",
        formula="100 * activeClosedDeals / closedDeals",
        unit="percent",
        status=MetricStatus.CONFIRMED,
    ),
    "contactRate": _m(
        key="contactRate",
        business_label="Contact Rate",
        source="Door",
        formula="100 * contacts / totalDoors",
        unit="percent",
        status=MetricStatus.CONFIRMED,
    ),
}


def get_metric(key: str) -> MetricDefinition | None:
    return METRIC_REGISTRY.get(key)


def list_metrics() -> list[MetricDefinition]:
    return list(METRIC_REGISTRY.values())


def metrics_by_status(status: MetricStatus) -> Iterable[MetricDefinition]:
    return (m for m in METRIC_REGISTRY.values() if m.status == status)
