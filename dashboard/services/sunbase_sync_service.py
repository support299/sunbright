import os
import re
import time
from datetime import date, datetime
from decimal import Decimal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from django.utils import timezone

from dashboard.models import Appointment, CxProject, Door, Project, SunbaseUser, SyncRun

BASE_URL = "https://server2.sunbasedata.com/sunbase/portal/api/dao"
JOB_LIST_KEY = "6ee21c0ccc5f4562bc5f29fa94eb6900"
CX_EXPERIENCE_KEY = "280ffb72dbae4ff0b15ac53d6027692c"
DOORS_LIST_KEY = "17d317dccb734d17931b8e55042643d1"
APPOINTMENT_STATUS_KEY = "51944068795144e38b0757bdc9123901"
USERS_REPORT_KEY = "9b2cae57f9e54e60b4e5b62b67e7c5d2"

QUICK_INSTALL_DAYS = 30


def _classify_stage_bucket(job_status):
    """
    Map raw Sunbase job_status text to a coarse pipeline bucket so the funnel
    visualisation can group hundreds of statuses into a handful of stages.
    """
    if not job_status:
        return ""
    s = job_status.lower()
    if s.startswith("cancel"):
        return "Cancelled"
    if s.startswith("on hold"):
        return "On Hold"
    if "complete" in s or "pto approved" in s:
        return "Completed"
    if "pto" in s:
        return "PTO"
    if "inspection" in s:
        return "Inspection"
    if "install" in s and "ready" in s:
        return "Ready for Install"
    if "install" in s:
        return "Install"
    if "permit" in s:
        return "Permitting"
    if "engineering" in s or "cio" in s:
        return "Engineering"
    if "site survey" in s or "ssr" in s:
        return "Site Survey"
    if "sold" in s or "crc" in s:
        return "Sold Projects"
    return "Sold Projects"


def _compute_quick_install(customer_since, install_completed, install_date):
    end = install_completed or install_date
    if not customer_since or not end:
        return False
    delta = (end - customer_since).days
    return 0 <= delta <= QUICK_INSTALL_DAYS

NON_CONTACT_STATUSES = {"NH1", "NH2", "NM", "SH"}
APPOINTMENT_STAGES = {
    "Appointment Cancelled", "Appointment Do not qualify", "Appointment No Show", "Appointment Rescheduled",
    "Cancelled at Door", "Closer Checked In", "Closer Missed Appointments", "Scheduling Conflict", "Set",
    "Set/Bill Uploaded", "Set/Confirmed", "Unconfirmed", "Unconfirmed Appointment", "set", "Reset",
    "in house - called in",
}
SHOW_STAGES = {"Credit Fail", "One Legger", "Sit Down - Go Back", "DNQ"}
QUALIFIED_SHOW_STAGES = {
    "Credit Passed - Not Closed", "Not Closed - Price", "Not Closed - Went with Competitor",
    "Proposal Issued - Follow Up", "Proposal Issued - Not Closed", "Sit - Not Solar Interested",
    "Qualified Show - Go Back", "Not Interested", "Price", "Proposal Issued", "WWAC", "Past Customer",
}
CLOSED_STAGES = {"Deal - Pending Review", "Sold", "Sold (CRC)", "Sold/ Installed", "Deal Cancelled", "Canceled", "Closed - Not Complete"}
PENDING_STAGES = {"Set", "Set/Bill Uploaded", "Set/Confirmed", "Unconfirmed", "Unconfirmed Appointment", "set", "None"}


def _clean(value):
    if value is None:
        return None
    v = str(value).strip()
    if v in {"", "nan", "NaN", "None"}:
        return None
    return v


def _to_int(value):
    v = _clean(value)
    if not v:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_decimal(value):
    v = _clean(value)
    if not v:
        return None
    try:
        return Decimal(v.replace(",", "").replace("$", ""))
    except Exception:
        return None


def _to_date(value):
    v = _clean(value)
    if not v:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}", v):
        try:
            return datetime.strptime(v[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    parts = v.split(" ")[0].split("/")
    if len(parts) == 3:
        try:
            return date(int(parts[2]), int(parts[0]), int(parts[1]))
        except ValueError:
            return None
    return None


def _to_datetime(value):
    v = _clean(value)
    if not v:
        return None
    parsed = None
    for fmt in ("%m/%d/%Y %I:%M %p", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(v, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    if timezone.is_naive(parsed):
        return timezone.make_aware(parsed, timezone.get_current_timezone())
    return parsed


def _days_between(d1, d2):
    if not d1 or not d2:
        return None
    return (d2 - d1).days


def _classify_project(status):
    if not status:
        return "Unknown"
    s = status.lower()
    if s.startswith("cancel"):
        return "Cancelled"
    if s.startswith("on hold"):
        return "On Hold"
    if s.startswith("red flag"):
        return "Red Flagged"
    if s.startswith("disqualif"):
        return "Disqualified"
    return "Active"


def _first_reason_match(status, patterns):
    if not status:
        return ""
    s = status.strip()
    for pattern in patterns:
        m = re.match(pattern, s, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _classify_deal_stage(stage, appt_date):
    if not stage:
        return "other"
    if stage in APPOINTMENT_STAGES:
        return "appointment"
    if stage in SHOW_STAGES:
        return "show"
    if stage in QUALIFIED_SHOW_STAGES:
        return "qualified_show"
    if stage in CLOSED_STAGES:
        return "closed"
    if stage in PENDING_STAGES and appt_date and appt_date < datetime.now():
        return "pending"
    return "other"


def _fetch_report(report_key):
    schema = os.getenv("SUNBASE_SCHEMA")
    api_key = os.getenv("SUNBASE_API_KEY")
    if not schema or not api_key:
        raise ValueError("SUNBASE_SCHEMA and SUNBASE_API_KEY must be configured.")
    url = f"{BASE_URL}/report_cmd.jsp?schema={schema}&key={report_key}&apiKey={api_key}"
    try:
        with urlopen(url, timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            if "Unauthorized" in body:
                raise ValueError("Sunbase API unauthorized. Check credentials.")
            if "Report not found" in body:
                raise ValueError(f"Sunbase report not found for key: {report_key}")
            return body
    except HTTPError as exc:
        raise ValueError(f"Sunbase API HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise ValueError(f"Sunbase API connection failed: {exc.reason}") from exc


def _split_csv_line(line):
    fields = []
    field = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                field.append('"')
                i += 1
            else:
                in_quotes = not in_quotes
        elif ch == "," and not in_quotes:
            fields.append("".join(field))
            field = []
        else:
            field.append(ch)
        i += 1
    fields.append("".join(field))
    return fields


def _parse_csv(text):
    lines = []
    current = []
    in_quotes = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            if in_quotes and i + 1 < len(text) and text[i + 1] == '"':
                current.append('"')
                i += 1
            else:
                in_quotes = not in_quotes
                current.append(ch)
        elif (ch == "\n" or ch == "\r") and not in_quotes:
            line = "".join(current).strip()
            if line:
                lines.append(line)
            current = []
            if ch == "\r" and i + 1 < len(text) and text[i + 1] == "\n":
                i += 1
        else:
            current.append(ch)
        i += 1

    tail = "".join(current).strip()
    if tail:
        lines.append(tail)

    if not lines:
        return []

    rows = [_split_csv_line(line) for line in lines]
    headers = rows[0]
    result = []
    for row in rows[1:]:
        mapped = {}
        for idx, header in enumerate(headers):
            mapped[header] = row[idx] if idx < len(row) else None
        result.append(mapped)
    return result


def _csv_rows(text):
    return _parse_csv(text)


def _get_csv_cell(row, *candidates):
    """First non-empty cell matching any candidate header (exact, then case-insensitive strip)."""
    if not row:
        return None
    for c in candidates:
        if not c:
            continue
        v = row.get(c)
        if v is not None and str(v).strip() not in ("", "nan", "NaN", "None"):
            return v
    lower_map = {(k or "").strip().lower(): v for k, v in row.items()}
    for c in candidates:
        if not c:
            continue
        v = lower_map.get(c.strip().lower())
        if v is not None and str(v).strip() not in ("", "nan", "NaN", "None"):
            return v
    return None


def _sync_job_list():
    rows = _csv_rows(_fetch_report(JOB_LIST_KEY))
    inserted = 0
    errors = []
    Project.objects.all().delete()
    for row in rows:
        try:
            job_status = _clean(row.get("Job Status"))
            if not job_status:
                continue
            customer_since = _to_date(
                _get_csv_cell(row, "Customer Since", "customer since", "Customer Sign Date", "Sign Date")
            )
            install_date = _to_date(_get_csv_cell(row, "Install Date", "install date", "Scheduled Install"))
            site_survey_scheduled = _to_date(
                _get_csv_cell(
                    row,
                    "Site Survey Scheduled",
                    "site survey scheduled",
                    "Site Survey",
                    "Survey Scheduled",
                )
            )
            crc_date = _to_date(
                _get_csv_cell(
                    row,
                    "CRC",
                    "crc",
                    "CRC Date",
                    "crc date",
                    "CRC Complete",
                    "Customer Responsibility Complete",
                    "Customer Responsibility Certificate",
                    "NTP",
                    "NTP Approved",
                    "NTP Date",
                    "Notice to Proceed",
                )
            )
            permit_approved = _to_date(
                _get_csv_cell(row, "Permit Approved", "permit approved", "Permit", "PTO Permit Approved")
            )
            install_completed = _to_date(
                _get_csv_cell(
                    row,
                    "Install Completed",
                    "install completed",
                    "Install Complete",
                    "Installation Completed",
                    "Installed",
                )
            )
            pto_submitted = _to_date(
                _get_csv_cell(
                    row,
                    "PTO Submitted",
                    "pto submitted",
                    "PTO Submit",
                    "PTO Application Submitted",
                )
            )
            clean_deal_date = _to_date(row.get("Clean Deal"))
            category = _classify_project(job_status)
            site_survey_results = _to_date(
                _get_csv_cell(row, "Site Survey Results", "site survey results", "SSR")
            )
            site_survey_submitted = _to_date(
                _get_csv_cell(row, "Site Survey Submitted", "site survey submitted")
            )
            site_survey_approved = _to_date(
                _get_csv_cell(row, "Site Survey Approved", "site survey approved")
            )
            cio_date = _to_date(_get_csv_cell(row, "CIO", "CIO Date", "cio"))
            engineering_date = _to_date(
                _get_csv_cell(row, "Engineering", "Engineering Date", "engineering")
            )
            permit_submitted = _to_date(
                _get_csv_cell(row, "Permit Submitted", "permit submitted")
            )
            insurance_approved = _to_date(
                _get_csv_cell(row, "Insurance Approved", "insurance approved")
            )
            inspection_scheduled = _to_date(
                _get_csv_cell(row, "Inspection Scheduled", "inspection scheduled")
            )
            inspection_passed = _to_date(
                _get_csv_cell(row, "Inspection Passed", "inspection passed")
            )
            ready_for_pto = _to_date(
                _get_csv_cell(row, "Ready for PTO", "ready for pto")
            )
            pto_approved = _to_date(_get_csv_cell(row, "PTO Approved", "pto approved"))
            cancelled_date = _to_date(
                _get_csv_cell(row, "Cancelled Date", "Cancellation Date", "cancelled date")
            )
            project_manager = _clean(
                _get_csv_cell(row, "Project Manager", "project manager", "PM")
            ) or ""
            setter = _clean(_get_csv_cell(row, "Setter", "setter")) or ""
            sunbase_job_uuid = _clean(_get_csv_cell(row, "uuid", "Job UUID", "JobId")) or ""
            stage_bucket = _classify_stage_bucket(job_status)
            is_quick_install = _compute_quick_install(
                customer_since, install_completed, install_date
            )
            Project.objects.create(
                first_name=_clean(row.get("First Name")) or "",
                last_name=_clean(row.get("Last Name")) or "",
                sales_rep=_clean(row.get("Sales Rep")) or "",
                sales_team=_clean(row.get("sales_team")) or "",
                installer=_clean(row.get("Installer")) or "",
                lead_source=_clean(row.get("Lead Source")) or "",
                setter=setter,
                project_manager=project_manager,
                job_status=job_status,
                project_category=category,
                stage_bucket=stage_bucket,
                contract_amount=_to_decimal(row.get("Contract Amt")),
                customer_since=customer_since,
                install_date=install_date,
                site_survey_scheduled=site_survey_scheduled,
                site_survey_submitted=site_survey_submitted,
                site_survey_approved=site_survey_approved,
                site_survey_results=site_survey_results,
                cio_date=cio_date,
                engineering_date=engineering_date,
                crc_date=crc_date,
                permit_submitted=permit_submitted,
                permit_approved=permit_approved,
                insurance_approved=insurance_approved,
                install_completed=install_completed,
                inspection_scheduled=inspection_scheduled,
                inspection_passed=inspection_passed,
                ready_for_pto=ready_for_pto,
                pto_submitted=pto_submitted,
                pto_approved=pto_approved,
                cancelled_date=cancelled_date,
                cancellation_reason=_first_reason_match(
                    job_status,
                    (
                        r"^Cancel(?:l)?ed\s*-\s*(.+)$",
                        r"^Cancel(?:l)?ed\s*:\s*(.+)$",
                        r"^Deal\s+Cancel(?:l)?ed\s*-\s*(.+)$",
                        r"^Deal\s+Canceled\s*-\s*(.+)$",
                    ),
                ),
                on_hold_reason=_first_reason_match(
                    job_status,
                    (
                        r"^On\s+[Hh]old\s*-\s*(.+)$",
                        r"^On\s+[Hh]old\s*:\s*(.+)$",
                    ),
                ),
                is_clean_deal=bool(clean_deal_date),
                is_quick_install=is_quick_install,
                is_active=category == "Active",
                sunbase_job_uuid=sunbase_job_uuid,
            )
            inserted += 1
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(str(exc))
    return {"inserted": inserted, "errors": errors}


def _sync_cx_experience():
    rows = _csv_rows(_fetch_report(CX_EXPERIENCE_KEY))
    inserted = 0
    errors = []
    CxProject.objects.all().delete()
    for row in rows:
        try:
            install_date = _to_date(row.get("Install Date"))
            inspection_passed = _to_date(row.get("Inspection Passed"))
            pto_submitted = _to_date(row.get("PTO Submitted"))
            pto_approved = _to_date(row.get("PTO Approved"))
            review_captured_date = _to_date(row.get("Review Captured Date"))
            sunbase_job_uuid = _clean(_get_csv_cell(row, "uuid", "Job UUID", "JobId")) or ""
            CxProject.objects.create(
                row_number=_to_int(row.get("Row")),
                sunbase_job_uuid=sunbase_job_uuid,
                first_name=_clean(row.get("First Name")) or "",
                last_name=_clean(row.get("Last Name")) or "",
                job_status=_clean(row.get("Job Status")) or "",
                installer=_clean(row.get("Installer")) or "",
                install_date=install_date,
                install_completed=_to_date(row.get("Install Completed")),
                inspection_scheduled=_to_date(row.get("Inspection Scheduled")),
                inspection_passed=inspection_passed,
                pto_submitted=pto_submitted,
                pto_approved=pto_approved,
                review_captured_date=review_captured_date,
                testimonial_potential=(_clean(row.get("Testimonial Potential")) or "").lower() == "on",
                testimonial_done=(_clean(row.get("Testimonial Done")) or "").lower() == "on",
                model_home_program=(_clean(row.get("Model Home Program")) or "").lower() == "on",
                has_review=bool(review_captured_date),
                days_install_to_inspection_passed=_days_between(install_date, inspection_passed),
                days_install_to_pto_submitted=_days_between(install_date, pto_submitted),
                days_install_to_pto_approved=_days_between(install_date, pto_approved),
                days_install_to_review=_days_between(install_date, review_captured_date),
            )
            inserted += 1
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(str(exc))
    return {"inserted": inserted, "errors": errors}


def _sync_doors():
    rows = _csv_rows(_fetch_report(DOORS_LIST_KEY))
    inserted = 0
    skipped = 0
    errors = []
    cutoff = timezone.make_aware(datetime(datetime.now().year - 1, 7, 1), timezone.get_current_timezone())
    Door.objects.all().delete()
    for row in rows:
        try:
            canvasser = _clean(row.get("Canvasser"))
            status = _clean(row.get("Status"))
            create_time = _to_datetime(row.get("Create Time"))
            if not canvasser or canvasser == "000000000000" or not status:
                skipped += 1
                continue
            if create_time and create_time < cutoff:
                skipped += 1
                continue
            Door.objects.create(
                row_number=_to_int(row.get("Row")),
                first_name=_clean(row.get("First Name")) or "",
                last_name=_clean(row.get("Last Name")) or "",
                address=_clean(row.get("Address")) or "",
                city=_clean(row.get("City")) or "",
                state=_clean(row.get("State")) or "",
                canvasser=canvasser,
                status=status,
                create_time=create_time,
                appt_time=_to_datetime(row.get("Appt Time")),
                is_contact=status not in NON_CONTACT_STATUSES,
            )
            inserted += 1
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(str(exc))
    return {"inserted": inserted, "skipped": skipped, "errors": errors}


def _sync_appointments():
    rows = _csv_rows(_fetch_report(APPOINTMENT_STATUS_KEY))
    inserted = 0
    skipped = 0
    errors = []
    cutoff = timezone.make_aware(datetime(datetime.now().year - 1, 7, 1), timezone.get_current_timezone())
    Appointment.objects.all().delete()
    for row in rows:
        try:
            appt_dt = _to_datetime(row.get("Appointment Date/Time"))
            if appt_dt and appt_dt < cutoff:
                skipped += 1
                continue
            sales_rep = _clean(row.get("Sales Rep")) or ""
            setter = _clean(row.get("Setter")) or ""
            stage = _clean(row.get("Deal Stage")) or ""
            Appointment.objects.create(
                row_number=_to_int(row.get("Row")),
                first_name=_clean(row.get("First Name")) or "",
                last_name=_clean(row.get("Last Name")) or "",
                address=_clean(row.get("Address")) or "",
                city=_clean(row.get("City")) or "",
                state=_clean(row.get("State")) or "",
                zip_code=_clean(row.get("Zip Code")) or "",
                phone=_clean(row.get("Phone")) or "",
                appointment_datetime=appt_dt,
                lead_source=_clean(row.get("Lead Source")) or "",
                lead_date=_to_date(row.get("Lead Date")),
                language=_clean(row.get("Language")) or "",
                salvage_notes=_clean(row.get("Salvage Notes")) or "",
                deal_stage=stage,
                sales_rep=sales_rep,
                setter=setter,
                sales_team=_clean(row.get("sales_team")) or "",
                is_blitz_deal=(_clean(row.get("Blitz Deal")) or "").lower() == "on",
                is_self_set=bool(sales_rep and setter and sales_rep.lower() == setter.lower()),
                stage_category=_classify_deal_stage(stage, appt_dt),
            )
            inserted += 1
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(str(exc))
    return {"inserted": inserted, "skipped": skipped, "errors": errors}


def _sync_users():
    """
    Pull the General → Users report into SunbaseUser. Skip silently if the
    report key isn't configured or the response is empty so the rest of the
    sync isn't blocked.
    """
    inserted = 0
    errors = []
    try:
        rows = _csv_rows(_fetch_report(USERS_REPORT_KEY))
    except Exception as exc:  # pylint: disable=broad-except
        return {"inserted": 0, "errors": [f"users report fetch failed: {exc}"]}
    if not rows:
        return {"inserted": 0, "errors": []}
    SunbaseUser.objects.all().delete()
    for row in rows:
        try:
            uuid = _clean(_get_csv_cell(row, "uuid", "User UUID", "User Id")) or ""
            full_name = _clean(_get_csv_cell(row, "Fullname", "Full Name", "Name")) or ""
            if not uuid:
                continue
            SunbaseUser.objects.create(
                external_uuid=uuid,
                full_name=full_name,
                role=_clean(_get_csv_cell(row, "Role", "User Role")) or "",
                manager_name=_clean(_get_csv_cell(row, "Manager", "Manager Name")) or "",
                crew_name=_clean(_get_csv_cell(row, "Crew", "Team", "Crew Name")) or "",
                login_allowed=(_clean(row.get("Login Allowed")) or "").lower() == "yes",
            )
            inserted += 1
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(str(exc))
    return {"inserted": inserted, "errors": errors}


def run_full_sync():
    start = time.time()
    timestamp = timezone.now().isoformat()
    result = {
        "success": False,
        "timestamp": timestamp,
        "jobList": {"inserted": 0, "errors": []},
        "cxExperience": {"inserted": 0, "errors": []},
        "doorsList": {"inserted": 0, "skipped": 0, "errors": []},
        "appointmentStatus": {"inserted": 0, "skipped": 0, "errors": []},
        "users": {"inserted": 0, "errors": []},
        "duration": 0,
        "error": "",
    }
    try:
        result["jobList"] = _sync_job_list()
        result["cxExperience"] = _sync_cx_experience()
        result["doorsList"] = _sync_doors()
        result["appointmentStatus"] = _sync_appointments()
        try:
            result["users"] = _sync_users()
        except Exception as exc:  # pylint: disable=broad-except
            result["users"] = {"inserted": 0, "errors": [str(exc)]}
        result["success"] = True
    except Exception as exc:  # pylint: disable=broad-except
        print("SYNC ERROR:", str(exc))
        result["error"] = str(exc)
        raise
    finally:
        result["duration"] = int((time.time() - start) * 1000)
        SyncRun.objects.create(
            success=result["success"],
            duration_ms=result["duration"],
            payload=result,
            error=result.get("error", ""),
        )
    return result


def get_last_sync_result():
    record = SyncRun.objects.first()
    if not record:
        return None
    return record.payload
