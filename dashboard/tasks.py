from celery import shared_task

from dashboard.services.sunbase_sync_service import run_full_sync


@shared_task(name="dashboard.run_sunbase_full_sync", max_retries=0)
def run_sunbase_full_sync_task():
    """Run the same Sunbase full sync used by POST /api/sync/ (scheduled via Celery Beat)."""
    return run_full_sync()
