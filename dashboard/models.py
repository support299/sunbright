from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone


class TimeStampedSoftDeleteModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        abstract = True

    def soft_delete(self):
        self.deleted_at = timezone.now()
        self.save(update_fields=["deleted_at", "updated_at"])


class Installer(models.Model):
    name = models.CharField(max_length=255, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class LeadSource(models.Model):
    name = models.CharField(max_length=255, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Team(models.Model):
    name = models.CharField(max_length=255, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Rep(models.Model):
    name = models.CharField(max_length=255, unique=True)
    team = models.ForeignKey(
        Team, on_delete=models.SET_NULL, null=True, blank=True, related_name="reps"
    )

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Project(TimeStampedSoftDeleteModel):
    first_name = models.CharField(max_length=255, blank=True)
    last_name = models.CharField(max_length=255, blank=True)
    sales_rep = models.CharField(max_length=255, blank=True)
    sales_team = models.CharField(max_length=255, blank=True)
    installer = models.CharField(max_length=255, blank=True)
    lead_source = models.CharField(max_length=255, blank=True)
    setter = models.CharField(max_length=255, blank=True)
    project_manager = models.CharField(max_length=255, blank=True)
    job_status = models.CharField(max_length=255)
    project_category = models.CharField(max_length=64)
    stage_bucket = models.CharField(max_length=64, blank=True, db_index=True)
    contract_amount = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    customer_since = models.DateField(null=True, blank=True)
    install_date = models.DateField(null=True, blank=True)
    site_survey_scheduled = models.DateField(null=True, blank=True)
    site_survey_submitted = models.DateField(null=True, blank=True)
    site_survey_approved = models.DateField(null=True, blank=True)
    site_survey_results = models.DateField(null=True, blank=True)
    cio_date = models.DateField(null=True, blank=True)
    engineering_date = models.DateField(null=True, blank=True)
    crc_date = models.DateField(null=True, blank=True)
    permit_submitted = models.DateField(null=True, blank=True)
    permit_approved = models.DateField(null=True, blank=True)
    insurance_approved = models.DateField(null=True, blank=True)
    install_completed = models.DateField(null=True, blank=True)
    inspection_scheduled = models.DateField(null=True, blank=True)
    inspection_passed = models.DateField(null=True, blank=True)
    ready_for_pto = models.DateField(null=True, blank=True)
    pto_submitted = models.DateField(null=True, blank=True)
    pto_approved = models.DateField(null=True, blank=True)
    cancelled_date = models.DateField(null=True, blank=True)
    cancellation_reason = models.CharField(max_length=255, blank=True)
    on_hold_reason = models.CharField(max_length=255, blank=True)
    is_clean_deal = models.BooleanField(default=False)
    is_quick_install = models.BooleanField(default=False)
    is_active = models.BooleanField(default=False)
    sunbase_job_uuid = models.CharField(max_length=64, blank=True, db_index=True)
    installer_ref = models.ForeignKey(
        Installer,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="projects",
    )
    lead_source_ref = models.ForeignKey(
        LeadSource,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="projects",
    )
    rep = models.ForeignKey(
        Rep, on_delete=models.SET_NULL, null=True, blank=True, related_name="projects"
    )
    team = models.ForeignKey(
        Team, on_delete=models.SET_NULL, null=True, blank=True, related_name="projects"
    )

    class Meta:
        ordering = ["-id"]

    def __str__(self):
        return f"{self.first_name} {self.last_name}".strip() or f"Project {self.id}"


class CxProject(TimeStampedSoftDeleteModel):
    row_number = models.IntegerField(null=True, blank=True)
    sunbase_job_uuid = models.CharField(max_length=64, blank=True, db_index=True)
    first_name = models.CharField(max_length=255, blank=True)
    last_name = models.CharField(max_length=255, blank=True)
    job_status = models.CharField(max_length=255, blank=True)
    installer = models.CharField(max_length=255, blank=True)
    installer_ref = models.ForeignKey(
        Installer,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="cx_projects",
    )
    install_date = models.DateField(null=True, blank=True)
    install_completed = models.DateField(null=True, blank=True)
    inspection_scheduled = models.DateField(null=True, blank=True)
    inspection_passed = models.DateField(null=True, blank=True)
    pto_submitted = models.DateField(null=True, blank=True)
    pto_approved = models.DateField(null=True, blank=True)
    review_captured_date = models.DateField(null=True, blank=True)
    testimonial_potential = models.BooleanField(default=False)
    testimonial_done = models.BooleanField(default=False)
    model_home_program = models.BooleanField(default=False)
    has_review = models.BooleanField(default=False)
    days_install_to_inspection_passed = models.IntegerField(null=True, blank=True)
    days_install_to_pto_submitted = models.IntegerField(null=True, blank=True)
    days_install_to_pto_approved = models.IntegerField(null=True, blank=True)
    days_install_to_review = models.IntegerField(null=True, blank=True)
    days_inspection_scheduled_to_passed = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-id"]


class Door(TimeStampedSoftDeleteModel):
    row_number = models.IntegerField(null=True, blank=True)
    first_name = models.CharField(max_length=255, blank=True)
    last_name = models.CharField(max_length=255, blank=True)
    address = models.CharField(max_length=500, blank=True)
    city = models.CharField(max_length=255, blank=True)
    state = models.CharField(max_length=64, blank=True)
    canvasser = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=80, blank=True)
    create_time = models.DateTimeField(null=True, blank=True)
    appt_time = models.DateTimeField(null=True, blank=True)
    is_contact = models.BooleanField(default=False)

    class Meta:
        ordering = ["-id"]


class Appointment(TimeStampedSoftDeleteModel):
    row_number = models.IntegerField(null=True, blank=True)
    first_name = models.CharField(max_length=255, blank=True)
    last_name = models.CharField(max_length=255, blank=True)
    address = models.CharField(max_length=500, blank=True)
    city = models.CharField(max_length=255, blank=True)
    state = models.CharField(max_length=64, blank=True)
    zip_code = models.CharField(max_length=32, blank=True)
    phone = models.CharField(max_length=50, blank=True)
    appointment_datetime = models.DateTimeField(null=True, blank=True)
    lead_source = models.CharField(max_length=255, blank=True)
    lead_date = models.DateField(null=True, blank=True)
    language = models.CharField(max_length=50, blank=True)
    salvage_notes = models.TextField(blank=True)
    deal_stage = models.CharField(max_length=255, blank=True)
    sales_rep = models.CharField(max_length=255, blank=True)
    setter = models.CharField(max_length=255, blank=True)
    sales_team = models.CharField(max_length=255, blank=True)
    rep = models.ForeignKey(
        Rep, on_delete=models.SET_NULL, null=True, blank=True, related_name="appointments"
    )
    team = models.ForeignKey(
        Team, on_delete=models.SET_NULL, null=True, blank=True, related_name="appointments"
    )
    lead_source_ref = models.ForeignKey(
        LeadSource,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="appointments",
    )
    is_blitz_deal = models.BooleanField(default=False)
    is_self_set = models.BooleanField(default=False)
    stage_category = models.CharField(max_length=50, blank=True)

    class Meta:
        ordering = ["-id"]


class DashboardDataScope(models.Model):
    """
    Limits dashboard metrics for non-staff users to a sales team or rep.
    Staff (is_staff) always bypass scope and see the full organization.
    """

    class ScopeKind(models.TextChoices):
        TEAM = "team", "Single sales team"
        TEAMS = "teams", "Multiple sales teams"
        REP = "rep", "Single sales rep"

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="dashboard_scope",
    )
    scope_kind = models.CharField(
        max_length=16,
        choices=ScopeKind.choices,
        default=ScopeKind.REP,
    )
    sales_team = models.CharField(max_length=255, blank=True)
    sales_rep = models.CharField(max_length=255, blank=True)
    sales_teams = models.JSONField(default=list, blank=True)

    class Meta:
        verbose_name = "Dashboard data scope"
        verbose_name_plural = "Dashboard data scopes"

    def clean(self):
        k = self.scope_kind
        if k == self.ScopeKind.TEAM:
            if not (self.sales_team or "").strip():
                raise ValidationError({"sales_team": "Required for single-team scope."})
        elif k == self.ScopeKind.TEAMS:
            names = [n.strip() for n in (self.sales_teams or []) if isinstance(n, str) and n.strip()]
            if not names:
                raise ValidationError({"sales_teams": "Add at least one team name."})
        elif k == self.ScopeKind.REP:
            if not (self.sales_rep or "").strip():
                raise ValidationError({"sales_rep": "Required for rep scope."})

    def __str__(self):
        return f"Scope({self.user_id}, {self.scope_kind})"


class SunbaseUser(models.Model):
    external_uuid = models.CharField(max_length=64, unique=True)
    full_name = models.CharField(max_length=255, blank=True)
    role = models.CharField(max_length=255, blank=True)
    login_allowed = models.BooleanField(default=False)
    manager_name = models.CharField(max_length=255, blank=True)
    crew_name = models.CharField(max_length=255, blank=True)
    rep = models.ForeignKey(
        Rep,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="sunbase_users",
    )
    team = models.ForeignKey(
        Team,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="sunbase_users",
    )
    last_synced_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["full_name", "external_uuid"]

    def __str__(self):
        return self.full_name or self.external_uuid


class SyncRun(models.Model):
    success = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    duration_ms = models.IntegerField(default=0)
    payload = models.JSONField(default=dict)
    error = models.TextField(blank=True)

    class Meta:
        ordering = ["-timestamp"]


class InsightConversation(TimeStampedSoftDeleteModel):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        ARCHIVED = "archived", "Archived"
        CLOSED = "closed", "Closed"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="insight_conversations",
    )
    title = models.CharField(max_length=255, blank=True)
    date_from = models.DateField(null=True, blank=True)
    date_to = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.ACTIVE)
    scope_snapshot = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title or f"Conversation {self.id}"


class InsightMessage(TimeStampedSoftDeleteModel):
    class Role(models.TextChoices):
        SYSTEM = "system", "System"
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"
        TOOL = "tool", "Tool"

    class IntentLabel(models.TextChoices):
        DASHBOARD_QUERY = "DASHBOARD_QUERY", "Dashboard query"
        GREETING_SMALLTALK = "GREETING_SMALLTALK", "Greeting or smalltalk"
        OUT_OF_SCOPE = "OUT_OF_SCOPE", "Out of scope"

    conversation = models.ForeignKey(
        InsightConversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=16, choices=Role.choices)
    content = models.TextField()
    rendered_prompt = models.TextField(blank=True)
    context_meta = models.JSONField(default=dict, blank=True)
    intent_label = models.CharField(max_length=24, choices=IntentLabel.choices, blank=True)
    provider = models.CharField(max_length=64, blank=True)
    model = models.CharField(max_length=128, blank=True)
    latency_ms = models.IntegerField(default=0)
    token_input = models.IntegerField(default=0)
    token_output = models.IntegerField(default=0)
    finish_reason = models.CharField(max_length=64, blank=True)
    error_payload = models.JSONField(null=True, blank=True)
    safety_flags = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role} message #{self.id}"
