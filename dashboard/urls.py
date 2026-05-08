from django.urls import path
from rest_framework.routers import DefaultRouter

from dashboard.views.analytics_views import (
    CleanDealsView,
    CustomerExperienceView,
    DataSyncView,
    InsightsChatView,
    InsightsConversationListView,
    InsightsConversationMessagesView,
    InsightsGenerateView,
    ManagerPerformanceView,
    PerformanceView,
    PipelineView,
    ProjectsCancelledView,
    ProjectsOnHoldView,
    RetentionView,
    RolePerformanceView,
)
from dashboard.views.auth_views import GoogleLoginView, LoginView, RefreshTokenView
from dashboard.views.dashboard_views import (
    CancellationReasonsView,
    CategoryBreakdownView,
    FilterOptionsView,
    MetricsRegistryView,
    OnHoldReasonsView,
    OverviewView,
)
from dashboard.views.project_viewset import ProjectViewSet
from dashboard.views.user_admin_views import DashboardUserDetailView, DashboardUserListView

router = DefaultRouter()
router.register("projects", ProjectViewSet, basename="projects")

urlpatterns = [
    path("auth/login/", LoginView.as_view(), name="auth-login"),
    path("auth/google/", GoogleLoginView.as_view(), name="auth-google"),
    path("auth/refresh/", RefreshTokenView.as_view(), name="auth-refresh"),
    path("dashboard/overview/", OverviewView.as_view(), name="dashboard-overview"),
    path("dashboard/category-breakdown/", CategoryBreakdownView.as_view(), name="dashboard-category-breakdown"),
    path(
        "dashboard/cancellation-reasons/",
        CancellationReasonsView.as_view(),
        name="dashboard-cancellation-reasons",
    ),
    path("dashboard/on-hold-reasons/", OnHoldReasonsView.as_view(), name="dashboard-on-hold-reasons"),
    path("dashboard/filter-options/", FilterOptionsView.as_view(), name="dashboard-filter-options"),
    path("dashboard/metrics-registry/", MetricsRegistryView.as_view(), name="dashboard-metrics-registry"),
    path("clean-deals/", CleanDealsView.as_view(), name="clean-deals"),
    path("retention/", RetentionView.as_view(), name="retention"),
    path("performance/", PerformanceView.as_view(), name="performance"),
    path("pipeline/", PipelineView.as_view(), name="pipeline"),
    path("projects/on-hold/", ProjectsOnHoldView.as_view(), name="projects-on-hold"),
    path("projects/cancelled/", ProjectsCancelledView.as_view(), name="projects-cancelled"),
    path("cx/", CustomerExperienceView.as_view(), name="cx"),
    path("manager/", ManagerPerformanceView.as_view(), name="manager"),
    path("role-performance/", RolePerformanceView.as_view(), name="role-performance"),
    path("insights/generate/", InsightsGenerateView.as_view(), name="insights-generate"),
    path("insights/chat/", InsightsChatView.as_view(), name="insights-chat"),
    path("insights/conversations/", InsightsConversationListView.as_view(), name="insights-conversations"),
    path(
        "insights/conversations/<int:conversation_id>/messages/",
        InsightsConversationMessagesView.as_view(),
        name="insights-conversation-messages",
    ),
    path("sync/", DataSyncView.as_view(), name="sync"),
    path("users/", DashboardUserListView.as_view(), name="dashboard-users"),
    path("users/<str:pk>/", DashboardUserDetailView.as_view(), name="dashboard-user-detail"),
]

urlpatterns += router.urls
