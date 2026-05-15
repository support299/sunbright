from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from dashboard.permissions import IsDashboardAdmin
from dashboard.serializers.insight_chat_serializer import (
    InsightChatRequestSerializer,
    InsightConversationSerializer,
    InsightMessageSerializer,
)
from dashboard.serializers.project_serializer import ProjectSerializer
from dashboard.services.analytics_service import (
    get_clean_deals_bundle,
    get_cx_bundle,
    get_manager_bundle,
    get_performance_bundle,
    get_pipeline_bundle,
    get_retention_bundle,
    get_role_performance_bundle,
)
from dashboard.services.project_service import get_cancelled_projects, get_on_hold_projects
from dashboard.services.insights_service import (
    InsightsConversationError,
    InsightsLLMError,
    InsightsRateLimitError,
    chat_with_insights_assistant,
    generate_dashboard_insights,
    list_insight_conversations,
    list_insight_messages,
)
from dashboard.services.sunbase_sync_service import get_last_sync_result, run_full_sync
from dashboard.utils import error_response, parse_dashboard_filters_full, success_response


class CleanDealsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        f = parse_dashboard_filters_full(request)
        return Response(
            success_response(
                get_clean_deals_bundle(
                    f["date_from"],
                    f["date_to"],
                    request.user,
                    installer=f["installer"],
                    sales_team=f["sales_team"],
                    lead_source=f["lead_source"],
                    project_manager=f["manager"],
                    market=f["market"],
                    rep_kind=f["rep_kind"],
                    rep_name=f["rep_name"],
                )
            )
        )


class RetentionView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        f = parse_dashboard_filters_full(request)
        return Response(
            success_response(
                get_retention_bundle(
                    f["date_from"],
                    f["date_to"],
                    request.user,
                    installer=f["installer"],
                    sales_team=f["sales_team"],
                    lead_source=f["lead_source"],
                    project_manager=f["manager"],
                    market=f["market"],
                    rep_kind=f["rep_kind"],
                    rep_name=f["rep_name"],
                )
            )
        )


class PerformanceView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        f = parse_dashboard_filters_full(request)
        return Response(
            success_response(
                get_performance_bundle(
                    f["date_from"],
                    f["date_to"],
                    request.user,
                    installer=f["installer"],
                    sales_team=f["sales_team"],
                    lead_source=f["lead_source"],
                    project_manager=f["manager"],
                    market=f["market"],
                    rep_kind=f["rep_kind"],
                    rep_name=f["rep_name"],
                )
            )
        )


class PipelineView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        f = parse_dashboard_filters_full(request)
        return Response(
            success_response(
                get_pipeline_bundle(
                    f["date_from"],
                    f["date_to"],
                    request.user,
                    installer=f["installer"],
                    sales_team=f["sales_team"],
                    lead_source=f["lead_source"],
                    project_manager=f["manager"],
                    market=f["market"],
                    rep_kind=f["rep_kind"],
                    rep_name=f["rep_name"],
                )
            )
        )


class ProjectsOnHoldView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        f = parse_dashboard_filters_full(request)
        serializer = ProjectSerializer(
            get_on_hold_projects(
                f["date_from"],
                f["date_to"],
                request.user,
                installer=f["installer"],
                sales_team=f["sales_team"],
                lead_source=f["lead_source"],
                project_manager=f["manager"],
                market=f["market"],
                rep_kind=f["rep_kind"],
                rep_name=f["rep_name"],
            ),
            many=True,
        )
        return Response(success_response(serializer.data))


class ProjectsCancelledView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        f = parse_dashboard_filters_full(request)
        serializer = ProjectSerializer(
            get_cancelled_projects(
                f["date_from"],
                f["date_to"],
                request.user,
                installer=f["installer"],
                sales_team=f["sales_team"],
                lead_source=f["lead_source"],
                project_manager=f["manager"],
                market=f["market"],
                rep_kind=f["rep_kind"],
                rep_name=f["rep_name"],
            ),
            many=True,
        )
        return Response(success_response(serializer.data))


class CustomerExperienceView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        f = parse_dashboard_filters_full(request)
        return Response(
            success_response(
                get_cx_bundle(
                    f["date_from"],
                    f["date_to"],
                    request.user,
                    installer=f["installer"],
                    sales_team=f["sales_team"],
                    lead_source=f["lead_source"],
                    project_manager=f["manager"],
                    market=f["market"],
                    rep_kind=f["rep_kind"],
                    rep_name=f["rep_name"],
                )
            )
        )


class ManagerPerformanceView(APIView):
    """D2D / appointment metrics; scoped by user data scope (non-admins see their team/rep slice)."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        f = parse_dashboard_filters_full(request)
        return Response(
            success_response(
                get_manager_bundle(
                    f["date_from"],
                    f["date_to"],
                    request.user,
                    installer=f["installer"],
                    sales_team=f["sales_team"],
                    lead_source=f["lead_source"],
                    project_manager=f["manager"],
                    market=f["market"],
                    rep_kind=f["rep_kind"],
                    rep_name=f["rep_name"],
                )
            )
        )


class RolePerformanceView(APIView):
    """
    Setter / closer style tables keyed off Sunbase Users roles (setter vs Sales/closer).
    Uses doors, appointments, and projects matched by Fullname ↔ canvasser / setter / sales_rep.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        f = parse_dashboard_filters_full(request)
        return Response(
            success_response(
                get_role_performance_bundle(
                    f["date_from"],
                    f["date_to"],
                    request.user,
                    installer=f["installer"],
                    sales_team=f["sales_team"],
                    lead_source=f["lead_source"],
                    project_manager=f["manager"],
                    market=f["market"],
                    rep_kind=f["rep_kind"],
                    rep_name=f["rep_name"],
                )
            )
        )


class InsightsGenerateView(APIView):
    permission_classes = [IsAuthenticated, IsDashboardAdmin]

    def post(self, request):
        f = parse_dashboard_filters_full(request)
        try:
            payload = generate_dashboard_insights(
                f["date_from"],
                f["date_to"],
                request.user,
                installer=f["installer"],
                sales_team=f["sales_team"],
                lead_source=f["lead_source"],
                project_manager=f["manager"],
                market=f["market"],
                rep_kind=f["rep_kind"],
                rep_name=f["rep_name"],
            )
        except InsightsLLMError as exc:
            return Response(
                error_response([{"field": "llm", "message": str(exc)}]),
                status=502,
            )
        return Response(success_response(payload))


class DataSyncView(APIView):
    permission_classes = [IsAuthenticated, IsDashboardAdmin]

    def post(self, request):
        result = run_full_sync()
        return Response(success_response(result))

    def get(self, request):
        return Response(success_response({"lastResult": get_last_sync_result()}))


class InsightsChatView(APIView):
    permission_classes = [IsAuthenticated, IsDashboardAdmin]

    def post(self, request):
        serializer = InsightChatRequestSerializer(data=request.data or {})
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        try:
            data = chat_with_insights_assistant(
                user=request.user,
                message=payload.get("message", ""),
                conversation_id=payload.get("conversationId"),
                date_from=payload.get("dateFrom"),
                date_to=payload.get("dateTo"),
                sales_team=(payload.get("salesTeam") or "").strip() or None,
                installer=(payload.get("installer") or "").strip() or None,
                lead_source=(payload.get("leadSource") or "").strip() or None,
                project_manager=(payload.get("manager") or "").strip() or None,
                market=(payload.get("market") or "").strip() or None,
                rep_kind=payload.get("repKind"),
                rep_name=(payload.get("repName") or "").strip() or None,
            )
        except InsightsRateLimitError as exc:
            return Response(error_response([{"field": "rateLimit", "message": str(exc)}]), status=429)
        except InsightsConversationError:
            return Response(error_response([{"field": "conversation", "message": "Conversation not found."}]), status=404)
        except InsightsLLMError:
            return Response(
                error_response([{"field": "llm", "message": "I’m having trouble generating insights right now. Please try again."}]),
                status=502,
            )
        return Response(success_response(data))


class InsightsConversationListView(APIView):
    permission_classes = [IsAuthenticated, IsDashboardAdmin]

    def get(self, request):
        rows = list_insight_conversations(request.user)
        return Response(success_response(InsightConversationSerializer(rows, many=True).data))


class InsightsConversationMessagesView(APIView):
    permission_classes = [IsAuthenticated, IsDashboardAdmin]

    def get(self, request, conversation_id):
        try:
            rows = list_insight_messages(request.user, conversation_id)
        except InsightsConversationError:
            return Response(error_response([{"field": "conversation", "message": "Conversation not found."}]), status=404)
        return Response(success_response(InsightMessageSerializer(rows, many=True).data))
