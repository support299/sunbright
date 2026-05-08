from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from dashboard.metrics.registry import list_metrics
from dashboard.services.filter_options_service import get_filter_options
from dashboard.services.project_service import (
    get_cancellation_reasons_breakdown,
    get_category_breakdown,
    get_on_hold_reasons_breakdown,
    get_overview_metrics,
)
from dashboard.utils import parse_dashboard_date_range, success_response


class OverviewView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        d0, d1 = parse_dashboard_date_range(request)
        return Response(success_response(get_overview_metrics(d0, d1, request.user)))


class CategoryBreakdownView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        d0, d1 = parse_dashboard_date_range(request)
        return Response(success_response(get_category_breakdown(d0, d1, request.user)))


class CancellationReasonsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        d0, d1 = parse_dashboard_date_range(request)
        return Response(success_response(get_cancellation_reasons_breakdown(d0, d1, request.user)))


class OnHoldReasonsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        d0, d1 = parse_dashboard_date_range(request)
        return Response(success_response(get_on_hold_reasons_breakdown(d0, d1, request.user)))


class FilterOptionsView(APIView):
    """Distinct values for the global dashboard filter dropdowns."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(success_response(get_filter_options(request.user)))


class MetricsRegistryView(APIView):
    """Expose the metrics dictionary so the UI can self-document."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(success_response([m.to_dict() for m in list_metrics()]))
