from rest_framework import serializers

from dashboard.models import InsightConversation, InsightMessage


class InsightChatRequestSerializer(serializers.Serializer):
    conversationId = serializers.IntegerField(required=False)
    message = serializers.CharField(max_length=4000)
    dateFrom = serializers.DateField(required=False)
    dateTo = serializers.DateField(required=False)
    salesTeam = serializers.CharField(max_length=255, required=False, allow_blank=True)
    installer = serializers.CharField(max_length=255, required=False, allow_blank=True)
    leadSource = serializers.CharField(max_length=255, required=False, allow_blank=True)
    manager = serializers.CharField(max_length=255, required=False, allow_blank=True)
    market = serializers.CharField(max_length=255, required=False, allow_blank=True)
    repKind = serializers.CharField(max_length=32, required=False, allow_blank=True)
    repName = serializers.CharField(max_length=255, required=False, allow_blank=True)

    def validate(self, attrs):
        from dashboard.utils import normalize_rep_kind

        rk = normalize_rep_kind(attrs.get("repKind"))
        attrs["repKind"] = rk
        rn = (attrs.get("repName") or "").strip() or None
        attrs["repName"] = rn if rk else None
        return attrs


class InsightConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = InsightConversation
        fields = (
            "id",
            "title",
            "date_from",
            "date_to",
            "status",
            "created_at",
            "updated_at",
        )


class InsightMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = InsightMessage
        fields = (
            "id",
            "conversation_id",
            "role",
            "content",
            "intent_label",
            "context_meta",
            "created_at",
        )
