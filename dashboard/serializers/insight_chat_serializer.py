from rest_framework import serializers

from dashboard.models import InsightConversation, InsightMessage


class InsightChatRequestSerializer(serializers.Serializer):
    conversationId = serializers.IntegerField(required=False)
    message = serializers.CharField(max_length=4000)
    dateFrom = serializers.DateField(required=False)
    dateTo = serializers.DateField(required=False)


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
