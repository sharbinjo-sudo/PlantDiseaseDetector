from django.contrib import admin
from .models import PredictionRecord

@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'predicted_label', 'confidence', 'model_type', 'created_at')
    search_fields = ('predicted_label', 'model_type')
    list_filter = ('model_type', 'created_at')

