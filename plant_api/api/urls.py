from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.api_predict, name='api_predict'),

]
