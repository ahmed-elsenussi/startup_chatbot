from django.urls import path
from . import views

urlpatterns = [
    path("suggest_companies/", views.suggest_companies, name="suggest_companies"),
]
