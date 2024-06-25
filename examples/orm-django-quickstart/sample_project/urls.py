"""
URL configuration for sample_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from . import views

urlpatterns = [
    path("", views.list_routes, "index"),
    path("insert_documents", views.insert_documents, name="insert_documents"),
    path("get_nearest_neighbors_documents", views.get_nearest_neighbors_documents, name="get_nearest_neighbors_documents"),
    path("get_documents_within_distance", views.get_documents_within_distance, name="get_documents_within_distance"),
]
