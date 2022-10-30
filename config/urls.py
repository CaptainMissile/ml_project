from django.contrib import admin
from django.urls import path

from entry import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path('get-summary/', views.get_summary, name = 'get_summary')
]
