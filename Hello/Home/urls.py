from django.contrib import admin
from django.urls import path
from Home import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # once redirected here, then a function call back is attached to the request which specifies what to return
    path('',views.index, name='Home'),
    path('about',views.about, name='about'),
    path('index',views.index, name='index'),
    path('regressor/',views.regressor, name='regressor'),
    path('regressor_from_input',views.regressor_from_input, name='regressor_from_input'),
    path('input',views.input, name='about'),
    path('get_video',views.get_video, name='get_video'),
    path('get_video_classification',views.get_video_classification, name='get_video_classification'),

]