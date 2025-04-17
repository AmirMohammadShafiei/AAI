

from django.urls import path
from .views import ImageRecognitionAPIView,ImageRecognitionAPIView

urlpatterns = [
    path('recognize/', ImageRecognitionAPIView.as_view(), name='recognize'),
    path('analyze-image/', ImageRecognitionAPIView.as_view(), name='analyze-image'),
]
