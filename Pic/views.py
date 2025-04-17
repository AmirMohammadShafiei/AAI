
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UploadedImageSerializer
from .models import UploadedImage
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from gtts import gTTS

class ImageRecognitionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = UploadedImageSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            image_path = instance.image.path

            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0)

            model = models.resnet18(pretrained=True)
            model.eval()

            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = outputs.max(1)

            labels_path = os.path.join(torch.hub.get_dir(), 'imagenet_classes.txt')
            if not os.path.exists(labels_path):
                import urllib.request
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
                    labels_path
                )

            with open(labels_path) as f:
                labels = [line.strip() for line in f.readlines()]
            label = labels[predicted.item()]

            tts = gTTS(text=label, lang='en')
            audio_path = image_path.rsplit('.', 1)[0] + '.mp3'
            tts.save(audio_path)

            return Response({
                'label': label,
                'audio_file': request.build_absolute_uri('/media/' + os.path.basename(audio_path))
            }, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    



import base64
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.conf import settings
from openai import OpenAI
from gtts import gTTS
class ImageRecognitionAPIView(APIView):
    def post(self, request):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        image_file = request.FILES['image']
        image_path = default_storage.save('uploads/' + image_file.name, image_file)
        full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
        with open(full_image_path, "rb") as img:
            base64_image = base64.b64encode(img.read()).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{base64_image}"
        client = OpenAI(api_key="")  
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            max_tokens=500
        )

        gpt_text = response.choices[0].message.content

        tts = gTTS(text=gpt_text, lang='en')
        audio_filename = "audio_output.mp3"
        audio_path = os.path.join(settings.MEDIA_ROOT, 'tts', audio_filename)
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        tts.save(audio_path)

        audio_url = request.build_absolute_uri(settings.MEDIA_URL + 'tts/' + audio_filename)

        return Response({
            "description": gpt_text,
            "audio_url": audio_url
        }, status=status.HTTP_200_OK)