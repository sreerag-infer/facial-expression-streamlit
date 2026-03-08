import streamlit as st
import cv2
import torch
import timm
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ------------------ CONFIG ------------------
DEVICE = torch.device("cpu")

classes = ['angry', 'disgust', 'fear', 'happy',
           'neutral', 'sad', 'surprise']

# ------------------ MODEL ------------------

class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.model = timm.create_model(
            'resnet18',
            pretrained=False,
            num_classes=7
        )

    def forward(self, x):
        return self.model(x)

model = FaceModel()
model.load_state_dict(
    torch.load("best-resnet18.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

# ------------------ TRANSFORM ------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ FACE DETECTOR ------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ------------------ STREAMLIT UI ------------------

st.title("Facial Expression Recognition (ResNet18)")
st.write("Real-time Face Detection + Emotion Recognition")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = f"{classes[pred]} ({confidence*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)
        cv2.putText(frame, label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()