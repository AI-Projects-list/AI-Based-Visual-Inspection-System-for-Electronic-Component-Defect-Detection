# component-defect-detector

## Main Inference Script (main.py)
import cv2
from ultralytics import YOLO

model = YOLO('models/best.onnx')
cap = cv2.VideoCapture(0)  # Change to video file path if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('Defect Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

## FastAPI App (api.py)
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO

app = FastAPI()
model = YOLO("models/best.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    results = model(image)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    scores = results[0].boxes.conf.tolist()
    return {"boxes": boxes, "classes": classes, "scores": scores}

## Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

## requirements.txt
ultralytics
fastapi
uvicorn
opencv-python
numpy
pandas
sqlalchemy
psycopg2-binary

## Training Script (train.py)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=50, imgsz=416)
model.export(format='onnx')

## data.yaml (YAML file for YOLOv8 training)
# path: ./data
# train: images/train
# val: images/val
# test: images/test

# number of classes
nc: 5
names: ['OK', 'Missing_Part', 'Scratch', 'Bent_Pin', 'Soldering_Issue']

## Logging to Database (log.py)
import psycopg2
from datetime import datetime

def log_detection(result):
    conn = psycopg2.connect(dbname="defectdb", user="admin", password="admin", host="localhost")
    cur = conn.cursor()
    cur.execute("INSERT INTO detections (timestamp, class, score) VALUES (%s, %s, %s)",
                (datetime.now(), result['class'], result['score']))
    conn.commit()
    cur.close()
    conn.close()

## Simple Dashboard (dashboard.py)
import pandas as pd
import sqlalchemy
import streamlit as st

engine = sqlalchemy.create_engine('postgresql://admin:admin@localhost/defectdb')
df = pd.read_sql("SELECT * FROM detections ORDER BY timestamp DESC", engine)

st.title("Defect Detection Dashboard")
st.dataframe(df)

## GCP Deployment (gcp_deploy.md)
1. Create a Google Cloud Project
2. Enable Cloud Run and Container Registry
3. Install gcloud CLI and authenticate:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
4. Build Docker Image:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/defect-detector
   ```
5. Deploy to Cloud Run:
   ```bash
   gcloud run deploy defect-detector \
     --image gcr.io/YOUR_PROJECT_ID/defect-detector \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```
6. Access via provided URL

## Sample usage (bash)
# python main.py
# or
# docker build -t component-defect-detector .
# docker run -p 8000:8000 component-defect-detector

## README.md

# 🛠️ Component Defect Detector

An AI-powered visual inspection tool for detecting electronic component defects using YOLOv8 and computer vision. Supports real-time webcam or image-based detection, REST API, Docker deployment, database logging, and Streamlit dashboard.

---

## 📦 Features
- Real-time inference with OpenCV
- REST API with FastAPI
- Training pipeline with YOLOv8
- Dockerized for easy deployment
- PostgreSQL logging
- Streamlit dashboard for monitoring
- GCP deployment guide

---

## 🚀 Getting Started

### 1. Clone the repo:
```bash
git clone https://github.com/your-username/component-defect-detector.git
cd component-defect-detector
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run inference:
```bash
python app/main.py
```

### 4. Run API:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### 5. Run Streamlit dashboard:
```bash
streamlit run dashboard.py
```

### 6. Train new model:
```bash
python train.py
```

---

## 📊 Database Logging
Ensure PostgreSQL is installed and create the `detections` table:
```sql
CREATE TABLE detections (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMP,
  class TEXT,
  score FLOAT
);
```

---

## ☁️ Deploy on GCP
Follow `gcp_deploy.md` for full steps to containerize and deploy using Cloud Run.

---

## 📁 Project Structure
```
component-defect-detector/
├── app/
│   ├── main.py
│   └── api.py
├── data/
│   └── data.yaml
├── models/
│   └── best.onnx
├── train.py
├── log.py
├── dashboard.py
├── Dockerfile
├── requirements.txt
├── gcp_deploy.md
└── README.md
```

---

## 📜 License
MIT License. See `LICENSE` for details.
