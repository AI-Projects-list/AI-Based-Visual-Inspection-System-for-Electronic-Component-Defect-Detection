from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2

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
