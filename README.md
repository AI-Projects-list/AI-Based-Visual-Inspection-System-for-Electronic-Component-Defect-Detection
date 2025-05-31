# 🛠️ Component Defect Detector

An AI-powered visual inspection system to detect defects in electronic components using YOLOv8. Supports real-time detection, REST API integration, database logging, dashboard monitoring, and cloud deployment.

---

## 📸 Use Cases

- PCB visual inspection
- Soldering defect detection
- Component placement verification
- Real-time quality control in electronics manufacturing

---

## 🚀 Features

✅ Real-time detection with OpenCV  
✅ REST API using FastAPI  
✅ Deep learning model training with YOLOv8  
✅ Export to ONNX for efficient inference  
✅ Dockerized deployment  
✅ Logs to PostgreSQL  
✅ Streamlit-based dashboard  
✅ GCP deployment support

---

## 📁 Project Structure

```
component-defect-detector/
├── app/
│   ├── main.py           # Real-time detection with webcam
│   └── api.py            # FastAPI REST API
├── data/
│   └── data.yaml         # YOLOv8 dataset configuration
├── models/
│   └── best.onnx         # Exported ONNX model (after training)
├── train.py              # Model training script
├── log.py                # Log detection results to PostgreSQL
├── dashboard.py          # Streamlit dashboard for monitoring
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── gcp_deploy.md         # Deployment steps to Google Cloud
└── README.md             # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/component-defect-detector.git
cd component-defect-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Real-Time Detector
```bash
python app/main.py
```

### 4. Run the REST API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### 5. Train Your Own Model
```bash
python train.py
```

### 6. Launch the Dashboard
```bash
streamlit run dashboard.py
```

---

## 🧠 YOLOv8 Training Setup

### data.yaml
```yaml
path: ./data
train: images/train
val: images/val
test: images/test

nc: 5
names: ['OK', 'Missing_Part', 'Scratch', 'Bent_Pin', 'Soldering_Issue']
```

### Training Command
```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=416
```

---

## 🗃️ PostgreSQL Logging Setup

### Create detections table:
```sql
CREATE TABLE detections (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMP,
  class TEXT,
  score FLOAT
);
```

---

## ☁️ Deploy to Google Cloud

See [gcp_deploy.md](./gcp_deploy.md) for complete steps using Docker and Cloud Run.

---

## 🧪 API Testing

### Endpoint
```
POST /predict
```

### Sample cURL Request
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.jpg"
```

---

## 📊 Monitoring

Run the dashboard:
```bash
streamlit run dashboard.py
```

View all defect records from the PostgreSQL database in a live-updating table.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV, FastAPI, Streamlit, Docker, PostgreSQL
