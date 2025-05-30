# 🛠️ AI-Based Visual Inspection System for Electronic Component/PCBA Defect Detection

An AI-powered visual inspection system to detect defects in electronic components/PCBA using YOLOv8. Supports real-time detection, REST API integration, database logging, dashboard monitoring, and cloud deployment.

---

## 📸 Use Cases

- PCBA visual inspection
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
│ ├── main.py # Real-time detection with webcam
│ └── api.py # FastAPI REST API
├── data/
│ └── data.yaml # YOLOv8 dataset configuration
├── models/
│ └── best.onnx # Exported ONNX model (after training)
├── train.py # Model training script
├── log.py # Log detection results to PostgreSQL
├── dashboard.py # Streamlit dashboard for monitoring
├── Dockerfile # Docker configuration
├── requirements.txt # Python dependencies
├── gcp_deploy.md # Deployment steps to Google Cloud
└── README.md # Project documentation

```

