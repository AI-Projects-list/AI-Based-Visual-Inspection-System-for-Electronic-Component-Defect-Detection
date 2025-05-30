1. Create a Google Cloud Project
2. Enable Cloud Run and Container Registry
3. Install gcloud CLI and authenticate:
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID

4. Build Docker Image:
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/defect-detector

5. Deploy to Cloud Run:
   gcloud run deploy defect-detector \
     --image gcr.io/YOUR_PROJECT_ID/defect-detector \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
