## How to deploy on GCP

```
gcloud builds submit --tag gcr.io/<my-project>/face-restoration:latest --timeout=1200
```