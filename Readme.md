
# Cats vs Dogs Classification – MLOps Assignment 2 (Group 77)

This project implements an **end-to-end MLOps pipeline** for classifying images of **Cats and Dogs** using a **Convolutional Neural Network (CNN)**.

## M1 – Model Development & Experiment Tracking

### Data & Code Versioning
- Git is used for source code versioning.
- DVC is used for dataset, processed data, and model versioning.
  - Raw dataset (data/raw)
  - Processed dataset (data/processed)

### Model Building
- Baseline CNN implemented using PyTorch
- Images resized to 224×224 RGB
- Model artifact saved as:
  - models/baseline_cnn.pt

### Experiment Tracking
- MLflow logs parameters, metrics, and artifacts
- Confusion matrix and loss curve stored as artifacts

## M2 – Model Packaging & Containerization

### Inference Service
- Flask-based REST API
- Endpoints:
  - GET /health
  - POST /predict

### Environment Specification
- Dependencies defined in requirements.txt
- Version pinning for reproducibility

### Containerization
- Dockerfile used to build inference image
- Image tested locally via curl/Postman

## M3 – CI Pipeline

### Automated Testing
- PyTest-based unit tests
- Covers preprocessing and model inference utilities

### CI Setup
- Implemented using GitHub Actions
- Runs tests and builds Docker image on each push

### Artifact Publishing
- Docker credentials are stored using GitHub Secrets:
    - DOCKERHUB_USERNAME
    - DOCKERHUB_TOKEN
- Docker image pushed to Docker Hub
- Image tag: 2024aa05206/cats-dogs-mlops:latest


## Project Structure

```bash
MLOPS_ASSIGNMENT_2/
│
├── .dvc/                     
├── .github/
│   └── workflows/
│       └── cicd.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── baseline_cnn.pt
│
├── outputs/
│   ├── confusion_matrix.png
│   └── loss_curve.png
│
├── artifacts/
├── mlruns/
│
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   │
│   └── models/
│       ├── model_file.py
│       ├── training.py
│       ├── training_utils.py
│       └── kaggle_file.py
│
├── tests/
│   ├── test_model.py
│   ├── test_preprocess.py
│   └── test_training_utils.py
│
├── app.py
├── Dockerfile
├── requirements.txt
├── dvc.yaml
├── dvc.lock
├── mlflow.db
│
├── .dockerignore
├── .dvcignore
├── .env
├── .gitignore
└── README.md
```


### Run Application Locally

**Setup**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Run Pipeline with DVC**
```bash
dvc repro
```
This will execute:

1.  Download Dataset
2.  Preprocess Data
3.  Train the Model

After training, the following will be generated:

-   `models/baseline_cnn.pt`
-   `outputs/confusion_matrix.png`
-   `outputs/loss_curve.png`
-   `mlruns/` (MLflow logs)

**Run Tests**
```bash
python -m pytest tests
```

**Run Application**
```bash
python app.py
```

## Docker - Instead of running application from local, pull the image from docker and run application

docker pull 2024aa05206/cats-dogs-mlops:latest

docker run -p 5000:5000 2024aa05206/cats-dogs-mlops:latest


## Test Endpoints

**Health Check**
```bash
curl http://localhost:5000/health
```

**Prediction**
```bash
curl -X POST http://localhost:5000/predict -F "file=@dog.jpg"
```


## MLflow

**mlflow ui**
```bash
Open http://localhost:5000
```