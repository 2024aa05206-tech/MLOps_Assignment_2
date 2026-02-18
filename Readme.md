
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

MLOPS_ASSIGNMENT_2/
│
├── .dvc/                     # DVC internal metadata
├── .github/
│   └── workflows/
│       └── cicd.yml          # CI/CD pipeline (tests + Docker build + push)
│
├── data/
│   ├── raw/                  # Raw dataset (tracked via DVC)
│   └── processed/            # Preprocessed dataset (tracked via DVC)
│
├── models/
│   └── baseline_cnn.pt       # Trained CNN model
│
├── outputs/
│   ├── confusion_matrix.png  # Evaluation confusion matrix
│   └── loss_curve.png        # Training loss curve
│
├── artifacts/                # Optional saved artifacts
├── mlruns/                   # MLflow tracking directory
│
├── src/
│   ├── data/
│   │   └── preprocessing.py  # Data preprocessing logic
│   │
│   └── models/
│       ├── model_file.py     # CNN architecture
│       ├── training.py       # Model training script
│       ├── training_utils.py # Helper utilities
│       └── kaggle_file.py    # Dataset download script
│
├── tests/
│   ├── test_model.py
│   ├── test_preprocess.py
│   └── test_training_utils.py
│
├── app.py                    # Flask inference app
├── Dockerfile                # Docker image definition
├── requirements.txt          # Python dependencies
├── dvc.yaml                  # DVC pipeline stages
├── dvc.lock                  # Locked DVC stage versions
├── mlflow.db                 # MLflow backend store
│
├── .dockerignore
├── .dvcignore
├── .env                      # Environment variables (not committed)
├── .gitignore
└── README.md

### Run Application Locally

## Setup

python -m venv venv

source venv/bin/activate  # Linux/Mac

venv\Scripts\activate   # Windows

pip install -r requirements.txt


## Run Pipeline with DVC

dvc repro

This will execute:

1.  Download Dataset

2.  Preprocess Data

3.  Train the Model

After training, the following will be generated:

-   `models/baseline_cnn.pt`

-   `outputs/confusion_matrix.png`

-   `outputs/loss_curve.png`

-   `mlruns/` (MLflow logs)


## Run Tests

python -m pytest tests


## Run Application

python app.py


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


## MLflow

mlflow ui

Open http://localhost:5000