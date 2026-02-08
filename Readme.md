# Cats vs Dogs – MLOps Pipeline

## How to Run (VS Code)

### 1. Create virtual env
python -m venv venv
venv\Scripts\activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run python files
python kaggle_file.py
python preprocessing.py
python training.py

### 4. Run tests
python -m pytest tests

### 4. Run app.py
python app.py

### 5. Docker
docker build -t cats-dogs-flask .
docker run -p 5000:5000 cats-dogs-flask

### 6. Test url
Health check : curl http://localhost:5000/health
Predict : curl -X POST http://localhost:5000/predict -F "file=@<filename with full location>"
