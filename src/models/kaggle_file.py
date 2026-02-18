import os
import json
from pathlib import Path
import dotenv

DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"
DATA_DIR = "data/raw"

# Read secrets from .env file
dotenv.load_dotenv()
kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")

if not kaggle_username or not kaggle_key:
    raise RuntimeError("Kaggle secrets not found")

kaggle_creds = {
    "username": kaggle_username,
    "key": kaggle_key
}

# Create BOTH possible Kaggle config paths
paths = [
    Path.home() / ".kaggle",
    Path.home() / ".config" / "kaggle"
]

kaggle_config_dir = paths[1] # Set it to ~/.config/kaggle, which is the path mentioned in the error

for p in paths:
    p.mkdir(parents=True, exist_ok=True)
    kaggle_json = p / "kaggle.json"
    with open(kaggle_json, "w") as f:
        json.dump(kaggle_creds, f)
    os.chmod(kaggle_json, 0o600)

# Set KAGGLE_CONFIG_DIR environment variable to ensure Kaggle API finds the credentials
os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_config_dir)

print("kaggle.json written to all expected locations and KAGGLE_CONFIG_DIR set.")

# Import KaggleApi AFTER setting the environment variable and creating the config file
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate
api = KaggleApi()
api.authenticate()

# Download dataset
os.makedirs(DATA_DIR, exist_ok=True)
print("Downloading dataset...")
api.dataset_download_files(DATASET, path=DATA_DIR, unzip=True)

print("Dataset downloaded to data/raw/")