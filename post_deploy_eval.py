import csv
import requests

API_URL = "http://localhost:5000/predict"   # K8s/VM URL

def get_prediction(image_path: str) -> str:
    with open(image_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(API_URL, files=files, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["label"]

def main():
    y_true, y_pred = [], []

    with open("data/eval_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = row["true_label"]
            pred_label = get_prediction(row["image_path"])
            y_true.append(true_label)
            y_pred.append(pred_label)

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    print(y_pred)
    print(f"Post-deployment accuracy: {accuracy:.3f}")

    with open("outputs/post_deploy_metrics.txt", "w") as f:
        f.write(f"accuracy={accuracy:.3f}\n")

if __name__ == "__main__":
    main()