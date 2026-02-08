from pathlib import Path
from PIL import Image

RAW_DIR = Path("data/raw/PetImages")
PROCESSED_DIR = Path("data/processed")
IMG_SIZE = (224, 224)

for label in ["Cat", "Dog"]:
    input_dir = RAW_DIR / label
    output_dir = PROCESSED_DIR / label.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(input_dir.iterdir()):
        for img_path in input_dir.iterdir():
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                img.save(output_dir / img_path.name)
            except Exception as e:
                print(f"Could not process {img_path}: {e}")
    else:
        print(f"Warning: {input_dir} is empty")


print("Preprocessing completed")

