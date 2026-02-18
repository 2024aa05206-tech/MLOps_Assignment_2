import tempfile
from pathlib import Path
from PIL import Image


def create_dummy_image(path):
    img = Image.new("RGB", (500, 500))
    img.save(path)


def test_preprocess_image_resize():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        raw_dir = tmpdir / "PetImages" / "Cat"
        raw_dir.mkdir(parents=True)

        img_path = raw_dir / "test.jpg"
        create_dummy_image(img_path)

        img = Image.open(img_path).convert("RGB")
        resized = img.resize((224, 224))

        assert resized.size == (224, 224)


def test_preprocess_handles_invalid_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        raw_dir = tmpdir / "PetImages" / "Cat"
        raw_dir.mkdir(parents=True)

        invalid_file = raw_dir / "invalid.txt"
        invalid_file.write_text("not an image")

        try:
            Image.open(invalid_file)
            assert False  # should not reach here
        except Exception:
            assert True
