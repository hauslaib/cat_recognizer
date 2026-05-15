"""Watch Frigate's cat snapshots and label each with the identified cat's name.

Requires a trained model produced by train.py. If the model is missing the
service waits for it to appear so you can train without restarting the add-on.
"""
import os
import time

from PIL import Image, ImageDraw, ImageFont

from cat_identifier import CatIdentifier

FRIGATE_CLIPS_DIR = '/media/frigate/clips'
HA_WWW_DIR = '/config/www'
MODEL_PATH = '/share/cat_recognizer_model.joblib'

FOLDER_PATH = os.path.join(FRIGATE_CLIPS_DIR, 'cat')
OUTPUT_FOLDER = os.path.join(HA_WWW_DIR, 'cat_images')
POLL_INTERVAL_SECONDS = 10
MODEL_WAIT_SECONDS = 30
VALID_EXTS = {'.jpg', '.jpeg', '.png'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_image_paths(folder_path):
    paths = []
    if not os.path.isdir(folder_path):
        return paths
    for root, _, files in os.walk(folder_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                paths.append(os.path.join(root, f))
    return paths


def _load_font():
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, 24)
        except OSError:
            continue
    return ImageFont.load_default()


_FONT = None


def label_and_save(img_path, label, score, output_folder):
    global _FONT
    if _FONT is None:
        _FONT = _load_font()
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"{label} ({score:.2f})", fill=(255, 0, 0), font=_FONT)
    img.save(os.path.join(output_folder, os.path.basename(img_path)))
    img.save(os.path.join(output_folder, 'latest_cat.jpg'))


def wait_for_model(path):
    warned = False
    while not os.path.exists(path):
        if not warned:
            print(
                f"Trained model not found at {path}. "
                "Run train.py with labeled training images; this service will pick it up."
            )
            warned = True
        time.sleep(MODEL_WAIT_SECONDS)


def main():
    print("Starting cat recognizer...")
    wait_for_model(MODEL_PATH)
    identifier = CatIdentifier(MODEL_PATH)
    print(f"Loaded model with {len(identifier.classes)} cats: {identifier.classes}")

    processed = set()
    while True:
        for img_path in get_image_paths(FOLDER_PATH):
            if img_path in processed:
                continue
            processed.add(img_path)
            try:
                label, score = identifier.identify(img_path)
                label_and_save(img_path, label, score, OUTPUT_FOLDER)
                print(f"{os.path.basename(img_path)}: {label} (score={score:.3f})")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == '__main__':
    main()
