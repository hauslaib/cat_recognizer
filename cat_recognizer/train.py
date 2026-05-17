"""Train a cat identifier from labeled images.

Expected layout:
    <training-dir>/
        Whiskers/  img1.jpg img2.jpg ...
        Mittens/   img1.jpg ...
        Luna/      ...
        Oreo/      ...

Each subfolder name is used as the cat's label. Run:
    python3 train.py --training-dir /share/cat_training_images \\
                     --model-path  /share/cat_recognizer_model.joblib
"""
import argparse
import os
import sys

import joblib
import numpy as np

from feature_extractor import extract_features, l2_normalize

DEFAULT_TRAINING_DIR = '/share/cat_training_images'
DEFAULT_MODEL_PATH = '/share/cat_recognizer_model.joblib'
VALID_EXTS = {'.jpg', '.jpeg', '.png'}


def collect_labeled_images(training_dir):
    samples = []
    for name in sorted(os.listdir(training_dir)):
        sub = os.path.join(training_dir, name)
        if not os.path.isdir(sub):
            continue
        for f in sorted(os.listdir(sub)):
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                samples.append((name, os.path.join(sub, f)))
    return samples


def train(training_dir, model_path):
    if not os.path.isdir(training_dir):
        sys.exit(f"Training directory does not exist: {training_dir}")

    samples = collect_labeled_images(training_dir)
    if not samples:
        sys.exit(
            f"No training images found under {training_dir}. "
            "Expected one subfolder per cat, each containing .jpg/.jpeg/.png images."
        )

    labels = [s[0] for s in samples]
    paths = [s[1] for s in samples]
    classes = sorted(set(labels))
    print(f"Found {len(samples)} images across {len(classes)} cats: {classes}")

    feats = np.zeros((len(samples), 4096), dtype=np.float32)
    for i, p in enumerate(paths):
        feats[i] = extract_features(p)
        if (i + 1) % 10 == 0 or i == len(paths) - 1:
            print(f"  extracted {i + 1}/{len(samples)}")
    feats = l2_normalize(feats)

    prototypes = np.zeros((len(classes), feats.shape[1]), dtype=np.float32)
    for idx, c in enumerate(classes):
        mask = np.array([lab == c for lab in labels])
        prototypes[idx] = l2_normalize(feats[mask].mean(axis=0))
        print(f"  {c}: {int(mask.sum())} images")

    model = {
        'classes': classes,
        'prototypes': prototypes,
        'samples': feats,
        'sample_labels': labels,
    }
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--training-dir', default=DEFAULT_TRAINING_DIR)
    p.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    args = p.parse_args()
    train(args.training_dir, args.model_path)


if __name__ == '__main__':
    main()
