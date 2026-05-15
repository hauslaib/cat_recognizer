# Home Assistant Add-ons

This repository contains custom add-ons for Home Assistant.

## Add-ons

- **Cat Recognizer**: An add-on to recognize and distinguish between cats using machine learning.

## How to Use

Add this repository URL to your Home Assistant add-on store to install the custom add-ons.

## Cat Recognizer: Training

The recognizer uses a supervised classifier (VGG16 features + nearest-prototype /
k-NN with cosine similarity). Train it once with labeled photos of each cat:

1. On the Home Assistant host, create one folder per cat under
   `/share/cat_training_images/`, named after the cat:

   ```
   /share/cat_training_images/
       Whiskers/  whiskers_01.jpg whiskers_02.jpg ...
       Mittens/   mittens_01.jpg ...
       Luna/      ...
       Oreo/      ...
   ```

   Aim for 15–30 varied photos per cat (different poses, lighting, angles).

2. Open a shell inside the add-on container and run:

   ```
   python3 train.py
   ```

   This writes `/share/cat_recognizer_model.joblib`. The running service
   picks it up automatically and starts labeling new Frigate snapshots with
   the cat's name (or `unknown` when no match is confident enough).

Re-run `train.py` whenever you add or correct training images.
