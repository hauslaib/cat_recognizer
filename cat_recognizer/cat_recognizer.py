import os
import time
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont

# Paths inside the add-on container
FRIGATE_CLIPS_DIR = '/media/frigate/clips'  # Adjust this path as per your Frigate setup
HA_WWW_DIR = '/config/www'                  # Home Assistant www directory

# Path to Frigate's snapshot folder for cat detections
folder_path = os.path.join(FRIGATE_CLIPS_DIR, 'cat')  # Update based on your Frigate setup

# Output folder for labeled images
output_folder = os.path.join(HA_WWW_DIR, 'cat_images')

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def get_image_paths(folder_path):
    """Get a list of image file paths in the given folder."""
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(root, f))
    return image_paths

def load_and_extract_features(image_paths):
    """Load images and extract features using the VGG16 model."""
    features = []
    for img_path in image_paths:
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)
            features.append(feature.flatten())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return np.array(features)

def label_and_save_images(image_paths, clusters, output_folder):
    """Label images with their cluster and save them to the output folder."""
    for img_path, cluster in zip(image_paths, clusters):
        try:
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            text_position = (10, 10)
            text = f"Cat {cluster}"
            draw.text(text_position, text, fill=(255, 0, 0), font=font)
            base_name = os.path.basename(img_path)
            output_path = os.path.join(output_folder, base_name)
            img.save(output_path)
            # Save the latest image as 'latest_cat.jpg' for easy reference
            latest_image_path = os.path.join(output_folder, 'latest_cat.jpg')
            img.save(latest_image_path)
        except Exception as e:
            print(f"Error labeling {img_path}: {e}")

# Initialize variables
processed_images = set()
features = np.empty((0, 4096))
image_paths = []
k = 4  # Number of cats to distinguish
kmeans = KMeans(n_clusters=k, random_state=0)

print("Starting the cat detection script...")
while True:
    current_image_paths = get_image_paths(folder_path)
    new_image_paths = [img_path for img_path in current_image_paths if img_path not in processed_images]

    if new_image_paths:
        print(f"Found {len(new_image_paths)} new images. Processing...")
        # Extract features for new images
        new_features = load_and_extract_features(new_image_paths)

        # Append features and image paths
        features = np.vstack([features, new_features])
        image_paths.extend(new_image_paths)

        # Update processed images
        processed_images.update(new_image_paths)

        # Re-cluster using all available features
        clusters = kmeans.fit_predict(features)

        # Label and save images
        label_and_save_images(image_paths, clusters, output_folder)

        # Log clustering results
        for img_path, cluster in zip(image_paths, clusters):
            print(f"Image {os.path.basename(img_path)} is identified as Cat {cluster}")

    # Sleep before checking for new images again
    time.sleep(10)