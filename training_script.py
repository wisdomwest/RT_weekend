#!/usr/bin/env python
import os
import argparse
import zipfile
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from collections import Counter
from sklearn.utils import resample
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from sklearn.utils.class_weight import compute_class_weight

##############################
# Utility Functions
##############################
def load_data_paths(data_dir, classes, subset="train", img_size=(224, 224), rebuild_cache=False):
    """Caches file paths and labels instead of full images."""
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{subset}_paths"
    cache_path = os.path.join(cache_dir, f"{cache_key}.npz")

    if not rebuild_cache and os.path.exists(cache_path):
        print(f"♻️ Loading cached file paths from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data['image_paths'], data['class_labels'], data['bboxes']
        except Exception as e:
            print(f"⚠ Cache loading failed: {e}. Reprocessing...")

    print(f"🔨 Processing data paths from scratch for {subset}...")
    image_paths, class_labels, bboxes = _load_data_paths(data_dir, classes, subset)

    print(f"💾 Caching processed file paths to {cache_path}")
    np.savez(cache_path, image_paths=image_paths, class_labels=class_labels, bboxes=bboxes)
    return image_paths, class_labels, bboxes


def _load_data_paths(data_dir, classes, subset="train"):
    """Load image paths with corresponding class labels and bounding boxes."""
    subset_folder = os.path.join(data_dir, subset)
    
    # Normalize class names to handle spaces
    class_to_idx = {cls.replace(" ", "_"): idx for idx, cls in enumerate(classes)}
    
    image_paths, class_indices, bboxes = [], [], []

    for cls in classes:
        cls_normalized = cls.replace(" ", "_")
        cls_folder = os.path.join(subset_folder, cls)
        labels_folder = os.path.join(cls_folder, "Label")

        if not os.path.exists(labels_folder):
            continue

        image_files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(cls_folder, img_file)
            label_path = os.path.join(labels_folder, os.path.splitext(img_file)[0] + ".txt")

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                line = f.readline().strip()

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                coords = list(map(float, parts[-4:]))
                class_name = ' '.join(parts[:-4]).replace(" ", "_")  # Normalize class name from label
            except ValueError:
                continue

            if class_name != cls_normalized:
                continue

            image_paths.append(img_path)
            class_indices.append(class_to_idx[cls_normalized])
            bboxes.append(coords)

    return np.array(image_paths), np.array(class_indices), np.array(bboxes)


def preprocess_image(img_path, img_size=(224, 224)):
    """Loads and preprocesses a single image on the fly."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size) / 255.0
    return img


def create_dataset(image_paths, class_labels, bboxes, batch_size=32, img_size=(224, 224)):
    """Creates a TensorFlow dataset for efficient training."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, class_labels, bboxes))
    dataset = dataset.shuffle(len(image_paths))
    dataset = dataset.map(lambda path, label, bbox: (preprocess_image(path, img_size), (tf.one_hot(label, depth=len(set(class_labels))), bbox)),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_model(input_shape=(224, 224, 3), num_classes=80):
    """Creates a multi-task CNN with ResNet50 for classification and bounding box regression."""
    base_model = ResNet50(include_top=False, weights='imagenet')
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)  # ✅ Apply augmentation
    x = base_model(x, training=False)  # Pass through ResNet50
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    class_head = layers.Dense(512, activation='relu')(x)
    class_head = layers.Dropout(0.5)(class_head)
    class_out = layers.Dense(num_classes, activation='softmax', name='class_output')(class_head)
    
    # BBox regression head
    bbox_head = layers.Dense(512, activation='relu')(x)
    bbox_head = layers.Dense(256, activation='relu')(bbox_head)
    bbox_out = layers.Dense(4, activation='sigmoid', name='bbox_output')(bbox_head)
    
    return tf.keras.Model(inputs, [class_out, bbox_out])

##############################
# Training Function
##############################
def train_model(args):
    # Load dataset
    train_folder = os.path.join(args.data_extract_path, "train")
    classes = [c for c in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, c))]
    
    train_image_paths, train_class_labels, train_bboxes = load_data_paths(args.data_extract_path, classes, subset="train")
    val_image_paths, val_class_labels, val_bboxes = load_data_paths(args.data_extract_path, classes, subset="test")

    train_dataset = create_dataset(train_image_paths, train_class_labels, train_bboxes, batch_size=args.batch_size)
    val_dataset = create_dataset(val_image_paths, val_class_labels, val_bboxes, batch_size=args.batch_size)

    def smooth_l1_loss(y_true, y_pred):
        loss = tf.keras.losses.Huber()(y_true, y_pred)
        return loss
    # Build and compile the model
    model = create_model(input_shape=(224, 224, 3), num_classes=len(classes))
    model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss={
        'class_output': 'categorical_crossentropy',
        'bbox_output': 'mse'
    },
    metrics={
        'class_output': 'accuracy',
        'bbox_output': 'mae'
    },
    loss_weights={'class_output': 0.05, 'bbox_output': 0.05}  # Reduce bbox weight
    )

    # Train the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs)

    # Save final model
    model.save("outputs/final_model.h5")
    print("✅ Training complete. Model saved.")

##############################
# Main Function
##############################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_extract_path", type=str, default="data/extracted")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    args = parser.parse_args()

    train_model(args)

if __name__ == "__main__":
    main()

