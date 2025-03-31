#!/usr/bin/env python
import os
import argparse
import zipfile
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from multiprocessing import cpu_count
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Azure ML SDK imports
from azureml.core import Workspace, Dataset, Experiment, ScriptRunConfig, Environment, Model

##############################
# Utility Functions
##############################
def unzip_data(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def parse_annotation(labels_folder, image_filename):
    base = os.path.splitext(image_filename)[0]
    label_file = os.path.join(labels_folder, base + ".txt")
    if not os.path.exists(label_file):
        return None, None
    with open(label_file, 'r') as f:
        line = f.readline().strip()
        parts = line.split()
        if len(parts) < 5:
            return None, None
        label = parts[0]
        bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        return label, bbox


def load_data(data_dir, classes, subset="train", img_size=(224, 224)):
    subset_folder = os.path.join(data_dir, subset)
    
    if not os.path.exists(subset_folder):
        print(f"⚠ Warning: {subset_folder} does not exist!")
        return np.array([]), np.array([]), np.array([])

    # Create class-to-index mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    print(f"Class mapping: {class_to_idx}")

    # First pass: Collect all valid data points
    image_paths, class_indices, bboxes = [], [], []
    
    for cls in classes:
        cls_folder = os.path.join(subset_folder, cls)
        labels_folder = os.path.join(cls_folder, "Label")

        if not os.path.exists(labels_folder):
            print(f"⚠ Warning: No labels folder for {cls}. Skipping.")
            continue

        # Get all image files with corresponding labels
        image_files = [f for f in os.listdir(cls_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {cls}"):
            img_path = os.path.join(cls_folder, img_file)
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_folder, txt_file)

            if not os.path.exists(label_path):
                print(f"⚠ Missing label for {img_file}. Skipping.")
                continue

            # Parse label file
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                
            if not line:
                print(f"⚠ Empty label in {label_path}. Skipping.")
                continue

            parts = line.split()
            split_idx = next((i for i, p in enumerate(parts) 
                           if p.replace('.', '').isdigit()), None)
            
            if split_idx is None or split_idx < 1:
                print(f"⚠ Invalid label format in {label_path}. Skipping.")
                continue

            class_name = ' '.join(parts[:split_idx])
            if class_name not in class_to_idx:
                print(f"⚠ Unknown class '{class_name}'. Skipping.")
                continue

            try:
                bbox = list(map(float, parts[split_idx:split_idx+4]))
            except ValueError:
                print(f"⚠ Invalid coordinates in {label_path}. Skipping.")
                continue

            image_paths.append(img_path)
            class_indices.append(class_to_idx[class_name])
            bboxes.append(bbox)

    # Second pass: Parallel image loading using TensorFlow
    def tf_load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        return tf.cast(img, tf.float32) / 255.0

    print(f"🔄 Loading {len(image_paths)} images...")
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(tf_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    images = [img.numpy() for img in tqdm(dataset, desc="Processing images")]

    # Convert to exact same format as original
    return (
        np.array(images),
        to_categorical(np.array(class_indices)),  # Maintain one-hot encoding
        np.array(bboxes)
    )

def create_model(input_shape=(224,224,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    bbox_output = layers.Dense(4, activation='linear', name='bbox_output')(x)
    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output])
    return model

##############################
# Training Function
##############################
def train_model(args):
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            tf.config.set_visible_devices([], 'GPU')
            print("No GPU detected. Running on CPU-only mode.")
    except Exception as e:
        print("GPU detection failed:", str(e))

    # Define the local path for the zip file and extraction folder.
    local_zip_path = os.path.join("data", "archive.zip")
    extract_folder = args.data_extract_path

    # Check if the file already exists locally.
    if args.zip_data_path.startswith("azureml://"):
        if os.path.exists(local_zip_path):
            print(f"Local dataset file found at {local_zip_path}. Skipping download.")
            zip_path = local_zip_path
        else:
            print("Detected Azure ML URI. Downloading dataset...")
            ws = Workspace(
                subscription_id="56fee90f-c26a-41c2-baa2-6206e26a96ac",
                resource_group="AI_west",
                workspace_name="West_ai"
            )
            datastore = ws.get_default_datastore()
            # Parse the relative path from the URI.
            # Expected format: .../paths/<folder_path>/archive.zip
            relative_path = args.zip_data_path.split("paths/")[-1]
            print("Parsed relative path:", relative_path)
            try:
                dataset = Dataset.File.from_files(path=(datastore, relative_path))
                local_paths = dataset.download(target_path="data", overwrite=True)
                if local_paths:
                    zip_path = local_paths[0]
                else:
                    raise Exception("Dataset download returned empty path list.")
                print(f"Dataset downloaded locally to {zip_path}")
            except Exception as e:
                print("Error downloading dataset:", str(e))
                raise
    else:
        # If not an Azure ML URI, assume it's a local path.
        zip_path = args.zip_data_path

    # Check if extraction folder exists; if not, unzip the file.
    if os.path.exists(extract_folder) and os.listdir(extract_folder):
        print(f"Extraction folder '{extract_folder}' already exists and is not empty. Skipping extraction.")
    else:
        unzip_data(zip_path, extract_folder)
        
    # Determine list of classes from the train folder.
    train_folder = os.path.join(args.data_extract_path, "train")
    classes = [c for c in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, c))]
    print("Classes found:", classes)
    
    # Load training data (from train folder) and validation data (from test folder).
    train_images, train_class_labels, train_bboxes = load_data(args.data_extract_path, classes, subset="train", img_size=(224,224))
    val_images, val_class_labels, val_bboxes = load_data(args.data_extract_path, classes, subset="test", img_size=(224,224))
    print(f"Loaded {len(train_images)} training images and {len(val_images)} validation images.")
    
    # Build and compile the model.
    model = create_model(input_shape=(224,224,3), num_classes=len(classes))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss={'class_output': 'categorical_crossentropy', 'bbox_output': 'mse'},
        loss_weights={'class_output': 1.0, 'bbox_output': 1.0},
        metrics={'class_output': 'accuracy', 'bbox_output': 'mse'}
    )
    
    # Define callbacks.
    checkpoint_cb = callbacks.ModelCheckpoint("outputs/model_epoch_{epoch:02d}.keras",

                                                monitor='val_class_output_accuracy',
                                                mode='max',
                                                save_best_only=False,
                                                verbose=1)
    earlystop_cb = callbacks.EarlyStopping(monitor='val_class_output_accuracy', patience=5, restore_best_weights=True)
    reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    history = model.fit(
        train_images,
        {'class_output': train_class_labels, 'bbox_output': train_bboxes},
        validation_data=(val_images, {'class_output': val_class_labels, 'bbox_output': val_bboxes}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
    )
    
    # Save the final model.
    model.save("outputs/final_model.h5")
    print("Training complete. Final model saved as outputs/final_model.h5")

##############################
# HyperDrive Submission Function
##############################
def submit_hyperdrive(args):
    ws = Workspace(
        subscription_id="56fee90f-c26a-41c2-baa2-6206e26a96ac",
        resource_group="AI_west",
        workspace_name="West_ai"
    )
    env = Environment.get(workspace=ws, name="AzureML-TensorFlow-2.4-CPU")
    src = ScriptRunConfig(
        source_directory=".",
        script="hyper_training.py",
        arguments=[
            "--zip_data_path", args.zip_data_path,
            "--data_extract_path", args.data_extract_path,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate)
        ],
        compute_target=args.compute_target,
        environment=env
    )
    param_sampling = RandomParameterSampling({
        "--learning_rate": uniform(0.0001, 0.01),
        "--batch_size": choice(16, 32, 64)
    })
    early_termination_policy = BanditPolicy(slack_factor=0.1, evaluation_interval=1, delay_evaluation=3)
    hd_config = HyperDriveConfig(
        run_config=src,
        hyperparameter_sampling=param_sampling,
        policy=early_termination_policy,
        primary_metric_name="val_class_output_accuracy",
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=20,
        max_concurrent_runs=4
    )
    experiment = Experiment(ws, "AnimalDetectionHyperDrive")
    hd_run = experiment.submit(hd_config)
    hd_run.wait_for_completion(show_output=True)
    print("HyperDrive run complete.")
    
    # Download best model from best run.
    best_run = hd_run.get_best_run_by_primary_metric()
    print("Best run ID:", best_run.id)
    best_run.download_file("outputs/final_model.h5", "best_model.h5")
    print("Best model downloaded as best_model.h5")

##############################
# Main Function
##############################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "hyperdrive"],
                        help="Run mode: 'train' for a single training run, 'hyperdrive' for hyperparameter tuning.")
    parser.add_argument("--zip_data_path", type=str, default="azureml://subscriptions/56fee90f-c26a-41c2-baa2-6206e26a96ac/resourcegroups/AI_west/workspaces/West_ai/datastores/workspaceblobstore/paths/UI/2025-03-31_105413_UTC/",
                        help="Azure ML URI for the zipped dataset. (Filename is archive.zip)")
    parser.add_argument("--data_extract_path", type=str, default="data/extracted",
                        help="Local folder where the dataset will be unzipped.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--compute_target", type=str, default="wizziekoome1",
                        help="Name of the compute target (e.g., 'cpu-cluster').")
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "hyperdrive":
        submit_hyperdrive(args)

if __name__ == "__main__":
    main()

