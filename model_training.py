import os
import random
import json
import time

import numpy as np
import tensorflow as tf

import keras
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam

from face_detector_model import FaceDetectorModel, compute_lr_decay, localization_loss_func, plot_training_performance
from utils import image_to_tensor, plot_image_with_rectangle, gpu_check, apply_gpu_optimizations


def get_label_info(label_path) -> tuple[list[tf.uint32], list[tf.float32]]:
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']


def train_val_test_split(dataset: tf.data.Dataset, train_percentage: float = 0.65,
                         val_percentage: float = 0.3,
                         batch_size: int = 8) -> tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    # Check if the sum of train_percentage and val_percentage is less than 1 (to leave room for test_dataset)
    if train_percentage + val_percentage >= 1:
        raise ValueError(f"No room left for test dataset: the sum of train_percentage and val_percentage"
                         f"is not less than 1")

    # Get the number of samples
    dataset_sample_number: int = len(dataset)

    # Shuffle the dataset in order to avoid overfitting
    # The shuffle buffer size is bigger than the dataset size in order to shuffle the entire dataset
    dataset = dataset.shuffle(int(1.2 * dataset_sample_number))

    # Get the number of samples for training and validation datasets
    train_samples: int = int(train_percentage * dataset_sample_number)
    val_samples: int = int(val_percentage * dataset_sample_number)

    # Split the dataset into training, validation and test sets
    train_dataset: tf.data.Dataset = dataset.take(train_samples)
    val_dataset: tf.data.Dataset = dataset.skip(train_samples).take(val_samples)
    test_dataset: tf.data.Dataset = dataset.skip(train_samples + val_samples)

    # Batch and prefetch the datasets
    # This is done in order to improve performance during training
    # thanks to the parallel processing
    train_dataset = train_dataset.batch(batch_size=batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size=batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size=batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def get_random_batch(dataset: tf.data.Dataset) -> tuple:
    # Convert the dataset to a list
    dataset_as_list = list(dataset.as_numpy_iterator())

    # Get a random batch from the dataset
    X, y = random.choice(dataset_as_list)

    return X, y


def plot_dataset_batch_with_bbox(image_batch, bbox_data, width: int, height: int):
    if len(image_batch) != len(bbox_data[0]) or len(image_batch) != len(bbox_data[1]):
        raise ValueError("The number of images and the number of bounding boxes must be the same")

    # For each image in the batch
    for batch_index in range(len(image_batch)):
        image = image_batch[batch_index].copy()
        face_detected = bbox_data[0][batch_index] > 0.9
        coords = bbox_data[1][batch_index]

        # Convert the normalized coordinates to the original image size
        coords = tuple(np.multiply(coords, [width, height, width, height]).astype(int))

        # Plot the image with the bounding box
        plot_image_with_rectangle(image=image,
                                  point1=(coords[0], coords[1]),
                                  point2=(coords[2], coords[3]),
                                  thickness=2 if face_detected else 0)

        # Wait for a short amount of time (if the batch size is too big, the plot operation could generate some errors)
        time.sleep(0.1)


if __name__ == '__main__':
    # If no GPU is available, exit the program
    if not gpu_check():
        print('No GPU available')
        exit(-1)

    # Apply GPU optimizations (memory growth)
    apply_gpu_optimizations()

    # Read the settings file and get the resize width and height to use for the model training
    with open(os.path.join(os.getcwd(), 'settings.json'), 'r') as f:
        settings = json.load(f)
        train_resize_width: int = settings['model_training']['resize_width']
        train_resize_height: int = settings['model_training']['resize_height']

    # Read the augmented images, resize them and normalize the RGB values of the pixels between 0 and 1
    augmented_images_pattern: str = os.path.join(os.getcwd(), 'data', 'augmented', 'images', '*.jpg')
    image_dataset: tf.data.Dataset = tf.data.Dataset.list_files(augmented_images_pattern, shuffle=False)
    image_dataset = image_dataset.map(lambda file_path: image_to_tensor(image_path=file_path))
    image_dataset = image_dataset.map(lambda tensor_image: tf.image.resize(images=tensor_image,
                                                                           size=(train_resize_height,
                                                                                 train_resize_width)))
    image_dataset = image_dataset.map(lambda tensor_image: tensor_image / 255)

    # Read the augmented labels and extract its information
    labels_pattern: str = os.path.join(os.getcwd(), 'data', 'augmented', 'labels', '*.json')
    label_dataset: tf.data.Dataset = tf.data.Dataset.list_files(labels_pattern, shuffle=False)
    label_dataset = label_dataset.map(lambda file_path: tf.py_function(func=get_label_info,
                                                                       inp=[file_path],
                                                                       Tout=[tf.uint32, tf.float32]))
    # Join the datasets into a single one
    merged_dataset: tf.data.Dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    # Split the merged dataset into training, validation and test batched datasets
    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset=merged_dataset, batch_size=32)

    # Plot the bounding boxes on a random batch of images to check if the data are correct
    # X, y_true = get_random_batch(train_dataset)
    # plot_dataset_batch_with_bbox(image_data=X, bbox_data=y_true, width=train_resize_width, height=train_resize_height)

    # Define the optimizer
    optimizer = Adam(learning_rate=10 ** -5,
                     decay=compute_lr_decay(batches_per_epoch=len(train_dataset)))

    # Instantiate the FaceDetector model and compile it with the optimizer and the loss functions
    face_detector = FaceDetectorModel(input_image_width=train_resize_width, input_image_height=train_resize_height)
    face_detector.compile(optimizer=optimizer,
                          classification_loss=BinaryCrossentropy(),
                          localization_loss=localization_loss_func)

    # Define the model file path
    model_file_path = os.path.join(os.getcwd(), 'face_detector.keras')

    # Define the number of epochs
    epoch_number = 30

    # Define checkpoint callbacks to save the best model and early stopping callbacks to avoid overfitting
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_file_path,
        monitor='val_total_loss',
        save_best_only=True,
        verbose=1
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_total_loss',
        patience=int(0.35 * epoch_number),
        min_delta=0.001,
        restore_best_weights=True
    )

    # Train the model
    history = face_detector.fit(train_dataset,
                                epochs=epoch_number,
                                validation_data=val_dataset,
                                callbacks=[checkpoint_callback, early_stopping_callback])

    # Plot the training performance
    plot_training_performance(history=history)

    # Plot the bounding boxes on a random batch of test images to check if the model is working
    X, y_true = get_random_batch(test_dataset)
    y_pred = face_detector.predict(X)
    plot_dataset_batch_with_bbox(image_batch=X, bbox_data=y_pred, width=train_resize_width, height=train_resize_height)

    # Exit the program after input pressed
    input("Press a key to exit...")
    exit(0)
