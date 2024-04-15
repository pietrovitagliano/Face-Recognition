import cv2
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt


def gpu_check() -> bool:
    # Get the list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Return if there are available GPUs
    return len(gpus) > 0


def apply_gpu_optimizations():
    # Get the list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Avoid Out of Memory errors by enabling, for any gpu,
    # the GPU Memory Consumption Growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def image_to_tensor(image_path) -> tf.Tensor:
    byte_img = tf.io.read_file(filename=image_path)
    tensor_img = tf.io.decode_jpeg(byte_img)

    return tensor_img


def plot_image_with_rectangle(image: np.ndarray, point1: tuple[int, int], point2: tuple[int, int], thickness: int = 2):
    # Create a new plot
    plt.figure(figsize=(20, 20))

    # Draw the rectangle on the image
    if thickness != 0:
        cv2.rectangle(image, point1, point2, (255, 0, 0), thickness)

    # Plot the augmented image
    plt.imshow(image)
    plt.show()
