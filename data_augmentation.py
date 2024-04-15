import os
import json
import cv2
import tensorflow as tf
import numpy as np
import albumentations as alb

from matplotlib import pyplot as plt

from utils import image_to_tensor


def plot_images(image_dataset: tf.data.Dataset, batch_size: int = 4):
    # Apply the image_to_tensor function to the dataset,
    # in order to convert the image names to tensors
    image_np_dataset = image_dataset.map(lambda file_path: image_to_tensor(file_path))

    # Define an iterator to plot 4 images at a time
    image_iterator = image_np_dataset.batch(batch_size).as_numpy_iterator()

    rows_to_plot: int = batch_size // 2
    cols_to_plot: int = batch_size // 2

    # Plot the images into a grid of 2x2 per batch
    while True:
        try:
            # Get the next batch of images
            image_batch = image_iterator.next()

            # Create a new plot
            _, plot = plt.subplots(nrows=rows_to_plot, ncols=cols_to_plot, figsize=(20, 20))

            # Assign the images to the plot
            for i in range(rows_to_plot):
                for j in range(cols_to_plot):
                    plot[i, j].imshow(image_batch[i + j])

            # Show the plot
            plt.show()
        except StopIteration:
            break


def get_label_path_from_image_path(image_path: str) -> str:
    # Get the path of the labels
    label_dir_path = os.path.join(os.getcwd(), 'data', 'labels')

    # Extract name with no extension from the image path
    image_name: str = os.path.basename(image_path).split(".")[0]

    # Get the path of the label
    label_path: str = os.path.join(label_dir_path, f"{image_name}.json")

    return label_path


def augment_image(image_np: np.ndarray, face_coords: list | None, image_path: str, augmented_images_dir_path: str,
                  augmented_labels_dir_path: str, augmentation_func_face_present: callable,
                  augmentation_func_face_absent: callable, augmentation_times: int,
                  min_face_area_percentage: float, min_face_area_delta: float):
    if not 0 < min_face_area_percentage <= 1:
        raise ValueError("The min_aug_face_area_percentage must be between 0 and 1")

    if min_face_area_delta >= min_face_area_percentage:
        raise ValueError("The min_aug_delta must be less than the min_aug_face_area_percentage")

    # Compute the area of the box if the face is present
    face_box_area = 0
    if face_coords is not None:
        face_box_width = face_coords[2] - face_coords[0]
        face_box_height = face_coords[3] - face_coords[1]
        face_box_area = face_box_width * face_box_height

    # Apply the augmentation more times in order to get more augmented images for each original image.
    # This is necessary in order to have enough data for training
    for i in range(augmentation_times):
        # Augment the image
        if face_box_area == 0:
            # If there are no coordinates, it means that there is no face in the image
            augmented_image: dict = augmentation_func_face_absent(image=image_np)
        else:
            # Repeat the augmentation until the face box area is acceptable
            # (greater than or equal to the minimum percentage),
            # or until the face box area is too small or absent
            while True:
                # If there are coordinates, it means that there is a face in the image
                augmented_image: dict = augmentation_func_face_present(image=image_np,
                                                                       bboxes=[face_coords],
                                                                       class_labels=['face'])
                # If a face is not present in the augmented image, exit the loop
                if "bboxes" not in augmented_image or len(augmented_image["bboxes"]) == 0:
                    break
                else:
                    # Get the coordinates of the augmented face
                    aug_face_coords = augmented_image["bboxes"][0]
                    aug_face_box_width = aug_face_coords[2] - aug_face_coords[0]
                    aug_face_box_height = aug_face_coords[3] - aug_face_coords[1]
                    aug_face_box_area = aug_face_box_width * aug_face_box_height
                    aug_face_box_area_percentage = aug_face_box_area / face_box_area

                    # If the percentage area of the augmented face is greater than or equal to the minimum percentage,
                    # exit the loop, since the augmented crop is acceptable
                    if aug_face_box_area_percentage >= min_face_area_percentage:
                        break

                    # If the percentage area of the augmented face is greater than or equal to
                    # the minimum percentage less the delta, repeat the loop, in order to get a new crop
                    if aug_face_box_area_percentage >= min_face_area_percentage - min_face_area_delta:
                        continue
                    # In this case, the percentage area of the augmented face is too small,
                    # thus the image is considered witoout a face
                    else:
                        # Remove the bboxes from the augmented image,
                        # because the face box is not acceptable
                        augmented_image.pop("bboxes")

        # Get the file name without extension
        augmented_image_basename: str = os.path.basename(image_path).split(".")[0]
        augmented_image_basename = f"{augmented_image_basename}_{i}"

        # Save the augmented image
        augmented_image_path: str = os.path.join(augmented_images_dir_path, f'{augmented_image_basename}.jpg')
        cv2.imwrite(augmented_image_path, augmented_image['image'])

        # Save the augmented label
        augmented_label_path: str = os.path.join(augmented_labels_dir_path, f'{augmented_image_basename}.json')
        with open(augmented_label_path, 'w') as f:
            if "bboxes" in augmented_image and len(augmented_image["bboxes"]) > 0:
                augmented_label = {
                    "bbox": augmented_image["bboxes"][0],
                    "class": 1
                }
            else:
                augmented_label = {
                    "bbox": [0] * 4,
                    "class": 0
                }

            json.dump(augmented_label, f)


def export_augmented_data(image_dataset: tf.data.Dataset, augmentation_times: int = 70,
                          min_face_area_percentage: float = 0.6, min_face_area_delta: float = 0.3):
    # Create the directory to save the augmented images if it does not exist
    augmented_images_dir_path = os.path.join(os.getcwd(), 'data', 'augmented', 'images')
    if not os.path.exists(augmented_images_dir_path):
        os.makedirs(augmented_images_dir_path)

    # Create the directory to save the augmented labels if it does not exist
    augmented_labels_dir_path = os.path.join(os.getcwd(), 'data', 'augmented', 'labels')
    if not os.path.exists(augmented_labels_dir_path):
        os.makedirs(augmented_labels_dir_path)

    # Read the settings file and get the crop width and height
    with open(os.path.join(os.getcwd(), 'settings.json'), 'r') as f:
        settings = json.load(f)
        crop_width: int = settings['data_augmentation']['crop_width']
        crop_height: int = settings['data_augmentation']['crop_height']

    # Setup Albumentations Transform Pipeline
    augmentation_transforms = [alb.RandomCrop(width=crop_width, height=crop_height),
                               alb.HorizontalFlip(p=0.5),
                               alb.RandomBrightnessContrast(p=0.2),
                               alb.RandomGamma(p=0.2),
                               alb.RGBShift(p=0.2),
                               alb.VerticalFlip(p=0.5)]

    # Define the augmentation functions
    augmentation_func_face_absent: callable = alb.Compose(transforms=augmentation_transforms)
    augmentation_func_face_present: callable = alb.Compose(transforms=augmentation_transforms,
                                                           bbox_params=alb.BboxParams(format='albumentations',
                                                                                      label_fields=['class_labels']))

    for image_path_bytes_format in image_dataset.as_numpy_iterator():
        # Convert the image format from bytes to string
        image_path: str = image_path_bytes_format.decode('utf-8')

        # Convert the image to a numpy array
        image_tensor: tf.Tensor = image_to_tensor(image_path)
        image_np = np.array(image_tensor)

        # Get the label path
        label_path = get_label_path_from_image_path(image_path)

        # Define the coordinates of the bounding box
        coords = None

        # Check if the label exists
        if os.path.exists(label_path):
            # Extract the coordinates from the label and normalize them to the image size
            with open(label_path, 'r') as label_file:
                label_json = json.load(label_file)

            x1 = label_json['shapes'][0]['points'][0][0]
            y1 = label_json['shapes'][0]['points'][0][1]
            x2 = label_json['shapes'][0]['points'][1][0]
            y2 = label_json['shapes'][0]['points'][1][1]

            # Create the coordinates list
            # NOTE: the label_json has been created with the labelme tool: the min and max functions are needed,
            # in order to avoid errors, if the points are taken in different order
            coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

            # Normalize the coordinates to the image resolution
            image_width = image_np.shape[1]
            image_height = image_np.shape[0]
            coords = list(np.divide(coords, [image_width, image_height, image_width, image_height]))

        # Apply augmentation
        augment_image(image_np=image_np, face_coords=coords, image_path=image_path,
                      augmented_images_dir_path=augmented_images_dir_path,
                      augmented_labels_dir_path=augmented_labels_dir_path,
                      augmentation_func_face_present=augmentation_func_face_present,
                      augmentation_func_face_absent=augmentation_func_face_absent,
                      augmentation_times=augmentation_times,
                      min_face_area_percentage=min_face_area_percentage,
                      min_face_area_delta=min_face_area_delta)

    print('Data augmentation completed!')


if __name__ == '__main__':
    # Get the pattern to read the images
    images_pattern: str = os.path.join(os.getcwd(), 'data', 'images', '*.jpg')

    # Read the images and convert them to numpy arrays
    image_dataset: tf.data.Dataset = tf.data.Dataset.list_files(images_pattern, shuffle=False)

    # Plot the images
    # plot_images(image_dataset=image_dataset)

    # Apply Image Augmentation and export the augmented images and labels
    # NOTE: This process requires to use the LabelMe tool,
    # in order to get the json file created with it, for each image
    export_augmented_data(image_dataset=image_dataset)
    exit(0)
