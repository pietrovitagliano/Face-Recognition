import os
import time
import uuid

import cv2


def capture_images(images_dir_path: str, image_number: int = 30, time_between_images: float = 0.5):
    if not os.path.exists(images_dir_path):
        os.makedirs(images_dir_path)

    video_capture = cv2.VideoCapture(0)
    for img in range(image_number):
        print('Collecting image {}'.format(img))
        _, frame = video_capture.read()
        img_name = os.path.join(images_dir_path, f'{str(uuid.uuid1())}.jpg')

        cv2.imshow('Image Acquisition', frame)
        cv2.imwrite(img_name, frame)
        time.sleep(time_between_images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Define the path to save the images and the number of images to capture
    images_dir_path = os.path.join(os.getcwd(), 'data', 'images')
    image_number = 10

    # Capture the images
    capture_images(images_dir_path=images_dir_path, image_number=image_number, time_between_images=2)
    exit(0)
