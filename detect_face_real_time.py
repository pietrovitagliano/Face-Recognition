import json
import os
import cv2
import keras
import numpy as np
import tensorflow as tf

from keras import Model


def detect_face(frame: cv2.Mat | np.ndarray, face_detector: Model) -> tuple[np.ndarray, np.ndarray]:
    global crop_width, crop_height, training_resize_width, training_resize_height

    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Crop the frame to the augmented size
    frame = cv2.resize(frame, (crop_width, crop_height))

    # Convert the frame to a tensor resizing it to the training size
    frame = tf.image.resize(frame, (training_resize_height, training_resize_width))

    # Normalize the RGB values of the pixels between 0 and 1
    frame = frame / 255

    # Add a batch dimension
    frame = tf.expand_dims(frame, axis=0)

    # Make the prediction
    face_detected, coords = face_detector.predict(frame)

    # The results are in a batch of one, so get the first result for each element of the tuple
    return face_detected[0], coords[0]


def draw_rectangles_around_face(frame: cv2.Mat | np.ndarray, point1: tuple[int, int], point2: tuple[int, int]):
    # Draw the face rectangle
    cv2.rectangle(frame, point1, point2, (255, 0, 0), 2)

    # Draw the label rectangle
    cv2.rectangle(frame,
                  tuple(np.add(point1, [-2, -40])),
                  tuple(np.add(point1, [80, 0])),
                  (255, 0, 0), -1)

    # Draw the label text
    cv2.putText(frame,
                'Face',
                tuple(np.add(point1, [0, -10])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA)


if __name__ == '__main__':
    # Define the face detection threshold
    face_detection_threshold = 0.8

    # Define the model file path
    model_file_path = os.path.join(os.getcwd(), 'face_detector.keras')

    # Check if the model file exists
    if not os.path.exists(model_file_path):
        print("The model file does not exist: create one first.")
        exit(-1)

    # Load the model
    face_detector = keras.models.load_model(model_file_path)

    # Read the settings file and get the crop width and height and the training resize width and height
    with open(os.path.join(os.getcwd(), 'settings.json'), 'r') as f:
        settings = json.load(f)

        crop_width: int = settings['data_augmentation']['crop_width']
        crop_height: int = settings['data_augmentation']['crop_height']
        training_resize_width: int = settings['model_training']['resize_width']
        training_resize_height: int = settings['model_training']['resize_height']

    # Initialize the video capture
    video_capture = cv2.VideoCapture(0)

    while video_capture.isOpened():
        _, frame = video_capture.read()

        face_detected, coords = detect_face(frame=frame, face_detector=face_detector)
        if face_detected >= face_detection_threshold:
            # Get frame resolution
            frame_width: int = frame.shape[1]
            frame_height: int = frame.shape[0]

            # Get the coordinates adapted for the original frame size
            x1, y1, x2, y2 = np.multiply(coords, [frame_width,
                                                  frame_height,
                                                  frame_width,
                                                  frame_height]).astype(int)

            # Draw the rectangle. with opposite corners at (x1, y1) and (x2, y2), around the face
            draw_rectangles_around_face(frame=frame, point1=(x1, y1), point2=(x2, y2))

        cv2.imshow('Face Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    exit(0)
