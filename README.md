# Face-Recognition
This project utilizes Python and TensorFlow to perform face recognition using the images captured from a PC camera. It employs the tool labelme for data preparation, enabling users to define facial boundaries within images. Data augmentation is applied by incorporating various random transformations. The model is constructed by building upon the pre-existing VGG16 model, which is designed for image recognition, and adding additional layers. This final model facilitates face detection within images. Additionally, it can process video frames, enabling face recognition in real-time video streams.

## Project Setup
- ### Install Requirements
  Ensure you have installed the libraries listed in the file requirements.txt

- ### Data Preparation
  Utilize Labelme to annotate images, defining facial boundaries for each individual.
  Organize annotated images into appropriate directories for training and validation.

## Model Training
- ### Data Augmentation
  The code implements data augmentation techniques to improve the training dataset and enhance model robustness.
  This is achieved by applying random transformations such as flipping, rotating, and scaling images to the existing data.

- ### Model Construction
  The pre-trained VGG16 model serves as the foundation for the face recognition model.
  Additional convolutional and fully connected layers are then added on top of VGG16.
  These additional layers are specifically tailored for the task of face recognition.
  Finally, the model is compiled with an appropriate optimizer and loss function suitable for the problem.

- ### Model Training
  The augmented training dataset is used to train the model.
  The code monitors the training progress to track how well the model is learning.
  A validation dataset is also used to evaluate the model's performance during training.
  This helps to identify overfitting and ensure the model generalizes well to unseen data.

## Model Usage
The model can be used to recognize face in images from the webcam or use pre-existing images.
