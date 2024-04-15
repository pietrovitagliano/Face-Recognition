import tensorflow as tf

import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D
from keras.applications import VGG16
from matplotlib import pyplot as plt


def compute_lr_decay(batches_per_epoch: int) -> float:
    return (1 / 0.75 - 1) / batches_per_epoch


def localization_loss_func(y_true, y_pred):
    # Extract the coordinates from the true and predicted values
    x1_true, y1_true, x2_true, y2_true = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    x1_pred, y1_pred, x2_pred, y2_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

    # Compute the loss for the first couple of coordinate
    delta_first_coord = tf.reduce_sum(tf.square(x1_true - x1_pred) + tf.square(y1_true - y1_pred))

    # Compute true width and height
    w_true = x2_true - x1_true
    h_true = y2_true - y1_true

    # Compute predicted width and height
    w_pred = x2_pred - x1_pred
    h_pred = y2_pred - y1_pred

    # Compute the loss for the width and height
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    # Return the sum of the two losses
    return delta_first_coord + delta_size


def plot_training_performance(history: keras.callbacks.History):
    _, ax = plt.subplots(ncols=3, figsize=(20, 5))

    ax[0].plot(history.history['total_loss'], color='teal', label='train total loss')
    ax[0].plot(history.history['val_total_loss'], color='orange', label='val total loss')
    ax[0].title.set_text('Total Loss')
    ax[0].legend()

    ax[1].plot(history.history['classification_loss'], color='teal', label='train classification loss')
    ax[1].plot(history.history['val_classification_loss'], color='orange', label='val classification loss')
    ax[1].title.set_text('Classification Loss')
    ax[1].legend()

    ax[2].plot(history.history['localization_loss'], color='teal', label='train localization loss')
    ax[2].plot(history.history['val_localization_loss'], color='orange', label='val localization loss')
    ax[2].title.set_text('Localization Loss')
    ax[2].legend()

    plt.show()


class FaceDetectorModel(Model):
    def __init__(self, input_image_width, input_image_height, **kwargs):
        super().__init__(**kwargs)

        # Define the input Layer
        input_layer = Input(shape=(input_image_height, input_image_width, 3))

        # Instantiate the VGG16 model without the top layer
        vgg = VGG16(include_top=False)(input_layer)

        # First top layer: Classification model
        classificationModel = Sequential([GlobalMaxPooling2D(),
                                          Dense(4096, activation=keras.activations.relu),
                                          Dropout(0.25),
                                          Dense(1, activation=keras.activations.sigmoid)])(vgg)

        # Second top layer: Localization model (Regression Problem)
        localizationModel = Sequential([GlobalMaxPooling2D(),
                                        Dense(4096, activation=keras.activations.relu),
                                        Dropout(0.25),
                                        Dense(4, activation=keras.activations.sigmoid)])(vgg)

        # Create the model
        self.model = Model(inputs=input_layer, outputs=[classificationModel, localizationModel])

        # Define other attributes
        self.optimizer = None
        self.classification_loss = None
        self.localization_loss = None

    def compile(self, optimizer, classification_loss, localization_loss, **kwargs):
        # Compile the model without passing optimizer and loss function
        super().compile(**kwargs)

        # Define optimizer and loss functions
        self.optimizer = optimizer
        self.classification_loss = classification_loss
        self.localization_loss = localization_loss

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

    def train_step(self, batch):
        # Extract the images and the labels from the batch
        X, y = batch

        # Extract the true classes and coordinates
        batch_classes_true, batch_coords_true = y

        # Forward pass
        with tf.GradientTape() as tape:
            # Get the predictions in training mode
            batch_classes_pred, batch_coords_pred = self.model(X, training=True)

            # Compute the classification and localization losses, then the total loss
            classification_loss = self.classification_loss(batch_classes_true, batch_classes_pred)
            localization_loss = self.localization_loss(batch_coords_true, batch_coords_pred)

            # Total loss needs to take into account both classification and localization losses,
            # but classification_loss has a lower weight in the total loss
            total_loss = 0.5 * classification_loss + localization_loss

            # Compute the gradients based on the total loss
            grad = tape.gradient(total_loss, self.model.trainable_variables)

        # Apply the gradients to adjust the model's weights
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss,
                "classification_loss": classification_loss,
                "localization_loss": localization_loss}

    def test_step(self, batch):
        # Extract the images and the labels from the batch
        X, y = batch

        # Extract the true classes and coordinates
        batch_classes_true, batch_coords_true = y

        # Get the predictions in test mode
        batch_classes_pred, batch_coords_pred = self.model(X, training=False)

        # Compute the classification and localization losses, then the total loss
        classification_loss = self.classification_loss(batch_classes_true, batch_classes_pred)
        localization_loss = self.localization_loss(batch_coords_true, batch_coords_pred)
        total_loss = 0.5 * classification_loss + localization_loss

        return {"total_loss": total_loss,
                "classification_loss": classification_loss,
                "localization_loss": localization_loss}

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        self.model.save(filepath, overwrite, save_format, **kwargs)
