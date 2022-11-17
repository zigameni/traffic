import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    # getting the root directory of all the data.
    root = os.path.join(os.getcwd(), data_dir)

    # getting the list of all the subdirs within the data folder,
    # each is a label for our model
    subdirs = sorted([int(i) for i in os.listdir(root)])

    for subdir in subdirs:
        for filename in os.listdir(os.path.join(root, str(subdir))):
            # loading the ima file using the opencv module
            # image path, we dont need to seperate this, but it has batter readability
            imagepath = os.path.join(root, str(subdir), filename)
            img = cv2.imread(imagepath)

            # resizing the image matrix to make it suitable for reading into the nural net
            resized_img = cv2.resize(
                img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

            # store image into list.
            images.append(resized_img)

            # each subdir is the name of the sign
            labels.append(subdir)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([
        # Convolutional layer. Learns 32 filters using 3x3 kernel
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # add droupout to remove 20% random neurons from the net on every itration
        tf.keras.layers.Dropout(0.2),

        # Convolutional layer, Learns 64 filters using 3x3 kernel
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

        # Max-pooling layer, using 2x2
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # adding dropout to remove 20% neurons from the net by iteration
        tf.keras.layers.Dropout(0.2),

        # flatten units
        tf.keras.layers.Flatten(),

        # adding hidden layer with 128 neurons
        tf.keras.layers.Dense(128, activation='relu'),

        # to avoid overfitting we add anouth drop out
        tf.keras.layers.Dropout(0.2),

        # Adding an output layer with N neurons for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')

    ])

    # traning the neural net
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    main()
