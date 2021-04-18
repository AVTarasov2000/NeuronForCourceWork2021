import io, base64

import tensorflow as tf
from PIL import Image
from cv2.cv2 import CascadeClassifier
from imageio import imread
import cv2.cv2 as cv2
import numpy as np
from skimage import transform
from skimage.color import rgb2gray


model = tf.keras.models.load_model('model_2')
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
SIZE = 62

def predict(image):

    pixels = imread(io.BytesIO(base64.b64decode(image.split(',')[1])))
    img = Image.open(io.BytesIO(base64.b64decode(image.split(',')[1])))
    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    boxes = classifier.detectMultiScale(pixels, scaleFactor=1.1, minNeighbors=5, minSize=(SIZE, SIZE))
    if len(boxes) == 0:
        return 1
    x, y, width, height = boxes[0]
    x2, y2 = x + width, y + height
    cropped = img.crop((x, y, x2, y2))

    image = np.array(tf.keras.preprocessing.image.img_to_array(cropped))
    image = transform.resize(image, (SIZE, SIZE))

    images = rgb2gray(np.array([image]))
    images = np.array([image / np.amax(image) for image in images])
    images = images.reshape(images.shape + (1,))

    return model.predict(np.array(images))[0][0]