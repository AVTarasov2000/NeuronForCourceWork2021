from datasets.atentionDataset import atentionDataset
import numpy as np
import tensorflow as tf
import json
from skimage import transform
from skimage.color import rgb2gray
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt

SIZE = 224
IM_COUNT = 2677


def create_model(size):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            kernel_size=5,
            filters=20,
            activation='relu',
            input_shape=(size, size, 1)
    ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2),
            strides=(2,2)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='MSE')
    return model


def resize_img(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (SIZE, SIZE))
    img = img/SIZE
    return img, label


def loadData(img_path, data_path):
    images = []
    labels = []
    res = []
    for i in range(IM_COUNT):
        image = tf.keras.preprocessing.image.load_img(f'{img_path}/{i}.jpg')
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        images.append(input_arr)
        with open(f'{data_path}/{i}.txt') as data:
            j = json.loads(data.read())
            labels.append(j['radius'])
            res.append([input_arr, j['radius']])

    return np.array(images), np.array(labels)


images, labels = loadData("testFaces/images",'testFaces/labels')

max_1 = len(max(images, key=lambda x: len(x)))
max_2 = len(max(images, key=lambda x: len(x[0]))[0])
max_3 = len(max(images, key=lambda x: len(x[0][0]))[0][0])

min_1 = len(min(images, key=lambda x: len(x)))
min_2 = len(min(images, key=lambda x: len(x[0]))[0])
min_3 = len(min(images, key=lambda x: len(x[0][0]))[0][0])

print(f"min_x:{min_1}, min_y{min_2}, min_r{min_3}")
print(f"min_x:{max_1}, min_y{max_2}, min_r{max_3}")

images28 = []
for image in images:
    if len(image) > min_1 and len(image[0]) > min_2:
        images28.append(transform.resize(image, (min_1, min_2)))
        # images28.append(image)
    else:
        images28.append(image)
print(len(images28))
images28 = np.array(images28)
images28 = rgb2gray(images28)
images28 = np.array([image/np.amax(image) for image in images28])


max_label = max(labels)
labels = np.array([i/max_label for i in labels])

for ind,i in enumerate(images28):
    print(f"{ind}:{len(i)}, {len(i[0])}")

# Задание (случайных) номеров изображений, которые вы хотите вывести

traffic_signs = [300, 550, 850, 1200]

# Заполнение графиков изображениями и вывод размеров

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray") #/np.amax(images28[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)

plt.show()

images28 = images28.reshape(images28.shape + (1,))
check = images28[0]
train_images = []
train_labels = []

validation_images = []
validation_labels = []

for i in range(len(images28)):
    if i % 100 == 0:
        validation_images.append(images28[i])
        validation_labels.append(labels[i])
        # train_images.append(images28[i])
        # train_labels.append(labels[i])
    else:
        train_images.append(images28[i])
        train_labels.append(labels[i])

train_images = np.array(train_images)
train_labels = np.array(train_labels)

validation_labels = np.array(validation_labels)
validation_images = np.array(validation_images)

model = create_model(min_1)
# model = tf.keras.models.load_model('model')

model.fit(train_images,
          train_labels,
          batch_size=64,
          epochs=25,
          shuffle=True,
          validation_data=(validation_images, validation_labels)
          )

res = model.predict(validation_images)
for ind, i in enumerate(res):
    print(i)
    plt.axis('off')
    plt.title(f"{i} = {validation_labels[ind]}")
    plt.imshow(validation_images[ind], cmap="gray") #/np.amax(images28[traffic_signs[i]])
    # plt.subplots_adjust(wspace=0.5)
    plt.show()

model.save("model_2")


