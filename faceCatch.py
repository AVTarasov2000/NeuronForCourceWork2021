# plot photo with detected faces using opencv cascade classifier
from cv2.cv2 import imshow
from cv2.cv2 import waitKey
from cv2.cv2 import destroyAllWindows
from cv2.cv2 import CascadeClassifier
from cv2.cv2 import rectangle
from imageio import imread
import cv2.cv2 as cv2
from PIL import Image

import io, base64
import json


name = "yulya_child_9.txt"
count = 2678
with open(f"./datasets/{name}", "r") as data:
    item = data.readline()
    while item:
        tmp = item.replace(" ","").replace("\n","").replace(";","")
        loadedData = json.loads(tmp)
        coded_str = loadedData['photo']
        img = io.BytesIO(base64.b64decode(coded_str.split(',')[1]))

        image = Image.open(io.BytesIO(base64.b64decode(coded_str.split(',')[1])))

        pixels = imread(img)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # load the pre-trained model
        classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

        # perform face detection
        bboxes = classifier.detectMultiScale(pixels, scaleFactor=1.1, minNeighbors=5)

        # print bounding box for each detected face
        for box in bboxes:
            # extract
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # draw a rectangle over the pixels
            rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)
            cropped = image.crop((x, y, x2, y2))
            cropped.save(f'testFaces/images/{count}.jpg')
            with open(f"./testFaces/labels/{count}.txt", "w") as res:
                res.write(json.dumps({'radius': loadedData['radius']}))
            count += 1
            break

        # show the image
        imshow(f'face detection{count}', pixels)
        # keep the window open until we press a key
        waitKey(0)

        # close the window
        # destroyAllWindows()
        item = data.readline()
        # count += 1
        # break
print(count)
