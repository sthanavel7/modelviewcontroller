import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

x,y = fetch_openml("mnist_784", version = 1, return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 2500, train_size = 7500, random_state = 9)
x_train_scale = x_train / 255.0
x_test_scale = x_test / 255.0

classifier = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(x_train_scale, y_train)

def get_pred(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert("l")

    image_bw_resize = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20

    min_pixel = np.percentile(image_bw_resize, pixel_filter)

    image_bw_resize_scale = np.clip(image_bw_resize - min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resize)
    image_bw_resize_scale = np.asarray(image_bw_resize_scale) / max_pixel
    test_sample = np.array(image_bw_resize_scale).reshape(1,784)
    test_prediction = classifier.predict(test_sample)

    return test_prediction[0]