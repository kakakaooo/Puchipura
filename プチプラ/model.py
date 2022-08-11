import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

#base_model = VGG16(weights=’imagenet’)

base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet')

base_model.summary()


from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# img_path = "親プチ２/デパ２/shopping (1).jpg"

# img = image.load_img(img_path, target_size=(224, 224))

# input = image.img_to_array(img)

# result = base_model.predict(np.array([input]))

# print("array", result)

# print("length:", len(result[0]))


from tensorflow.keras import Model, layers

model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

# print(model.input)

# print(model.output)


# img = image.load_img(img_path, target_size=(224, 224))

# input = image.img_to_array(img)

# result = model.predict(np.array([input]))

# print("array", result)

# print("len: ", len(result[0]))


from annoy import AnnoyIndex

dim = 4096

annoy_model = AnnoyIndex(dim)

import glob

numimg=0

glob_dir = "static/親プチ２/*/*.jpg"
files = glob.glob(glob_dir)

d={}
for index, file in enumerate(files):
    d[index]=file

    img_path = file

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    fc2_features = model.predict(x)

    annoy_model.add_item(numimg, fc2_features[0])

    numimg += 1


print("num files=" + str(numimg))

import os
annoy_model.build(numimg)

annoy_model.save("ann/result.ann")


# 保存結果を読み込む場合はこれ

#annoy_model.unload()

#trained_model.load(save_path)