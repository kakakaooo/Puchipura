from annoy import AnnoyIndex
from tensorflow.keras.preprocessing import image
from IPython.display import Image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import glob

#base_model = VGG16(weights=’imagenet’)

base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
from tensorflow.keras import Model, layers
model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

trained_model = AnnoyIndex(4096)
trained_model.load("ann/result.ann")

glob_dir = "static/親プチ２/*/*.jpg"
files = glob.glob(glob_dir)
d={}
for index, file in enumerate(files):
    d[index]=file

img_path = "デパコス２/333.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
fc2_features = model.predict(x)
result = trained_model.get_nns_by_vector(fc2_features[0], 3, search_k=-1, include_distances=False)
print(d[result[0]])
print(d[result[1]])
print(d[result[2]])
print(result)
# インデックス0付近の10000個のデータを返す。全データがこの値より小さいときは実データ数になるっぽい
#print(trained_model.get_nns_by_item(0, 10000))