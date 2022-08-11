import os
import re
from flask import (
     Flask, 
     request, 
     render_template)
#画像のアップロード先のディレクトリ
UPLOAD_FOLDER='./static/make_image'

#FlaskでAPIを書くときのおまじない
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_user_files():
    if request.method == 'POST':
        upload_file = request.files['upload_file']
        img_path = os.path.join(UPLOAD_FOLDER,upload_file.filename)
        upload_file.save(img_path)

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
        #img_path = img_pat
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fc2_features = model.predict(x)
        result = trained_model.get_nns_by_vector(fc2_features[0], 3, search_k=-1, include_distances=False)
        #print(d[result[0]])
        #print(d[result[1]])
        #print(d[result[2]])
        #print(resul
        
        dir_pathname=os.path.dirname(d[result[0]].replace("\\","/",2))
        Result_img_path=dir_pathname +"/*200.jpg"
        result_img_path=glob.glob(Result_img_path)
        return render_template('result.html', img_path=img_path,result_img_path=result_img_path[0])
if __name__ == '__main__':
  app.run(debug=True)