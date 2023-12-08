#here we have created a helper file for preprocessing images and savinffg model 
from PIL import Image
import pickle
import os
import numpy as np
import keras 
from keras.models import load_model
from keras.utils import img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join('d:\YG\Python','model1.h5')
print(model_path)

if os.path.exists(model_path):
    print("Model file exists.")
    
else:
    print("Model file does not exist. Check the file path.")
model = load_model(model_path)
#print(os.path(model))
#preprocessing the image 

def preprocess_img(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape

def predict_result(predict):
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)







































'''

img=cv2.imread('data/pred/pred1.jpg')
img=Image.fromarray(img)
img=img.resize((64,64))
img=np.array(img)
img=np.expand_dims(img,axis=0)
print(np.argmax(model.predict(img),axis=1))
# print(model.accuracy())

# print(img)
# print(model.predict(img))
'''