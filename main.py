import os 
from flask import Flask,render_template,request
from PIL import Image
import numpy as np

from model_p import predict_result,preprocess_img



app = Flask(__name__,template_folder='templates')



#home route
@app.route("/")
def main():
    return render_template('index copy.html')

#predict route 
@app.route('/predict',methods = ['POST'])

def predict():
    try :
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            predict = predict_result(img)
            return render_template("result.html", predictions=str(predict))

    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)




if __name__ == '__main__':
#this runs the python file 
    app.run()