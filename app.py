# importing necessary libraries and functions

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App

model = pickle.load(open('model.pkl', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    k = request.form.values()
    # retrieving values from form
    init_features = [float(x) for x in k]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction
    predict=str(prediction).strip('[]')
    return render_template('index1.html', prediction_text='{}'.format(predict)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)