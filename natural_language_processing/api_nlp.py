#importing necessarry libraries
#importing flask , jsonify,request

from flask import Flask, jsonify, request
#importing joblib to play with pkl file
import joblib as job

#importing pandas
import pandas as pd

# importing my custom library file methods
import sys
sys.path.append('/home/admin3/ml_with_phoenix/natural_language_processing/pkl_objects_and_lib/')
from ipynb.fs.full.library import *
# #importing numpy
import numpy as np

#loading all pkl objects
porter_stemmer=job.load('/home/admin3/ml_with_phoenix/natural_language_processing/pkl_objects_and_lib/porter_stemmer.pkl')
count_vectorizer=job.load('/home/admin3/ml_with_phoenix/natural_language_processing/pkl_objects_and_lib/count_vectorizer.pkl')
logistic_classifier=job.load('/home/admin3/ml_with_phoenix/natural_language_processing/pkl_objects_and_lib/logistic_classifier.pkl')



app = Flask(__name__)
# a general method which takes raw data and gives the predicted values directly
def pre_processing(data):
    return "hii"

# method takes post type of request
# takes raw data and returns predicted value based on model
@app.route('/predict',methods=['POST'])
def predict_by_natural_language_processing():
    raw_data = request.json
    prediction = pre_processing(pd.DataFrame(raw_data,index=[0]))
    return prediction




#seeting name to run this file as flask app
if __name__ == '__main__':
    app.run(debug=True)