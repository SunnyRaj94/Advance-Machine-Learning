#importing necessarry libraries
#importing flask , jsonify,request

from flask import Flask,  jsonify, request
#importing joblib to play with pkl file
import joblib as job

#importing pandas
import pandas as pd

# importing my custom library file methods
import sys
sys.path.append('/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib')
from ipynb.fs.full.library import *
#importing numpy
import numpy as np

#seeting name to run this file as flask app
app = Flask(__name__)

# loading all pickeled file objects at once
pca_scalar = job.load("/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib/pca_scalar_wine.pkl")
pca = job.load("/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib/pca_wine.pkl")
pca_logistic_classifier = job.load("/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib/pca_classifier_wine.pkl")
lda_scalar =job.load("/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib/lda_scalar_wine.pkl")
lda_logistic_classifier =job.load("/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib/lda_classifier_wine.pkl")
lda = job.load("/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib/lda_wine.pkl")
kernel_pca = job.load("/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib/k_pca_ad.pkl")
kernel_svc_classifier = job.load("/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib/k_pca_classifier_wine.pkl")
kernel_scalar = job.load("/home/admin3/ml_with_phoenix/dimensionality_reduction_techniques/pkl_objects_and_lib/k_pca_scalar_wine.pkl")


# a general method which takes raw data and gives the predicted values directly
def pre_processing_all(data,scalar,drc,classifier,flag = True):
    if flag:
        x_values, y_values = pre_processing(pd.DataFrame(data,index=[0]))
    else:
        x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    x_values = scalar.transform(x_values)

    x_values=drc.transform(x_values)

    test_prediction = fit_or_predict(x_values, y_values, classifier).tolist()
    result ={}
    result["Actual value "]=y_values.tolist()[0]
    result["Predicted value"]=test_prediction[0]

    return result

# method takes post type of request
# takes raw data and returns predicted value based on model
@app.route('/pca',methods=['POST'])
def predict_by_applying_pca():
    raw_data = request.json
    prediction = pre_processing_all(raw_data,pca_scalar,pca,pca_logistic_classifier)
    return prediction

# method takes post type of request
# takes raw data and returns predicted value based on model
@app.route('/lda',methods=['POST'])
def predict_by_applying_lda():
    raw_data = request.json
    prediction = pre_processing_all(raw_data,lda_scalar,lda,lda_logistic_classifier)
    return prediction

# method takes post type of request
# takes raw data and returns predicted value based on model
@app.route('/lda',methods=['POST'])
def predict_by_applying_kernel_pca():
    raw_data = request.json
    prediction = pre_processing_all(raw_data,kernel_scalar,kernel_pca,kernel_svc_classifier,False)
    return prediction


if __name__ == '__main__':
    app.run(debug=True)