#importing necessary libraries
from flask import Flask, jsonify, request
import joblib as job
import pandas as pd

# importing my custom library file methods
import sys
sys.path.append('/home/admin3/ml_with_phoenix/natural_language_processing/pkl_objects_and_lib/')
from ipynb.fs.full.library import *

#loading all pkl objects
porter_stemmer=job.load('/home/admin3/ml_with_phoenix/natural_language_processing/pkl_objects_and_lib/porter_stemmer.pkl')
count_vectorizer=job.load('/home/admin3/ml_with_phoenix/natural_language_processing/pkl_objects_and_lib/count_vectorizer.pkl')
logistic_classifier=job.load('/home/admin3/ml_with_phoenix/natural_language_processing/pkl_objects_and_lib/logistic_classifier.pkl')

app = Flask(__name__)
# a general method which takes raw data and gives the predicted values directly
def pre_processing(data):
    y_values = data['Liked'].values
    data = pre_processing_nltk(data['Review'][0],porter_stemmer)
    cv, x_values = fit_transform_nltk(list(data), count_vectorizer, True)
    prediction = fit_or_predict(x_values, y_values, logistic_classifier)
    result={}
    result["Actual Value : "]=y_values.tolist()[0]
    result['Predicted Value']=prediction.tolist()[0]
    return result

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