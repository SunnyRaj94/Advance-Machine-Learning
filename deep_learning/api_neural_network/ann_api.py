#importing necessarry libraries
from flask import Flask,  jsonify, request
#importing joblib to play with pkl file
import joblib as job
#importing pandas
import pandas as pd

# importing my custom library file methods
import sys
sys.path.append('/home/admin3/ml_with_phoenix/deep_learning/lib_and_pkl_files/')
from ipynb.fs.full.library import *

#seeting name to run this file as flask app
app = Flask(__name__)

def pre_processing(data_set,one_hot_encoder,std_scalar,classifier):
    x_values, y_values, one_hot_encoder, std_scalar = pre_processing_bank_data(data_set,one_hot_encoder,
                                                                               std_scalar)
    prediction = fit_predict(x_values, y_values,classifier)
    result = {}
    result["Actual Value"]=y_values.tolist()[0]
    result["Predicted Value"] = 1 if prediction.tolist()[0][0]==True else 0
    return result

# methods takes a post type argument and returns predicted values of the data obtained
@app.route('/ann_predict',methods=['POST'])
def obtain_prediction():
    # importing all pkl files
    ann_one_hot_encoder = job.load(
        '/home/admin3/ml_with_phoenix/deep_learning/lib_and_pkl_files/ann_one_hot_encoder.pkl')
    ann_std_scalar = job.load('/home/admin3/ml_with_phoenix/deep_learning/lib_and_pkl_files/std_scalar.pkl')
    ann_sequential_classifier = job.load(
        '/home/admin3/ml_with_phoenix/deep_learning/lib_and_pkl_files/ann_sequential_classifier.pkl')
    return pre_processing(pd.DataFrame(request.json , index=[0]),ann_one_hot_encoder,ann_std_scalar,ann_sequential_classifier)


if __name__ == '__main__':
    app.run(debug=True)