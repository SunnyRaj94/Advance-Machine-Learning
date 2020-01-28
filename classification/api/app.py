#importing necessarry libraries
#importing flask , jsonify,request

from flask import Flask,  jsonify, request
#importing joblib to play with pkl file
import joblib as job

#importing pandas
import pandas as pd

# importing my custom library file methods
import sys
sys.path.append('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/')
from ipynb.fs.full.my_custom_library import *
#importing numpy
import numpy as np

#seeting name to run this file as flask app
app = Flask(__name__)

@app.route('/',methods=['GET'])
def demo():
    result = "<html> <body>  <h1> Hello There !!!!!!</hi>   </body>   </html>"
    return result

#method takes a post type request
#returns predicted output based on algorithm
#in this method we are using logistic regression
@app.route('/ad/log',methods=['POST'])
def predict_logistic_ad():

    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    x_values,y_values            =  pre_processing_ad(pd.DataFrame(data,index=[0]))

    # loading standard scalar object through pickel file
    scalar                       =  job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/scalar.pkl')

    # obtaining scaled values after scaling based on scalar object
    x_values                     =  scalar.transform(x_values)

    # getting classifier object from pkl file
    classifier                   =  job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/logical_classifier_ad.pkl')

    # using that classifier object to obtain predicted output
    predicted                    =  fit_or_predict(x_values,y_values,classifier).tolist()

    # making an dictionary to store result
    result                       =  {}

    # saving actual and predicted value into dictionary
    result["actual value : "]    =  y_values.tolist()[0][0]
    result['predicted value']    =  predicted[0]
    # returning result
    return jsonify(result)


# method takes a post type request
# returns predicted output based on algorithm
# in this method we are using k-nearest neighbour
@app.route('/ad/knn', methods=['POST'])
def predict_knn_ad():
    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    # loading standard scalar object through pickel file
    scalar = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/scalar.pkl')

    # obtaining scaled values after scaling based on scalar object
    x_values = scalar.transform(x_values)

    # getting classifier object from pkl file
    classifier = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/knn_classifier_ad.pkl')

    # using that classifier object to obtain predicted output
    predicted = fit_or_predict(x_values, y_values, classifier).tolist()

    # making an dictionary to store result
    result = {}

    # saving actual and predicted value into dictionary
    result["actual value : "] = y_values.tolist()[0][0]
    result['predicted value'] = predicted[0]
    # returning result
    return jsonify(result)


# method takes a post type request
# returns predicted output based on algorithm
# in this method we are using Support Vector Machine
@app.route('/ad/svm', methods=['POST'])
def predict_support_ad():
    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    # loading standard scalar object through pickel file
    scalar = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/scalar.pkl')

    # obtaining scaled values after scaling based on scalar object
    x_values = scalar.transform(x_values)

    # getting classifier object from pkl file
    classifier = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/support_classifier_ad.pkl')

    # using that classifier object to obtain predicted output
    predicted = fit_or_predict(x_values, y_values, classifier).tolist()

    # making an dictionary to store result
    result = {}

    # saving actual and predicted value into dictionary
    result["actual value : "] = y_values.tolist()[0][0]
    result['predicted value'] = predicted[0]
    # returning result
    return jsonify(result)



# method takes a post type request
# returns predicted output based on algorithm
# in this method we are using Decision Tree Classification
@app.route('/ad/dtc', methods=['POST'])
def predict_decision_ad():
    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    # loading standard scalar object through pickel file
    scalar = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/scalar.pkl')

    # obtaining scaled values after scaling based on scalar object
    x_values = scalar.transform(x_values)

    # getting classifier object from pkl file
    classifier = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/decision_classifier_ad.pkl')

    # using that classifier object to obtain predicted output
    predicted = fit_or_predict(x_values, y_values, classifier).tolist()

    # making an dictionary to store result
    result = {}

    # saving actual and predicted value into dictionary
    result["actual value : "] = y_values.tolist()[0][0]
    result['predicted value'] = predicted[0]
    # returning result
    return jsonify(result)


# method takes a post type request
# returns predicted output based on algorithm
# in this method we are using Random Forest Classification
@app.route('/ad/rfc', methods=['POST'])
def predict_random_ad():
    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    # loading standard scalar object through pickel file
    scalar = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/scalar.pkl')

    # obtaining scaled values after scaling based on scalar object
    x_values = scalar.transform(x_values)

    # getting classifier object from pkl file
    classifier = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/forest_classifier_ad.pkl')

    # using that classifier object to obtain predicted output
    predicted = fit_or_predict(x_values, y_values, classifier).tolist()

    # making an dictionary to store result
    result = {}

    # saving actual and predicted value into dictionary
    result["actual value : "] = y_values.tolist()[0][0]
    result['predicted value'] = predicted[0]
    # returning result
    return jsonify(result)




# method takes a post type request
# returns predicted output based on algorithm
# in this method we are using Logical Regression
@app.route('/hiv/log', methods=['POST'])
def predict_logical_hiv():
    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    # x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    # loading standard scalar object through pickel file
    encoder = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/encoder')

    # obtaining x and y values seperated after pre processing through my general method for this data set
    x_values, y_values, encoder = pre_processing_hiv(pd.DataFrame(data, index=[0]), encoder,1)

    # getting classifier object from pkl file
    classifier = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/logistic_classifier_hiv.pkl')

    # using that classifier object to obtain predicted output
    predicted = fit_or_predict(x_values, y_values, classifier).tolist()

    # making an dictionary to store result
    result = {}

    # saving actual and predicted value into dictionary
    result["actual value : "] = y_values.tolist()[0]
    result['predicted value'] = predicted[0]
    # returning result
    return jsonify(result)


# method takes a post type request
# returns predicted output based on algorithm
# in this method we are using K- Nearest Neighbor
@app.route('/hiv/knn', methods=['POST'])
def predict_knn_hiv():
    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    # x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    # loading standard scalar object through pickel file
    encoder = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/encoder')

    # obtaining x and y values seperated after pre processing through my general method for this data set
    x_values, y_values, encoder = pre_processing_hiv(pd.DataFrame(data, index=[0]), encoder,1)

    # getting classifier object from pkl file
    classifier = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/knn_classifier_hiv.pkl')

    # using that classifier object to obtain predicted output
    predicted = fit_or_predict(x_values, y_values, classifier).tolist()

    # making an dictionary to store result
    result = {}

    # saving actual and predicted value into dictionary
    result["actual value : "] = y_values.tolist()[0]
    result['predicted value'] = predicted[0]
    # returning result
    return jsonify(result)

# method takes a post type request
# returns predicted output based on algorithm
# in this method we are using Support Vector Machine
@app.route('/hiv/svm', methods=['POST'])
def predict_svm_hiv():
    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    # x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    # loading standard scalar object through pickel file
    encoder = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/encoder')

    # obtaining x and y values seperated after pre processing through my general method for this data set
    x_values, y_values, encoder = pre_processing_hiv(pd.DataFrame(data, index=[0]), encoder,1)

    # getting classifier object from pkl file
    classifier = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/support_classifier_hiv.pkl')

    # using that classifier object to obtain predicted output
    predicted = fit_or_predict(x_values, y_values, classifier).tolist()

    # making an dictionary to store result
    result = {}

    # saving actual and predicted value into dictionary
    result["actual value : "] = y_values.tolist()[0]
    result['predicted value'] = predicted[0]
    # returning result
    return jsonify(result)

# method takes a post type request
# returns predicted output based on algorithm
# in this method we are using Decision Tree Classifier
@app.route('/hiv/dtc', methods=['POST'])
def predict_dtc_hiv():
    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    # x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    # loading standard scalar object through pickel file
    encoder = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/encoder')

    # obtaining x and y values seperated after pre processing through my general method for this data set
    x_values, y_values, encoder = pre_processing_hiv(pd.DataFrame(data, index=[0]), encoder,1)

    # getting classifier object from pkl file
    classifier = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/decision_classifier_hiv.pkl')

    # using that classifier object to obtain predicted output
    predicted = fit_or_predict(x_values, y_values, classifier).tolist()

    # making an dictionary to store result
    result = {}

    # saving actual and predicted value into dictionary
    result["actual value : "] = y_values.tolist()[0]
    result['predicted value'] = predicted[0]
    # returning result
    return jsonify(result)


# method takes a post type request
# returns predicted output based on algorithm
# in this method we are using Random Forest Classification
@app.route('/hiv/dtc', methods=['POST'])
def predict_rfc_hiv():
    # taking input data from request and saving it in variable
    data = request.json

    # obtaining  x and y values based on preprocessing that we have done at the time of training our data
    # x_values, y_values = pre_processing_ad(pd.DataFrame(data, index=[0]))

    # loading standard scalar object through pickel file
    encoder = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/encoder')

    # obtaining x and y values seperated after pre processing through my general method for this data set
    x_values, y_values, encoder = pre_processing_hiv(pd.DataFrame(data, index=[0]), encoder,1)

    # getting classifier object from pkl file
    classifier = job.load('/home/admin3/ml_with_phoenix/classification/pkl_files_and_lib/forest_classifier_hiv.pkl')

    # using that classifier object to obtain predicted output
    predicted = fit_or_predict(x_values, y_values, classifier).tolist()

    # making an dictionary to store result
    result = {}

    # saving actual and predicted value into dictionary
    result["actual value : "] = y_values.tolist()[0]
    result['predicted value'] = predicted[0]
    # returning result
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)