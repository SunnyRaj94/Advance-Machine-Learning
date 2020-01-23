#importing necessarry libraries
#importing flask , jsonify,request

from flask import Flask,  jsonify, request
#importing joblib to play with pkl file
import joblib as job

#importing pandas
import pandas as pd
#importing numpy
import numpy as np

#seeting name to run this file as flask app
app = Flask(__name__)


#this function is pre - processing the json request data
# the same as done in training the model
def pre_processing(data,encoder):
    #making data frame from obtained json data
    data_set = pd.DataFrame(data,index=[0])
    # seperating catagorical columns
    catagorical_cols = ['weekday', 'weathersit', 'season', 'mnth']

    # obtaining one hot encoded array
    transformed_cols = encoder.transform(data_set[catagorical_cols]).toarray()

    # obtaining non- catagorical columns or continous columns from data set
    continous_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'weathersit', 'temp', 'windspeed']

    # making numpy array from continous columns
    x_values = np.array(data_set[continous_cols])

    # making numpy array from target variable
    y_values = np.array(data_set['cnt'])

    # concatenating both arrays
    x_values = np.append(x_values, transformed_cols, axis=1)

    return x_values,y_values


# this function takes the post type of request
# and returns the predicted y value
@app.route('/predict',methods=['POST'])
def predict_output():
    #obtaining regressor object from pkl file
    regressor          = job.load('regressor.pkl')

    #obtaining encoder object from pkl file
    encoder            = job.load('encoder.pkl')

    #obtaining data from request body
    data               = request.json

    #obtaining data after pre-processing the same way that we did at time of training
    x_values, y_values = pre_processing(data,encoder)

    # making prediction based on regressor
    prediction         = regressor.predict(x_values)

    #making dictionary to store results
    result             = {}
    # m_s_e                              =  metrics.mean_squared_error(y_values, prediction)
    # r_m_s_e                            =  np.sqrt(metrics.mean_absolute_error(y_values, prediction))
    # m_a_e                              =  metrics.mean_absolute_error(y_values, prediction)
    result['predicted count']          =  prediction[0].tolist()
    result['actual count']             =  y_values[0].tolist()
    #returning result to user
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True)