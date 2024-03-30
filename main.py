from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from flask_cors import CORS

sysCropCode = 0
sysCropDays = 0
sysTemperature = 0
sysHumidity = 0
sysSoilMoisture = 0

app = Flask(__name__)
CORS(app)

def train(cropCode:int,cropDays:int,soilMoisture:int,temperature:int,humidity:int):
    dataset = pd.read_csv('api\crops.csv')
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    POST = classifier.predict(sc.transform([[cropCode,cropDays,soilMoisture,temperature,humidity]]))

    return POST

@app.route("/",methods=['GET'])
def home():
    return "<h1> Hello World </h1>"

@app.route("/api/store/<cropCode>/<cropDays>/<temperature>/<humidity>")
def store(cropCode:int,cropDays:int,temperature:int,humidity:int):
    sysCropCode = cropCode
    sysCropDays = cropDays
    sysTemperature = temperature
    sysHumidity = humidity

    return jsonify({'response':"Data Uploaded"})

@app.route('/soil-moisture', methods=['GET'])
def handle_soil_moisture():
    # Get soil moisture level from query parameter
    moisture_level = int(request.args.get('level'))
    sysSoilMoisture = moisture_level
    requiredMoisture = train(sysCropCode,sysCropDays,sysSoilMoisture,sysTemperature,sysHumidity)
    # Example logic: If moisture level is below a threshold, turn on the pump
    if moisture_level < requiredMoisture:
        # Return 1 to indicate pump is turned on
        return jsonify({'response': 1})

    # Otherwise, return 0 to indicate pump is off
    return jsonify({'response': 0})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
