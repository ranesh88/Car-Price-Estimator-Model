from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
from sklearn.preprocessing import StandardScaler
import sklearn
import pickle
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('xgb.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
car=pd.read_csv('car_data.csv')

@app.route('/',methods=['GET','POST'])
def index():
    company=sorted(car['company'].unique())
    name = sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel=sorted(car['fuel'].unique())
    seller_type = sorted(car['seller_type'].unique())
    transmission = sorted(car['transmission'].unique())
    owner = sorted(car['owner'].unique())
    seats = sorted(car['seats'].unique(), reverse=True)

    company.insert(0,'Select Company name')
    name.insert(0, 'Select Model name')
    year.insert(0, 'Select Purchasing year')
    fuel.insert(0, 'Select Fuel Type')
    seller_type.insert(0, 'Select nature of seller')
    transmission.insert(0, 'Select Mode of transmission')
    owner.insert(0, 'Select type of owner')
    seats.insert(0, 'Select number of seats')
    return render_template('index.html',company=company,name=name,year=year,fuel = fuel,seller_type=seller_type,transmission=transmission,owner=owner,seats=seats)


@app.route('/predict',methods=['POST'])
@cross_origin()

def predict():
    name  = request.form.get('name')
    company = request.form.get('company')
    year = request.form.get('year')
    fuel= request.form.get('fuel')
    seller_type = request.form.get('seller_type')
    transmission = request.form.get('transmission')
    owner = request.form.get('owner')
    seats = request.form.get('seats')
    km_driven = request.form.get('km_driven')
    mileage = request.form.get('mileage')
    engine = request.form.get('engine')
    max_power = request.form.get('max_power')

    features = np.array([[name, company, year, km_driven, fuel, seller_type,transmission, owner, mileage, engine, max_power, seats]],dtype=object)
    transformed_features = preprocessor.transform(features)
    prediction = model.predict(transformed_features).reshape(1,-1)

    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.debug = True
    app.run()