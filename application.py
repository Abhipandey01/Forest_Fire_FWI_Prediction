import pickle
from flask import Flask,request , jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# import ridge regressor and Standard Scaler pickle file
ridge_model=pickle.load(open("models/ridge.pkl",'rb'))
StandardScaler_model=pickle.load(open("models/scaler.pkl","rb"))

#Route for Homepage
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=['GET','POST'])
def home():
        if (request.method=="POST"):
            Temperature=float(request.form.get("Temperature"))
            RH=float(request.form.get("Ws"))
            Ws=float(request.form.get("RH"))
            Rain =float(request.form.get("Rain"))
            FFMC=float(request.form.get("FFMC"))
            DMC=float(request.form.get("DMC"))
            ISI=float(request.form.get("ISI"))
            Classes=float(request.form.get("Classes"))
            region=int(request.form.get("region"))

            new_data_scaled= StandardScaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,
            Classes,region]])
            result=ridge_model.predict(new_data_scaled)

            return render_template("home.html", result=result[0])

        else:
            return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")