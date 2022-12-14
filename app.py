import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
#Loading pickle model
model = pickle.load(open("regresmodel.pkl","rb"))
scalar = pickle.load(open("scaling.pkl","rb"))

@app.route('/')
def homepage():
    return render_template('Homepage.html')

#predict 
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data) 
    print(np.array(list(data.values())).reshape(1,-1))
    newdata=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(newdata)
    print(output[0])
    return jsonify(output[0])

@app.route("/predict",methods=["POST"])
def prediction():
    data = [float(x) for x in request.form.values()]
    trans = scalar.transform(np.array(data).reshape(1,-1))
    output = model.predict(trans)[0]
    return render_template("Homepage.html",predicted_value="Predicted House price is (in 1000$):     {}".format(output))


if __name__=="__main__":
    app.run(debug=True)