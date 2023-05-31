from flask import Flask, redirect,render_template,request
import numpy as np
import joblib
app=Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route("/submit",methods=['POST','GET'])
def predict():
    if request.method=='POST':
        eop=float(request.form['eop'])
        reg=joblib.load('regression_model.joblib')
        data=np.array([eop]).reshape(1,-1)
        results=reg.predict(data)
        results=str(results)
        print(results)
    return results

if __name__=='__main__':
    app.run(debug=True)