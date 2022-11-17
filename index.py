from sklearn import datasets
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

df = pd.read_csv("/Users/armen/Desktop/web/files/bitacora.csv")
df = pd.DataFrame(df)
X= pd.DataFrame(df,
               columns=["Nivel Designado","Mes R","Año R","Empresa"])
y=df['diferencia'] #Target
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 5)
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
def predecir(n1,n2,n3,n4):
  return(model.predict(X=[[n1,n2,n3,n4]]))

app= Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictivo', methods=["GET","POST"])
def about():
    if request.method == 'POST':
        nivel= request.form['nivel'] 
        mes=request.form['mes'] 
        year=request.form['year']
        empresa=request.form['empresa'] 
        result=predecir(nivel,mes, year, empresa)
        
        return render_template('resul.html', result=result)
        #return f'<div class="alert alert-primary" role="alert">Los días que se va tardar son:'+ str(result) +'</div>'
    else :
        return render_template('predictivo.html')

if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0')
