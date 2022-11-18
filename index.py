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
from sklearn.linear_model import LinearRegression 
from sklearn import linear_model

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

df = pd.read_csv("/Users/armen/Desktop/web/files/bitacora.csv")
df = pd.DataFrame(df)
X= pd.DataFrame(df,
               columns=["Nivel Designado","Mes R","Año R","Empresa"])
y=df['diferencia'] #Target
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 5)
#Forest
model = RandomForestClassifier(n_estimators=1000)#Forest
model.fit(X_train, y_train)
ypred = model.predict(X_test)

#predecir para forest
def predecir(n1,n2,n3,n4):
  return(model.predict(X=[[n1,n2,n3,n4]]))

#para regresion
regr = linear_model.LinearRegression()
X1= pd.DataFrame(df,
               columns=["Nivel Designado"])

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y,random_state = 5)

# Entrenamos nuestro modelo
modelo= regr.fit(X1_train, y1_train)
 
# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y1_pred = regr.predict(X1_train)
#predecirR
def predecirR(n1):
  return( regr.predict(X=[[n1]]))

app= Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictivoForest', methods=["GET","POST"])
def about():
    if request.method == 'POST':
        nivel= request.form['nivel'] 
        mes=request.form['mes'] 
        year=request.form['year']
        empresa=request.form['empresa'] 
        result=predecir(nivel,mes, year, empresa)
        
        return render_template('resul.html', result=result)
    else :
        return render_template('predictivoForest.html')

@app.route('/predictivoRegresion', methods=["GET","POST"])
def aboutR():
    if request.method == 'POST':
        nivelr= request.form['nivelr'] 
        resultR=predecirR(nivelr)
        
        return f'<div class="alert alert-primary" role="alert">'+ str(resultR) +'</div>'
    else :
        return render_template('predicitvoRegresion.html')

if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0')
