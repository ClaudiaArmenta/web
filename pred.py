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

#para abrir el archivo y verfificar que es correcto
df = pd.read_csv("/Users/armen/Desktop/web/files/bitacora.csv")
df = pd.DataFrame(df)

#seleccionamos las variables
X= pd.DataFrame(df,
               columns=["Nivel Designado","Mes R","Año R","Empresa"])
y=df['diferencia'] #Target

#creamos el modelo 
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 5)

model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
ypred = model.predict(X_test)

#para verificar la acurracy 
#print(metrics.classification_report(ypred, y_test))

#para hacer la compracion
comp =pd.DataFrame({"real":y_test, "preds": ypred})
#print(comp)

#otra compracion
ypred =model.predict(X=df[["Nivel Designado","Mes R","Año R","Empresa"]])
df.insert(0,'pred',ypred)

#print(df.head(10))

#Graficas
fig, axs = plt.subplots(ncols=1,figsize=(30,5))
sns.scatterplot(x="Mes R",y="diferencia",data=df)
sns.scatterplot(x="Mes R",y="pred",data=df)
#plt.show()

def predecir(n1,n2,n3,n4):
  return(model.predict(X=[[n1,n2,n3,n4]]))

