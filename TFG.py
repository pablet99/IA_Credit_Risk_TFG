import pandas as pd #para leer archivos CSV
import numpy as np #para arrays y operaciones matemáticas
import seaborn as sns #para ver datos y regresiones
import matplotlib.pyplot as plt #para sacar funciones por pantalla
from sklearn.metrics import accuracy_score #para la regresión logística
seaborn 

from sklearn.linear_model import LogisticRegression #para hacer la regresión logística
from sklearn.preprocessing import StandardScaler #para estandarizar datos
from sklearn.model_selection import train_test_split #para poder dividir los datos de entrenamiento y test

from google.colab import drive #para poder acceder a Drive de colab 

#A continuación se leen los datos y se eliminan las filas que tengan "null"
datos_train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cs-training.csv",sep = ',',index_col=[0]) 

#train_limpios = datos_train.dropna()
datos_test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cs-test.csv",sep = ',',index_col=[0]) 
#test_limpios = datos_test.dropna()

#Saco un porcentaje de los datos que son de entreno y los que son de test
total_data = len(datos_train) + len(datos_test)
test_percent = len(datos_test)/total_data
train_percent = len(datos_train)/total_data
#imprimo
print("Porcentaje de datos de test: %.3f " % test_percent)
print("Porcentaje de datos de train: %.3f" % train_percent)

# Matriz de correlaciones.

# La matriz de correlación nos permite averiguar cómo de correladas o
# o relacionadas están dos variables. De este modo, podremos prescindir
# de campos cuya información sea explicada por otros campos.

corr = datos_train.corr()

# Inicializamos la firgura de matplotlib.
f, ax = plt.subplots(figsize=(7, 7))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Dibujamos el mapa de calor.
sns.heatmap(corr, cmap=cmap, vmax=1, center=0, vmin=-1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
            
            
#Dado que numerosas variables tienen una matriz de correlación igual a 1, 
#eliminaremos una de las dos, ya que podemos incidir en problemas de colinealidad
#Las relaciones de 1 son: 
# NumberOfTimes90DaysLate-NumberOfTIme30-59DaysPastDueNotWorse (elimino esta)
# NumberOfTimes90DaysLate-NumberOfTIme60-89DaysPastDueNotWorse (elimino esta)
#NumberOfTIme30-59DaysPastDueNotWorse-NumberOfTIme60-89DaysPastDueNotWorse (ambas)

datos_train2=datos_train.drop(["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse"],axis=1)
datos_test2=datos_test.drop(["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTime60-89DaysPastDueNotWorse"],axis=1)


datos_train2=datos_train2.dropna() #elimina las filas con Nan


#pasamos ahora a entrenar el modelo y comprobarlo mediante regresión logística

X_train_data = datos_train2.drop("SeriousDlqin2yrs",axis=1) #esta linea permite quitar la columna primera y quedarse con el resto
y_train_data= datos_train2["SeriousDlqin2yrs"] #la columna quitada es la que se adjudica a esa variable
X_test = datos_test2.drop("SeriousDlqin2yrs",axis=1) #lo mismo para el test
X_test=X_test.dropna() #elimino las filas que tienen valores NaN
y_test = datos_test2["SeriousDlqin2yrs"] #la primera columna del test se adjudica a esa variable

X_train, X_val, y_train, y_val = train_test_split(X_train_data, y_train_data, random_state = 42) #esto permite dividir los datos de la columna de forma aleatoria
#Las siguientes lineas sirven para estandarizar los datos
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# A continuación se realiza la regresión logística
lr = LogisticRegression(random_state=42, class_weight="balanced",max_iter=500) #se itera máximo 500 veces para que tenga fin 
lr.fit(X_train_scaled, y_train) 
y_pred = lr.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_pred)
print('Precisión: {:.2f}%'.format(accuracy*100))

output = pd.concat([X_test, pd.DataFrame({"predicciones": y_pred})], axis=1)
output.to_csv("/content/drive/MyDrive/Colab Notebooks/predicciones.csv", index=False)

#!pip install streamlit
import streamlit as st

st.title('Modelo de Riesgo de Crédito con Python')
st.write('Esta aplicación te permitirá predecir si un cliente es apto para recibir un crédito o no.')
uploaded_file = st.file_uploader("/content/drive/MyDrive/Colab Notebooks/predicciones.csv", type=['csv'])

if uploaded_file is not None:
    # Leer el archivo CSV y crear un DataFrame
    data = pd.read_csv(uploaded_file)

    # Mostrar el DataFrame en la aplicación
    st.write('Datos cargados:')
    st.write(data)

    # Predecir la probabilidad de crédito para cada cliente
    proba = model.predict_proba(data)[:, 1]

    # Mostrar las probabilidades en la aplicación
    st.write('Probabilidades de crédito:')
    st.write(proba)
    
