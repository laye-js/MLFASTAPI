# Importation des bibliothèques nécessaires
from sklearn.tree import DecisionTreeClassifier
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
import pandas as pd

app = FastAPI()
@app.get("/predict")
async def predict(height: float, weight: float,age: float):
    # Chargement des données
    data= pd.read_csv("./data.csv")
    X = data[["height", "weight", "age"]]
    y = data["sex"]

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialisation et entraînement du modèle d'arbre de décision
    dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
    dt.fit(X_train, y_train)

    # Prédiction sur des données de test
    new_data = [[height, weight, age], [1.6, 45, 35], [1.9, 78, 45]]
    predictions = dt.predict(new_data)
    print("Prédictions:", predictions[0])
    return predictions[0]
