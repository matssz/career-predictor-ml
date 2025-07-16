# career-predictor-ml
Classificador de carreira com machine learning baseado em perfil técnico.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Dataset simulado
data = pd.DataFrame({
    'anos_experiencia': [0, 2, 3, 5, 1, 4, 0, 6, 7, 2],
    'formacao': [0, 1, 1, 2, 0, 2, 0, 2, 2, 1],
    'linguagem': [1, 0, 2, 2, 1, 0, 1, 2, 0, 2],
    'interesse': [0, 1, 2, 0, 2, 1, 0, 1, 2, 0],
    'vaga': [0, 1, 2, 0, 2, 1, 0, 1, 2, 0]
})

X = data.drop('vaga', axis=1)
y = data['vaga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(\"Acurácia:\", accuracy_score(y_test, pred))

joblib.dump(model, 'modelo_carreiras.pkl')
