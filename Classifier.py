import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold 


# Carga de datos
data = pd.read_csv('base_de_datos.csv')

X = data.iloc[: ,2:]
y = data["Category"]


# Crear un modelo de árbol de decisión multiclase
clf = DecisionTreeClassifier()

# Especificar el número de pliegues (k) para la validación cruzada
k = 5 

kf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 42) 

# Realizar validación cruzada y obtener la puntuación de precisión
scores = cross_val_score(clf, X, y, cv = kf, scoring = 'accuracy')

# Mostrar las puntuaciones de precisión para cada pliegue
for fold, score in enumerate(scores, 1):
    print(f"Precisión en el pliegue {fold}: {score}")

# Calcular la precisión promedio de todos los pliegues
average_accuracy = scores.mean()
print(f"Precisión promedio en la validación cruzada de {k}-fold: {average_accuracy}")
