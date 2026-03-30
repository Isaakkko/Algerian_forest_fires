import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Se parte de un dataset previamente limpio y procesado, proporcionado por el equipo de trabajo.

df = pd.read_csv("Algerian_forest_fires_dataset_CLEANED.csv")
df.head()

df_cluster = df.drop(columns=["day", "month", "year", "Classes"])
df_cluster.head()

df_cluster.describe()

# Escalamiento de datos

# Se aplica normalización para que todas las variables tengan la misma importancia en el cálculo de distancias
# del algoritmo K-Means.

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

# Método del codo

inertia = []

k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del Codo")
plt.show()

# Se busca el punto donde la curva de bajar de forma brusca.

# Silhouette Score

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    print(f"k = {k}, Silhouette Score = {score}")

# El valor más alto indica mejor separación entre clusters.
# Se usa junto con el método del codo.

# Para determinar el número óptimo de clusters, se aplicó el método del codo, donde se observó una disminución
# significativa de la inercia hasta k = 3, a partir del cual la mejora se vuelve menos pronunciada.

# Adicionalmente, se utilizó el Silhouette Score como métrica de validación, obteniendo el valor más alto para
# k = 2 (0.308), lo que indica una mejor separación y cohesión de los grupos.

# Por lo tanto, se selecciona k = 2 como el número óptimo de clusters, ya que ofrece el mejor equilibrio entre
# simplicidad del modelo y calidad de agrupamiento.