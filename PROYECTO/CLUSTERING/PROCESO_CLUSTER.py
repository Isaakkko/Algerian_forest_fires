from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# CONFIGURACIÓN

RANDOM_STATE = 42
K_MIN = 2
K_MAX = 9


# RUTAS DEL PROYECTO

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "DATA"
RESULTADOS_DIR = PROJECT_DIR / "RESULTADOS"
MODELOS_DIR = PROJECT_DIR / "MODELOS"

DATASET_PATH = DATA_DIR / "Algerian_forest_fires_dataset_CLEANED.csv"

RESULTADO_PATH = (
    RESULTADOS_DIR
    / "Algerian_forest_fires_con_clusters.csv"
)

CENTROIDES_PATH = (
    RESULTADOS_DIR
    / "centroides_clusters.csv"
)

SILHOUETTE_PATH = (
    RESULTADOS_DIR
    / "resultados_silhouette.csv"
)

PCA_RESULTADOS_PATH = (
    RESULTADOS_DIR
    / "datos_clusters_pca.csv"
)

GRAFICO_CODO_PATH = (
    RESULTADOS_DIR
    / "metodo_codo.png"
)

GRAFICO_SILHOUETTE_PATH = (
    RESULTADOS_DIR
    / "silhouette_score.png"
)

GRAFICO_CLUSTERS_PATH = (
    RESULTADOS_DIR
    / "visualizacion_clusters_pca.png"
)

MODELO_KMEANS_PATH = (
    MODELOS_DIR
    / "modelo_kmeans.pkl"
)

SCALER_PATH = (
    MODELOS_DIR
    / "scaler_clustering.pkl"
)

PCA_PATH = (
    MODELOS_DIR
    / "modelo_pca.pkl"
)

COLUMNAS_PATH = (
    MODELOS_DIR
    / "columnas_clustering.pkl"
)


RESULTADOS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELOS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# CARGA DE DATOS

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        "No se encontró el dataset en la ruta:\n"
        f"{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("DATASET CARGADO CORRECTAMENTE")
print("=" * 60)
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")
print()


# PREPARACIÓN DE VARIABLES

columnas_excluir = [
    "day",
    "month",
    "year",
    "Classes"
]

columnas_faltantes = [
    columna
    for columna in columnas_excluir
    if columna not in df.columns
]

if columnas_faltantes:
    raise ValueError(
        "No se encontraron las siguientes columnas requeridas: "
        + ", ".join(columnas_faltantes)
    )

df_cluster = df.drop(
    columns=columnas_excluir
).copy()

df_cluster = df_cluster.select_dtypes(
    include="number"
)

if df_cluster.empty:
    raise ValueError(
        "No existen columnas numéricas disponibles "
        "para realizar el clustering."
    )

df_cluster = df_cluster.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

if df_cluster.isnull().sum().sum() > 0:
    print("Se encontraron valores faltantes.")
    print("Se reemplazarán por la mediana de cada variable.")
    print()

    df_cluster = df_cluster.fillna(
        df_cluster.median(
            numeric_only=True
        )
    )

if df_cluster.isnull().sum().sum() > 0:
    raise ValueError(
        "Existen valores faltantes que no pudieron "
        "ser reemplazados."
    )

print("=" * 60)
print("VARIABLES UTILIZADAS PARA EL CLUSTERING")
print("=" * 60)

for columna in df_cluster.columns:
    print(f"- {columna}")

print()


# ESCALAMIENTO

scaler = StandardScaler()

df_scaled = scaler.fit_transform(
    df_cluster
)


# MÉTODO DEL CODO

inercias = []

valores_k_codo = range(
    1,
    K_MAX + 1
)

for k in valores_k_codo:
    modelo = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    modelo.fit(df_scaled)
    inercias.append(modelo.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    list(valores_k_codo),
    inercias,
    marker="o"
)

plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.xticks(list(valores_k_codo))
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAFICO_CODO_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# SILHOUETTE SCORE

resultados_silhouette = []

print("=" * 60)
print("RESULTADOS DEL SILHOUETTE SCORE")
print("=" * 60)

for k in range(K_MIN, K_MAX + 1):
    modelo = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    etiquetas = modelo.fit_predict(
        df_scaled
    )

    score = silhouette_score(
        df_scaled,
        etiquetas
    )

    resultados_silhouette.append(
        {
            "k": k,
            "silhouette_score": score
        }
    )

    print(
        f"k = {k}, "
        f"Silhouette Score = {score:.4f}"
    )

df_silhouette = pd.DataFrame(
    resultados_silhouette
)

mejor_fila = df_silhouette.loc[
    df_silhouette["silhouette_score"].idxmax()
]

mejor_k = int(
    mejor_fila["k"]
)

mejor_score = float(
    mejor_fila["silhouette_score"]
)

print()
print(f"Mejor número de clusters: {mejor_k}")
print(f"Mejor Silhouette Score: {mejor_score:.4f}")
print()


# GRÁFICO DEL SILHOUETTE SCORE

plt.figure(figsize=(8, 5))

plt.plot(
    df_silhouette["k"],
    df_silhouette["silhouette_score"],
    marker="o"
)

plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Evaluación de clusters con Silhouette Score")
plt.xticks(df_silhouette["k"])
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAFICO_SILHOUETTE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# MODELO FINAL

# Se selecciona automáticamente el valor de k
# con el mayor Silhouette Score.

K_FINAL = mejor_k

modelo_final = KMeans(
    n_clusters=K_FINAL,
    random_state=RANDOM_STATE,
    n_init=10
)

clusters = modelo_final.fit_predict(
    df_scaled
)


# AGREGAR CLUSTERS AL DATASET

df_resultado = df.copy()
df_resultado["Cluster"] = clusters

print("=" * 60)
print("CANTIDAD DE REGISTROS POR CLUSTER")
print("=" * 60)

print(
    df_resultado["Cluster"]
    .value_counts()
    .sort_index()
)

print()


# CENTROIDES

centroides_escalados = (
    modelo_final.cluster_centers_
)

centroides_originales = (
    scaler.inverse_transform(
        centroides_escalados
    )
)

df_centroides = pd.DataFrame(
    centroides_originales,
    columns=df_cluster.columns
)

df_centroides.index.name = "Cluster"

print("=" * 60)
print("CENTROIDES DE LOS CLUSTERS")
print("=" * 60)

print(
    df_centroides.round(2)
)

print()


# PCA PARA VISUALIZACIÓN

pca = PCA(
    n_components=2
)

datos_pca = pca.fit_transform(
    df_scaled
)

df_pca = pd.DataFrame(
    datos_pca,
    columns=[
        "Componente_Principal_1",
        "Componente_Principal_2"
    ]
)

df_pca["Cluster"] = clusters

varianza_explicada = (
    pca.explained_variance_ratio_ * 100
)

print("=" * 60)
print("VARIANZA EXPLICADA POR PCA")
print("=" * 60)

print(
    f"Componente principal 1: "
    f"{varianza_explicada[0]:.2f}%"
)

print(
    f"Componente principal 2: "
    f"{varianza_explicada[1]:.2f}%"
)

print(
    f"Varianza total representada: "
    f"{varianza_explicada.sum():.2f}%"
)

print()


# VISUALIZACIÓN DE CLUSTERS CON PCA

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    df_pca["Componente_Principal_1"],
    df_pca["Componente_Principal_2"],
    c=df_pca["Cluster"],
    alpha=0.7
)

plt.xlabel(
    "Componente principal 1 "
    f"({varianza_explicada[0]:.2f}%)"
)

plt.ylabel(
    "Componente principal 2 "
    f"({varianza_explicada[1]:.2f}%)"
)

plt.title(
    f"Clusters obtenidos con K-Means, k = {K_FINAL}"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    GRAFICO_CLUSTERS_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# GUARDAR RESULTADOS

df_resultado.to_csv(
    RESULTADO_PATH,
    index=False
)

df_centroides.to_csv(
    CENTROIDES_PATH
)

df_silhouette.to_csv(
    SILHOUETTE_PATH,
    index=False
)

df_pca.to_csv(
    PCA_RESULTADOS_PATH,
    index=False
)


# GUARDAR MODELOS

joblib.dump(
    modelo_final,
    MODELO_KMEANS_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

joblib.dump(
    pca,
    PCA_PATH
)

joblib.dump(
    list(df_cluster.columns),
    COLUMNAS_PATH
)


# RESUMEN FINAL

print("=" * 60)
print("PROCESO FINALIZADO CORRECTAMENTE")
print("=" * 60)

print(f"Número final de clusters: {K_FINAL}")
print(f"Silhouette Score final: {mejor_score:.4f}")
print()

print("Archivos generados:")
print(f"- Dataset con clusters:\n  {RESULTADO_PATH}")
print(f"- Centroides:\n  {CENTROIDES_PATH}")
print(f"- Resultados Silhouette:\n  {SILHOUETTE_PATH}")
print(f"- Datos PCA:\n  {PCA_RESULTADOS_PATH}")
print(f"- Gráfico del codo:\n  {GRAFICO_CODO_PATH}")
print(f"- Gráfico Silhouette:\n  {GRAFICO_SILHOUETTE_PATH}")
print(f"- Gráfico PCA:\n  {GRAFICO_CLUSTERS_PATH}")
print()

print("Modelos guardados:")
print(f"- K-Means:\n  {MODELO_KMEANS_PATH}")
print(f"- StandardScaler:\n  {SCALER_PATH}")
print(f"- PCA:\n  {PCA_PATH}")
print(f"- Columnas utilizadas:\n  {COLUMNAS_PATH}")