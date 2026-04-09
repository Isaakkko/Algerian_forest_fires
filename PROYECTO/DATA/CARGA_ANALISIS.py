# Clase cargador_datos

# Este script implementa una clase encargada de cargar un archivo CSV
# y calcular información básica sobre su estado:
# - Número de filas y columnas
# - Porcentaje total de valores nulos
# - Cantidad de registros duplicados
# - Tipos de dato


import pandas as pd


class cargador_datos:

    def __init__(self, ruta_csv):
        self.ruta_csv = ruta_csv
        self.data = None
        self.filas = 0
        self.columnas = 0
        self.porcentaje_nulos = 0
        self.duplicados = 0

    def cargar_datos(self):
        # Leer el archivo CSV
        self.data = pd.read_csv(self.ruta_csv)

        # Obtener dimensiones del dataset
        self.filas, self.columnas = self.data.shape

        # Calcular porcentaje de nulos
        total_celdas = self.filas * self.columnas
        total_nulos = self.data.isnull().sum().sum()
        self.porcentaje_nulos = (total_nulos / total_celdas) * 100

        # Calcular registros duplicados
        self.duplicados = self.data.duplicated().sum()

        # Tipos de datos
        self.tipos_variables = self.data.dtypes

    def resumen(self):
        print("Filas:", self.filas)
        print("Columnas:", self.columnas)
        print("Porcentaje de nulos:", round(self.porcentaje_nulos, 2), "%")
        print("Duplicados:", self.duplicados)
        print("Tipos de Variables:      ", self.tipos_variables)


# Ejecución de prueba
cargador = cargador_datos("../DATA/Algerian_forest_fires_dataset_CLEANED.csv")
cargador.cargar_datos()
cargador.resumen()






