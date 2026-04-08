# 🌲 Algerian Forest Fires Analysis

## 📌 Descripción del Proyecto

Este proyecto tiene como finalidad analizar datos relacionados con incendios forestales en Argelia, utilizando técnicas de análisis de datos para comprender cómo influyen distintas variables climáticas en la aparición de incendios.

El dataset incluye información como temperatura, humedad relativa, velocidad del viento, lluvia y varios índices del sistema FWI (Fire Weather Index). A partir de estos datos, se realiza un análisis exploratorio que permite identificar patrones y comportamientos relevantes.

Este trabajo se enfoca en aplicar conceptos de minería de datos, limpieza de información y visualización, siendo útil como base para futuros modelos predictivos.

---

## 📊 Descripción del Dataset

El conjunto de datos utilizado en este proyecto contiene observaciones climáticas registradas durante el año 2012, específicamente entre los meses de junio y septiembre, en dos regiones de Argelia.

Estas observaciones permiten analizar el comportamiento de distintas variables meteorológicas y su relación con la ocurrencia de incendios forestales.

### 📋 Variables del Dataset

- **Fecha** → Formato `DD/MM/AAAA`, incluye día, mes y año de la observación.  
- **Temp** → Temperatura máxima al mediodía en grados Celsius, con valores entre 22 y 42.  
- **RH** → Humedad relativa expresada en porcentaje, con valores entre 21 y 90.  
- **Ws** → Velocidad del viento en km/h, con valores entre 6 y 29.  
- **Rain** → Cantidad total de lluvia del día en milímetros, con valores entre 0 y 16.8.  
- **FFMC** → Fine Fuel Moisture Code, índice del sistema FWI relacionado con la humedad del combustible fino, con valores entre 28.6 y 92.5.  
- **DMC** → Duff Moisture Code, índice del sistema FWI relacionado con la humedad de la materia orgánica intermedia, con valores entre 1.1 y 65.9.  
- **DC** → Drought Code, índice del sistema FWI que representa la sequía acumulada, con valores entre 7 y 220.4.  
- **ISI** → Initial Spread Index, índice del sistema FWI que mide la velocidad inicial de propagación del fuego, con valores entre 0 y 18.5.  
- **BUI** → Buildup Index, índice del sistema FWI que representa la acumulación de combustible disponible, con valores entre 1.1 y 68.  
- **FWI** → Fire Weather Index, índice general del sistema FWI que estima el potencial de peligro de incendio, con valores entre 0 y 31.1.  
- **Classes** → Variable categórica con dos clases: `Fire` y `Not Fire`.  
- **Region** → Variable que identifica la región de procedencia de los datos: `0` y `1`, correspondientes a Sidi Bel-Abbès y Bejaia.  

---

## 📂 Estructura del Proyecto

- **DATA/** → Contiene los datasets utilizados en el proyecto.  
- **CARGA_ANALISIS/** → Código para cargar, limpiar y preparar los datos.  
- **notebooks/** → Análisis exploratorio y visualizaciones.  
- **README.md** → Documentación del proyecto.  

---

## ⚙️ Funcionamiento del Proyecto

El proyecto sigue un flujo sencillo pero completo:

1. **Carga de datos**  
   Se leen los archivos desde la carpeta `DATA` utilizando scripts en Python.

2. **Limpieza de datos**  
   Se eliminan valores nulos, duplicados y se ajustan tipos de variables para mejorar la calidad del análisis.

3. **Análisis exploratorio (EDA)**  
   Se calculan estadísticas básicas y se generan gráficos para comprender el comportamiento de las variables.

4. **Interpretación de resultados**  
   Se identifican patrones entre las condiciones climáticas y la ocurrencia de incendios forestales.

El archivo `cargador_datos.py` centraliza la lógica de carga y preparación de datos, facilitando el uso del dataset dentro de notebooks o scripts de análisis.

---

## 🛠️ Funcionalidades

- Carga de datos desde archivos.  
- Limpieza y validación de datos.  
- Análisis estadístico básico.  
- Visualización de datos.  
- Identificación de patrones en incendios forestales.  
- Soporte para análisis en notebooks.  

---

## 📊 Tecnologías Utilizadas

- Python  
- Pandas  
- Matplotlib  
- Jupyter Notebook  

---

## 👤 Autores

- **Isaac Ulloa Calvo**  
- **Tiffany Méndez Quirós**  
- **Edward Vindas Rivera**  
- **Jean Carlo Ramírez Carranza** 
---

## 📌 Notas Finales

Este proyecto representa una introducción al análisis de datos aplicado a problemas reales, permitiendo comprender cómo factores ambientales influyen en los incendios forestales.

Además, sirve como base para futuras implementaciones de modelos de predicción o sistemas inteligentes.
