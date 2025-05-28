import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import seaborn as sns

# Hipotesis 1:
# Cargar datos
data = pd.read_csv("../Data/Demand_data.csv") 
#Leer el CSV desde un string

data['Hora'] = pd.to_datetime(data['Hora'])

# Extraer hora en formato decimal y día de la semana
data['Hora_num'] = data['Hora'].dt.hour + data['Hora'].dt.minute / 60
data['Dia_semana'] = data['Hora'].dt.dayofweek  # Lunes=0, Domingo=6

# Clasificar en periodos
def clasificar_periodo(row):
    hora = row['Hora_num']
    dia = row['Dia_semana']
    
    if dia >= 5:  # Sábado o domingo
        return 'Valle'
    elif 0 <= hora < 8:
        return 'Valle'
    elif 10 <= hora < 14 or 18 <= hora < 22:
        return 'Punta'
    else:
        return 'Llano'

# Aplicar la clasificación
data['Periodo'] = data.apply(clasificar_periodo, axis=1)

# Grafica boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x='Periodo', y='Real', data=data, order=['Valle', 'Llano', 'Punta'], palette='Set2')
plt.title('Distribución de la demanda eléctrica por periodo')
plt.xlabel('Periodo')
plt.ylabel('Demanda real (MW)')
plt.grid(True)
plt.show()

# Grafica de barras


