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
media_por_periodo = data.groupby('Periodo')['Real'].mean().loc[['Valle', 'Llano', 'Punta']]
media_por_periodo.plot(kind='bar', color=['skyblue', 'orange', 'lightcoral'])
plt.title('Demanda media por periodo del día')
plt.ylabel('Demanda real (MW)')
plt.grid(axis='y')
plt.show()

#mapa de calor
# Crear variables dummy para los periodos
period_dummies = pd.get_dummies(data['Periodo'])

# Concatenar con la columna 'Real'
corr_input = pd.concat([data['Real'], period_dummies], axis=1)

# Calcular la matriz de correlación
corr_matrix = corr_input.corr()

# Extraer solo la fila de 'Real' para ver su relación con cada periodo
corr_real = corr_matrix.loc[['Real'], ['Valle', 'Llano', 'Punta']]

# Graficar el mapa de calor
plt.figure(figsize=(6, 2))
sns.heatmap(corr_real, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación entre demanda real y periodos')
plt.show()


# Hipotesis 2:
#1. Cargar datasets
df1 = pd.read_csv("../Data/energy_dataset.csv", parse_dates=["time"], na_values=["", " "])
df2 = pd.read_csv("../Data/spain_energy_market.csv", parse_dates=["datetime"])

# 2. Convertir a datetime si no lo están, y eliminar zona horaria
df1["time"] = pd.to_datetime(df1["time"], errors="coerce", utc=True).dt.tz_convert(None)
df2["datetime"] = pd.to_datetime(df2["datetime"], errors="coerce").dt.tz_localize(None)

# 3. Crear columna de fecha
df1["date"] = df1["time"].dt.normalize()
df2["date"] = df2["datetime"].dt.normalize()

# 4. Agrupar precio spot diario
df2_daily = df2[["date", "value"]].rename(columns={"value": "spot_price_daily"})

# 5. Unir por fecha
df = df1.merge(df2_daily, on="date", how="left").drop(columns=["date"])

# 6. Exportar si quieres
df.to_csv("../Data/energy_merged.csv", index=False)

# Agrupar variables
df_grouped = df.copy()

# Rellenar valores nulos con 0 para el cálculo de sumas de grupos.
df_filled_for_sum = df_grouped.fillna(0)

# Función auxiliar para sumar columnas presentes
def sumar_columnas_existentes(df, columnas, nombre_nueva_columna):
    columnas_presentes = [col for col in columnas if col in df.columns]
    if len(columnas_presentes) < len(columnas):
        columnas_faltantes = [col for col in columnas if col not in df.columns]
        print(f"Advertencia: Las siguientes columnas NO se encontraron: {columnas_faltantes}. Se usarán solo las existentes.")
    return df[columnas_presentes].sum(axis=1) if columnas_presentes else 0.0

# 1. Generación Fósil (sin 'generation fossil gas')
fosil = [
    'generation fossil brown coal/lignite',
    'generation fossil coal-derived gas',
    'generation fossil hard coal',
    'generation fossil oil',
    'generation fossil oil shale',
    'generation fossil peat'
]
df_grouped['total_generation_fosil'] = sumar_columnas_existentes(df_filled_for_sum, fosil, 'total_generation_fosil')

# 2. Generación Gas (solo 'generation fossil gas')
if 'generation fossil gas' in df_filled_for_sum.columns:
    df_grouped['total_generation_gas'] = df_filled_for_sum['generation fossil gas']
else:
    print("Advertencia: La columna 'generation fossil gas' no se encontró.")
    df_grouped['total_generation_gas'] = 0.0

# 3. Generación Solar
solar = ['generation solar']
df_grouped['total_generation_solar'] = sumar_columnas_existentes(df_filled_for_sum, solar, 'total_generation_solar')

# 4. Generación Eólica
eolica = ['generation wind offshore', 'generation wind onshore']
df_grouped['total_generation_eolica'] = sumar_columnas_existentes(df_filled_for_sum, eolica, 'total_generation_eolica')

# 5. Otras Renovables
otras_renovables = ['generation geothermal', 'generation other renewable', 'generation waste', 'generation marine']
df_grouped['total_generation_otras_renovables'] = sumar_columnas_existentes(df_filled_for_sum, otras_renovables, 'total_generation_otras_renovables')

# 6. Generación Hidráulica
hidraulica = [
    'generation hydro pumped storage aggregated',
    'generation hydro pumped storage consumption',
    'generation hydro run-of-river and poundage',
    'generation hydro water reservoir'
]
df_grouped['total_generation_hidraulica'] = sumar_columnas_existentes(df_filled_for_sum, hidraulica, 'total_generation_hidraulica')

# 7. Generación Nuclear
if 'generation nuclear' in df_filled_for_sum.columns:
    df_grouped['total_generation_nuclear'] = df_filled_for_sum['generation nuclear']
else:
    print("Advertencia: La columna 'generation nuclear' no se encontró.")
    df_grouped['total_generation_nuclear'] = 0.0

# 8. Generación de Biomasa
if 'generation biomass' in df_filled_for_sum.columns:
    df_grouped['biomasa'] = df_filled_for_sum['generation biomass']
else:
    print("Advertencia: La columna 'generation biomass' no se encontró.")
    df_grouped['biomasa'] = 0.0

# 9. Otras Generaciones
if 'generation other' in df_filled_for_sum.columns:
    df_grouped['otras_generaciones'] = df_filled_for_sum['generation other']
else:
    print("Advertencia: La columna 'generation other' no se encontró.")
    df_grouped['otras_generaciones'] = 0.0

# Mostrar resultado resumido
print("\nDataFrame con las nuevas columnas de generación agrupadas (primeras 5 filas):")
print(df_grouped[['time',
                  'total_generation_fosil',
                  'total_generation_gas',
                  'total_generation_solar',
                  'total_generation_eolica',
                  'total_generation_otras_renovables',
                  'total_generation_hidraulica',
                  'total_generation_nuclear',
                  'biomasa',
                  'otras_generaciones']].head())

# # Agrupar renovables no_renovables
df_grouped = df.copy()
df_filled_for_sum = df_grouped.fillna(0)

# Función auxiliar
def sumar_columnas_existentes(df, columnas, nombre_nueva_columna):
    columnas_presentes = [col for col in columnas if col in df.columns]
    if len(columnas_presentes) < len(columnas):
        columnas_faltantes = [col for col in columnas if col not in df.columns]
        print(f"Advertencia: Las siguientes columnas NO se encontraron: {columnas_faltantes}. Se usarán solo las existentes.")
    return df[columnas_presentes].sum(axis=1) if columnas_presentes else 0.0

# Renovables
renovables = [
    'generation solar',
    'generation wind onshore',
    'generation wind offshore',
    'generation hydro pumped storage aggregated',
    'generation hydro pumped storage consumption',
    'generation hydro run-of-river and poundage',
    'generation hydro water reservoir',
    'generation geothermal',
    'generation other renewable',
    'generation waste',
    'generation marine',
    'generation biomass'
]
df_grouped['total_generation_renovables'] = sumar_columnas_existentes(df_filled_for_sum, renovables, 'total_generation_renovables')

# No Renovables
no_renovables = [
    'generation fossil brown coal/lignite',
    'generation fossil coal-derived gas',
    'generation fossil gas',
    'generation fossil hard coal',
    'generation fossil oil',
    'generation fossil oil shale',
    'generation fossil peat',
    'generation nuclear',
    'generation other'  # si aplica como no renovable
]
df_grouped['total_generation_no_renovables'] = sumar_columnas_existentes(df_filled_for_sum, no_renovables, 'total_generation_no_renovables')

# Calcolo de porcentajes
df_grouped = df.copy()
df_filled_for_sum = df_grouped.fillna(0)

# Función auxiliar
def sumar_columnas_existentes(df, columnas, nombre_nueva_columna):
    columnas_presentes = [col for col in columnas if col in df.columns]
    if len(columnas_presentes) < len(columnas):
        columnas_faltantes = [col for col in columnas if col not in df.columns]
        print(f"Advertencia: Las siguientes columnas NO se encontraron: {columnas_faltantes}. Se usarán solo las existentes.")
    return df[columnas_presentes].sum(axis=1) if columnas_presentes else 0.0

# Grupos
fosil = [
    'generation fossil brown coal/lignite',
    'generation fossil coal-derived gas',
    'generation fossil hard coal',
    'generation fossil oil',
    'generation fossil oil shale',
    'generation fossil peat'
]
gas = ['generation fossil gas']
solar = ['generation solar']
eolica = ['generation wind onshore', 'generation wind offshore']
otras_renovables = ['generation geothermal', 'generation other renewable', 'generation waste', 'generation marine']
hidraulica = [
    'generation hydro pumped storage aggregated',
    'generation hydro pumped storage consumption',
    'generation hydro run-of-river and poundage',
    'generation hydro water reservoir'
]
biomasa = ['generation biomass']
nuclear = ['generation nuclear']
otras = ['generation other']

# Renovables = solar + eólica + hidráulica + otras renovables + biomasa
renovables = solar + eolica + hidraulica + otras_renovables + biomasa
no_renovables = fosil + gas + nuclear + otras

# Totales por grupo
df_grouped['total_generation_fosil'] = sumar_columnas_existentes(df_filled_for_sum, fosil, 'total_generation_fosil')
df_grouped['total_generation_gas'] = sumar_columnas_existentes(df_filled_for_sum, gas, 'total_generation_gas')
df_grouped['total_generation_solar'] = sumar_columnas_existentes(df_filled_for_sum, solar, 'total_generation_solar')
df_grouped['total_generation_eolica'] = sumar_columnas_existentes(df_filled_for_sum, eolica, 'total_generation_eolica')
df_grouped['total_generation_otras_renovables'] = sumar_columnas_existentes(df_filled_for_sum, otras_renovables, 'total_generation_otras_renovables')
df_grouped['total_generation_hidraulica'] = sumar_columnas_existentes(df_filled_for_sum, hidraulica, 'total_generation_hidraulica')
df_grouped['total_generation_nuclear'] = sumar_columnas_existentes(df_filled_for_sum, nuclear, 'total_generation_nuclear')
df_grouped['biomasa'] = sumar_columnas_existentes(df_filled_for_sum, biomasa, 'biomasa')
df_grouped['otras_generaciones'] = sumar_columnas_existentes(df_filled_for_sum, otras, 'otras_generaciones')

# Agregados principales
df_grouped['total_generation_renovables'] = sumar_columnas_existentes(df_filled_for_sum, renovables, 'total_generation_renovables')
df_grouped['total_generation_no_renovables'] = sumar_columnas_existentes(df_filled_for_sum, no_renovables, 'total_generation_no_renovables')

# Total de generación por fila
df_grouped['total_generation'] = df_grouped[
    ['total_generation_renovables', 'total_generation_no_renovables']
].sum(axis=1)

# Porcentajes
def porcentaje(col):
    return (df_grouped[col] / df_grouped['total_generation']) * 100

df_grouped['%_renovables'] = porcentaje('total_generation_renovables')
df_grouped['%_no_renovables'] = porcentaje('total_generation_no_renovables')
df_grouped['%_fosil'] = porcentaje('total_generation_fosil')
df_grouped['%_gas'] = porcentaje('total_generation_gas')
df_grouped['%_solar'] = porcentaje('total_generation_solar')
df_grouped['%_eolica'] = porcentaje('total_generation_eolica')
df_grouped['%_hidraulica'] = porcentaje('total_generation_hidraulica')
df_grouped['%_nuclear'] = porcentaje('total_generation_nuclear')
df_grouped['%_biomasa'] = porcentaje('biomasa')
df_grouped['%_otras_generaciones'] = porcentaje('otras_generaciones')

# Mostrar una vista previa
columnas_mostrar = [
    'time',
    'total_generation', 'total_generation_renovables', 'total_generation_no_renovables',
    '%_renovables', '%_no_renovables', '%_fosil', '%_gas',
    '%_solar', '%_eolica', '%_hidraulica', '%_nuclear', '%_biomasa', '%_otras_generaciones'
]


# Grafico de lineas
# Asegúrate de que la columna 'time' esté en formato datetime
df_grouped['time'] = pd.to_datetime(df_grouped['time'])

# Crear figura y subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle("Generación Renovable vs No Renovable y Precio Actual", fontsize=16)

# --- Gráfico superior: Porcentaje Renovables vs No Renovables ---
ax1.plot(df_grouped['time'], df_grouped['%_renovables'], label='% Renovables', color='green', linewidth=2)
ax1.plot(df_grouped['time'], df_grouped['%_no_renovables'], label='% No Renovables', color='darkred', linewidth=2)
ax1.set_ylabel('Porcentaje (%)')
ax1.set_title('Porcentaje de Generación Renovable y No Renovable')
ax1.legend(loc='upper right')
ax1.grid(True)

# --- Gráfico inferior: Precio Actual ---
ax2.plot(df_grouped['time'], df_grouped['price actual'], label='Precio Actual', color='blue', linewidth=1.5)
ax2.set_ylabel('€/MWh')
ax2.set_xlabel('Tiempo')
ax2.set_title('Precio Actual de la Electricidad')
ax2.legend(loc='upper right')
ax2.grid(True)

# Ajuste de diseño
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Mapa de calor
# Seleccionar columnas relevantes
columnas_analisis = [
    '%_fosil', '%_gas', '%_solar', '%_eolica',
    '%_hidraulica', '%_nuclear', '%_biomasa',
    '%_otras_generaciones', '%_renovables',
    '%_no_renovables', 'price actual'
]

# Calcular correlación
correlation_matrix = df_grouped[columnas_analisis].corr()

# Visualizar con heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['price actual']].sort_values(by='price actual', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlación entre fuentes de generación y Precio")
plt.show()


# Hipoteis 3:
# Cargar el CSV
df = pd.read_csv("../Data/energy_grouped.csv", parse_dates=['time'])

# 1. Eliminar filas duplicadas
df = df.drop_duplicates()

# 2. Convertir columnas numéricas (forzar errores a NaN si aparecen strings)
for col in df.columns:
    if col != 'time':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Eliminar columnas con valores faltantes críticos o que estén completamente vacías
df = df.dropna(axis=1, how='all')  # elimina columnas totalmente vacías


# 5. Validar valores extremos
# Por ejemplo, limpiar spot_price_daily si hay valores anómalos
q_low = df["spot_price_daily"].quantile(0.01)
q_hi  = df["spot_price_daily"].quantile(0.99)
df = df[(df["spot_price_daily"] >= q_low) & (df["spot_price_daily"] <= q_hi)]

# 6. Reiniciar el índice
df.reset_index(drop=True, inplace=True)

# Guardar el dataset limpio
df.to_csv("dataset_limpio.csv", index=False)

# Cargar el dataset
df = pd.read_csv('../Data/dataset_limpio.csv', parse_dates=['time'])

# Asegúrate de que 'time' es tipo datetime
df['time'] = pd.to_datetime(df['time'])

# Crear la columna de franjas de 3 horas
def obtener_franja_3h(dt):
    inicio = dt.hour - (dt.hour % 3)
    fin = inicio + 3
    return f"{inicio:02d}:00-{fin:02d}:00"

df['franja_horaria_3h'] = df['time'].apply(obtener_franja_3h)

# Boxplot de la generación renovable por franja horaria
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='franja_horaria_3h', y='total_generation_renovable',
            order=[f"{i:02d}:00-{i+3:02d}:00" for i in range(0, 24, 3)])
plt.title('Distribución de generación renovable por franja horaria (3h)')
plt.xlabel('Franja horaria (3h)')
plt.ylabel('Generación renovable (MWh)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Comparativa por frnaja horaria
# Asegúrate de que 'time' es datetime
df['time'] = pd.to_datetime(df['time'])

# Crear franja horaria de 3 horas
def obtener_franja_3h(dt):
    inicio = dt.hour - (dt.hour % 3)
    fin = inicio + 3
    return f"{inicio:02d}:00-{fin:02d}:00"

df['franja_horaria_3h'] = df['time'].apply(obtener_franja_3h)

# Orden correcto de las franjas
orden_franjas = [f"{i:02d}:00-{i+3:02d}:00" for i in range(0, 24, 3)]

# Agrupar y calcular estadísticas
renovable_stats = df.groupby('franja_horaria_3h')['total_generation_renovable'].agg(['mean', 'std']).reindex(orden_franjas)
no_renovable_stats = df.groupby('franja_horaria_3h')['total_generation_no_renovable'].agg(['mean', 'std']).reindex(orden_franjas)

# Crear figura y ejes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Gráfico 1: Renovables
ax1.plot(renovable_stats.index, renovable_stats['mean'], label='Media renovable', color='green', marker='o')
ax1.fill_between(renovable_stats.index,
                 renovable_stats['mean'] - renovable_stats['std'],
                 renovable_stats['mean'] + renovable_stats['std'],
                 color='green', alpha=0.3, label='±1 STD')
ax1.set_title('Generación Renovable por Franja Horaria (3h)')
ax1.set_ylabel('Generación (MWh)')
ax1.set_ylim(0, 25000)
ax1.grid(True)
ax1.legend()

# Gráfico 2: No renovables
ax2.plot(no_renovable_stats.index, no_renovable_stats['mean'], label='Media no renovable', color='darkred', marker='o')
ax2.fill_between(no_renovable_stats.index,
                 no_renovable_stats['mean'] - no_renovable_stats['std'],
                 no_renovable_stats['mean'] + no_renovable_stats['std'],
                 color='red', alpha=0.3, label='±1 STD')
ax2.set_title('Generación No Renovable por Franja Horaria (3h)')
ax2.set_xlabel('Franja Horaria (3h)')
ax2.set_ylabel('Generación (MWh)')
ax2.set_ylim(0, 25000)
ax2.grid(True)
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Comparativa por estaciones

# Asegúrate de que 'time' es datetime
df['time'] = pd.to_datetime(df['time'])

# Crear franja horaria de 3 horas
def obtener_franja_3h(dt):
    inicio = dt.hour - (dt.hour % 3)
    fin = inicio + 3
    return f"{inicio:02d}:00-{fin:02d}:00"

df['franja_horaria_3h'] = df['time'].apply(obtener_franja_3h)

# Orden correcto de las franjas
orden_franjas = [f"{i:02d}:00-{i+3:02d}:00" for i in range(0, 24, 3)]

# Agrupar y calcular estadísticas
renovable_stats = df.groupby('franja_horaria_3h')['total_generation_renovable'].agg(['mean', 'std']).reindex(orden_franjas)
no_renovable_stats = df.groupby('franja_horaria_3h')['total_generation_no_renovable'].agg(['mean', 'std']).reindex(orden_franjas)

# Crear figura y ejes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Gráfico 1: Renovables
ax1.plot(renovable_stats.index, renovable_stats['mean'], label='Media renovable', color='green', marker='o')
ax1.fill_between(renovable_stats.index,
                 renovable_stats['mean'] - renovable_stats['std'],
                 renovable_stats['mean'] + renovable_stats['std'],
                 color='green', alpha=0.3, label='±1 STD')
ax1.set_title('Generación Renovable por Franja Horaria (3h)')
ax1.set_ylabel('Generación (MWh)')
ax1.set_ylim(0, 25000)
ax1.grid(True)
ax1.legend()

# Gráfico 2: No renovables
ax2.plot(no_renovable_stats.index, no_renovable_stats['mean'], label='Media no renovable', color='darkred', marker='o')
ax2.fill_between(no_renovable_stats.index,
                 no_renovable_stats['mean'] - no_renovable_stats['std'],
                 no_renovable_stats['mean'] + no_renovable_stats['std'],
                 color='red', alpha=0.3, label='±1 STD')
ax2.set_title('Generación No Renovable por Franja Horaria (3h)')
ax2.set_xlabel('Franja Horaria (3h)')
ax2.set_ylabel('Generación (MWh)')
ax2.set_ylim(0, 25000)
ax2.grid(True)
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Hipotesis 4:

tecnologias = [
    "Hidroeléctrica",
    "Eólica terrestre",
    "Fotovoltaica",
    "Térmica (carbón)",
    "Nuclear",
    "Ciclo combinado"
]

# Costes de instalación en millones de euros por MW
coste_instalacion_millones_por_MW = [3.5, 1.5, 0.9, 2.5, 8.0, 1.0]

# Años de amortización
amortizacion_anios = [40, 20, 15, 30, 50, 25]

# Costes operativos en euros por MWh
coste_operativo_por_MWh = [12, 15, 10, 45, 30, 55]

# Producción anual en MWh por MW instalado
produccion_anual_MWh = [3942, 2628, 1752, 5256, 7884, 4818]

# Cálculo del LCOE
LCOE = []
for i in range(len(tecnologias)):
    capex_por_MWh = (coste_instalacion_millones_por_MW[i] * 1_000_000) / (amortizacion_anios[i] * produccion_anual_MWh[i])
    lcoe_i = capex_por_MWh + coste_operativo_por_MWh[i]
    LCOE.append(round(lcoe_i, 2))

# Grafico de barras
#Datos
tecnologias = [
    "Hidroeléctrica", "Eólica terrestre", "Fotovoltaica",
    "Térmica (carbón)", "Nuclear", "Ciclo combinado"
]

coste_instalacion = [3.5, 1.5, 0.9, 2.5, 8.0, 1.0]         # M€/MW
amortizacion = [40, 20, 15, 30, 50, 25]                    # años
coste_operativo = [12, 15, 10, 45, 30, 55]                 # €/MWh
lcoe = [22.26, 28.47, 43.24, 60.84, 40.03, 63.25]          # €/MWh

# Agrupar los datos
labels = tecnologias
x = np.arange(len(labels))  # ubicaciones de grupos
width = 0.2  # ancho de barra

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - 1.5*width, coste_instalacion, width, label='Coste instalación (M€/MW)')
bars2 = ax.bar(x - 0.5*width, amortizacion, width, label='Amortización (años)')
bars3 = ax.bar(x + 0.5*width, coste_operativo, width, label='Coste operativo (€/MWh)')
bars4 = ax.bar(x + 1.5*width, lcoe, width, label='LCOE (€/MWh)')

# Etiquetas y formato
ax.set_ylabel('Valor')
ax.set_title('Comparación de variables por tecnología')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Grafico de burbujas
ecnologias = [
    "Hidroeléctrica", "Eólica terrestre", "Fotovoltaica",
    "Térmica (carbón)", "Nuclear", "Ciclo combinado"
]

coste_instalacion = [3.5, 1.5, 0.9, 2.5, 8.0, 1.0]         # M€/MW
amortizacion = [40, 20, 15, 30, 50, 25]                    # años
coste_operativo = [12, 15, 10, 45, 30, 55]                 # €/MWh
lcoe = [22.26, 28.47, 43.24, 60.84, 40.03, 63.25]          # €/MWh

# ---------------------------
# DataFrame para análisis
# ---------------------------
df = pd.DataFrame({
    "Tecnología": tecnologias,
    "Coste instalación (M€/MW)": coste_instalacion,
    "Amortización (años)": amortizacion,
    "Coste operativo (€/MWh)": coste_operativo,
    "LCOE (€/MWh)": lcoe
})


plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["Coste instalación (M€/MW)"],
    df["Coste operativo (€/MWh)"],
    s=[a * 10 for a in df["Amortización (años)"]],  # tamaño burbuja
    c=df["LCOE (€/MWh)"], cmap='viridis', alpha=0.8, edgecolors='black'
)

plt.colorbar(scatter, label="LCOE (€/MWh)")
for i, tech in enumerate(df["Tecnología"]):
    plt.text(df["Coste instalación (M€/MW)"][i]+0.05, df["Coste operativo (€/MWh)"][i], tech, fontsize=9)

plt.xlabel("Coste instalación (M€/MW)")
plt.ylabel("Coste operativo (€/MWh)")
plt.title("Gráfico de burbujas: tecnologías energéticas")
plt.grid(True)
plt.tight_layout()
plt.show()

# Hipotesis 5:

# Cargar el dataset
df = pd.read_csv('../Data/dataset_limpio.csv', parse_dates=['time'])

#Grafica
# Asegurar que la columna de tiempo es datetime
df["time"] = pd.to_datetime(df["time"])

# Extraer el año
df["Año"] = df["time"].dt.year

# Agrupar por año y sumar el consumo real (en MWh)
df_anual = df.groupby("Año")["total load actual"].sum().reset_index()

# Convertir a GWh para legibilidad
df_anual["Consumo_GWh"] = df_anual["total load actual"] / 1000

# Gráfico de evolución anual
plt.figure(figsize=(10, 6))
plt.plot(df_anual["Año"], df_anual["Consumo_GWh"], marker='o', linestyle='-', color='darkgreen')
plt.title("Evolución anual del consumo eléctrico (total load actual)")
plt.xlabel("Año")
plt.ylabel("Consumo total (GWh)")
plt.grid(True)
plt.tight_layout()
plt.show()

