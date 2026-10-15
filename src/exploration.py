from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

base = Path(__file__).resolve().parent.parent
csv = base / "data" / "houses.csv"
cache = base / "data" / "houses.parquet"

if cache.exists():
    df = pd.read_parquet(cache)
else:
    df = pd.read_csv(csv)
    df.to_parquet(cache)
# #Observamos al forma que tiene nuestro archivo csv
print(df.head())

print(df.isna().sum())

# #Observamos que algunos datos del dataframe son de tipos incorrectos y que aparecen NaN que nos pueden dar problemas
df = df.dropna(subset=["bathrooms"])
df = df.dropna(subset=["has_garage"])
df['bathrooms'] = df['bathrooms'].astype(int)
df['has_garage'] = df['has_garage'].astype(int).astype(bool)
df['has_elevator'] = df['has_elevator'].astype(bool)
df['distance_center_km'] = df['distance_center_km'].fillna(df['distance_center_km'].median())

#Guardamos en la cache el df limpio
df.to_parquet(cache)

print(df.isna().sum())

#Sacamos estadísticos básicos de algunos campos.

prices = df['price_eur'].tolist() #Una columna de un df es un series y lo queremos en lista
mean_prices = np.mean(prices)
std_prices = np.std(prices)
q25_prices = np.quantile(prices, 0.25)
q50_prices = np.quantile(prices, 0.5)
q75_prices = np.quantile(prices, 0.75)
min_prices = np.min(prices)
max_prices = np.max(prices)

m2 = df['surface_m2'].tolist()
mean_m2 = np.mean(m2)
std_m2 = np.std(m2)
q25_m2 = np.quantile(m2, 0.25)
q50_m2 = np.quantile(m2, 0.5)
q75_m2 = np.quantile(m2, 0.75)
min_m2 = np.min(m2)
max_m2 = np.max(m2)

print('Media de precios: ', round(mean_prices,2), '€.')
print('Desviación típica de precios: ', round(std_prices,2), '€.')
print('Mínimo de precios: ', round(min_prices,2), '€')
print('Máximo de precios: ', round(max_prices,2), '€')

print('Media de superficie: ', round(mean_m2,2), 'm^2.')
print('Desviación típica de superficie: ', round(std_m2,2), 'm^2.')
print('Mínimo de superficie: ', round(min_m2,2), 'm^2.')
print('Máximo de superficie: ', round(max_m2,2), 'm^2.')

#Sacamos algunas gráficas para observar los datos

#HISTOGRAMA DE PRECIOS
plt.hist(prices, bins = 25, edgecolor = 'black')
plt.xlabel('Precio de vivienda')
plt.ylabel('Número de vivienda')
plt.title('Distribución de los precios')
plt.show()

#RELACIÓN ENTRE SUPERFICIE Y PRECIO
res = stats.linregress(m2, prices)

plt.scatter(m2, prices, alpha=0.3, s=10)
xs = np.linspace(min_m2, max_m2, 100)
plt.plot(xs, res.slope * xs + res.intercept, color="red",
         label=f"y = {res.slope:.2f}x + {res.intercept:.2f}  (R² = {res.rvalue**2:.3f})")
plt.legend()
plt.show()

#RELACION ENTRE DISTANCIA AL CENTRO Y PRECIO
distance_center = df['distance_center_km'].tolist()
max_dist = np.max(distance_center)
min_dist = np.min(distance_center)

res1 = stats.linregress(distance_center, prices)

plt.scatter(distance_center, prices, alpha=0.3, s=10)
xs = np.linspace(min_dist, max_dist, 100)
plt.plot(xs, res1.slope * xs + res1.intercept, color="red",
         label=f"y = {res1.slope:.2f}x + {res1.intercept:.2f}  (R² = {res1.rvalue**2:.3f})")
plt.legend()
plt.show()