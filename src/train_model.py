import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error,r2_score
from pathlib import Path

base = Path(__file__).parent.parent
cache = base / "data" / "houses.parquet"

df = pd.read_parquet(cache)
# print(df.head())

X = df[
    ['surface_m2', 
     'rooms', 
     'bathrooms', 
     'year_built', 
     'distance_center_km', 
     'has_garage',
     'has_elevator']
    ]

y = df['price_eur']

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# print(model.predict([[90], [100], [120]]))    #Precio de una vivienda de 90, 100 y 120m^2 si X tuviera solo una dimensión
# print("Intercepto:", model.intercept_)  #Precio de una vivienda de 0m^2, aunque no tenga sentido físico es necesario calcularlo
# print("Coeficiente:", model.coef_)  #cuánto aumenta o disminuye el precio estimado por cada metro cuadrado adicional, según el modelo

#Evaluación del modelo
y_pred = model.predict(X_test)

df_pred = pd.DataFrame({'surface_m2': X_test['surface_m2'].to_numpy(), 'y_pred': y_pred, 'y_test':y_test})
# print(df_pred.head())


print("MAE = ", round(mean_absolute_error(y_test, y_pred), 2), '€')
print("RMSE = ", round(root_mean_squared_error(y_test, y_pred), 2),'€')
print("R2 = ", round(r2_score(y_test, y_pred), 3))

#Sacamos por gráfica los puntos del precio real vs el predicho. Si estan bien predichos,
#debería dibujar una linea diagonal creciente representando la función identidad.
plt.scatter(y_test, y_pred)
plt.xlabel("Precio real (€)")
plt.ylabel("Precio predicho (€)")
plt.title("Precios reales vs. predichos")
min_price = min(y_test.min(), y_train.min())
max_price = max(y_test.max(), y_train.max())
plt.plot([min_price, max_price], [min_price, max_price], linestyle =  '--', color = 'red', label = "y = x")

plt.show()