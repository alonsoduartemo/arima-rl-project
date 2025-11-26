import pandas as pd
import numpy as np
import time
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings


warnings.filterwarnings("ignore")

def evaluar(train, test, p, d, q, nombre):
    start = time.time()
    try:
        # Entrenar
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        
        # Predecir
        pred = model_fit.forecast(steps=len(test))
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(test, pred))
        aic = model_fit.aic
        tiempo = time.time() - start
        
        # Imprimir fila para copiar
        print(f"{nombre:<20} | ({p},{d},{q})      | {aic:<10.2f} | {rmse:<10.2f} | {tiempo:.4f}")
    except:
        print(f"{nombre:<20} | ({p},{d},{q})      | ERROR      | ERROR      | ----")

def main():
    print("\n🚀 GENERANDO DATOS PARA EXPERIMENTO A (d=1 fijo)...")
    
    # 1. Cargar datos
    try:
        df = pd.read_csv('data/germany_monthly_power.csv', index_col=0, parse_dates=True)
    except:
        df = pd.read_csv('../data/germany_monthly_power.csv', index_col=0, parse_dates=True)
        
    train = df.iloc[:-6]['load_gwh']
    test = df.iloc[-6:]['load_gwh'] # Últimos 6 meses (2019)

    print("-" * 75)
    print(f"{'Método':<20} | {'Config':<10} | {'AIC':<10} | {'RMSE':<10} | {'Tiempo'}")
    print("-" * 75)
    
    # 2. Calcular los Manuales
    evaluar(train, test, 1, 1, 1, "Manual (Simple)")
    evaluar(train, test, 2, 1, 2, "Manual (Complejo)")
    evaluar(train, test, 3, 1, 0, "Manual (Solo AR)")
    
    print("-" * 75)
    print("⚠️  PARA EL AGENTE: Mira tu web (localhost:8501) y anota lo que eligió.")
    print("-" * 75)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


try:
    df = pd.read_csv('data/germany_monthly_power.csv', index_col=0, parse_dates=True)
except:
    df = pd.read_csv('../data/germany_monthly_power.csv', index_col=0, parse_dates=True)

train = df.iloc[:-6]['load_gwh']
test = df.iloc[-6:]['load_gwh']


print("Calculando métricas del Agente...")
model = ARIMA(train, order=(4, 0, 2))
fit = model.fit()
pred = fit.forecast(steps=len(test))
rmse = np.sqrt(mean_squared_error(test, pred))
print(f"Agente (4,0,2) -> AIC: {fit.aic:.2f} | RMSE: {rmse:.2f}")