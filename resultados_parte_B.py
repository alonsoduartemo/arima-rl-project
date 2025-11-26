
import pandas as pd
import numpy as np
import time
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

def evaluar(train, test, p, d, q, nombre):
    try:
        start = time.time()
        model = ARIMA(train, order=(p, d, q))
        fit = model.fit()
        pred = fit.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, pred))
        aic = fit.aic
        duracion = time.time() - start
        
        print(f"{nombre:<25} | ({p},{d},{q})      | {aic:<10.2f} | {rmse:<10.2f} | {duracion:.4f}")
    except:
        print(f"{nombre:<25} | ({p},{d},{q})      | ERROR      | ERROR      | ----")

def main():
    print("\n🚀 GENERANDO DATOS PARA EXPERIMENTO B (Variando d)...")
    
    
    try:
        df = pd.read_csv('data/germany_monthly_power.csv', index_col=0, parse_dates=True)
    except:
        df = pd.read_csv('../data/germany_monthly_power.csv', index_col=0, parse_dates=True)
        
    train = df.iloc[:-6]['load_gwh']
    test = df.iloc[-6:]['load_gwh']

    
    p_agente = 4
    q_agente = 2

    print("-" * 80)
    print(f"{'Condición':<25} | {'Config':<10} | {'AIC':<10} | {'RMSE':<10} | {'Tiempo'}")
    print("-" * 80)
    
    # 1. Sin Diferenciar (d=0)
    evaluar(train, test, p_agente, 0, q_agente, "Sin Diferenciar")
    
    # 2. Diferenciación Estándar (d=1) -> ESTE ES EL DE TU TABLA ANTERIOR
    evaluar(train, test, p_agente, 1, q_agente, "Diferenciación Estándar")
    
    # 3. Sobre-diferenciación (d=2)
    evaluar(train, test, p_agente, 2, q_agente, "Sobre-diferenciación")
    
    print("-" * 80)

if __name__ == "__main__":
    main()