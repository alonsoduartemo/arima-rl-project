#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: download_data.py
# Descripción: Descarga y preprocesa datos OPSD de consumo eléctrico alemán
#              Genera 60 meses de datos (2013-2017) con división train/val/test
# ============================================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
import requests

class DataDownloader:
    """Descarga y preprocesa datos de consumo eléctrico alemán de OPSD."""
    
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def download_opsd_data(self):
        """
        Intenta descargar datos reales de OPSD.
        Si falla, genera datos sintéticos realistas.
        """
        print("=" * 80)
        print("📥 DESCARGANDO DATOS DE OPSD (Open Power System Data)")
        print("=" * 80)
        
        # URL de datos OPSD (time series data package)
        url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
        
        try:
            print(f"\n🌐 Intentando descargar desde: {url}")
            print("   (Esto puede tardar 1-2 minutos...)")
            
            # Descargar con timeout
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Guardar CSV temporalmente
            temp_file = os.path.join(self.output_dir, 'opsd_raw.csv')
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            print("✅ Descarga exitosa!")
            
            # Leer y procesar
            df = pd.read_csv(temp_file, parse_dates=['utc_timestamp'])
            
            # Filtrar datos de Alemania (columna DE_load_actual_entsoe_transparency)
            if 'DE_load_actual_entsoe_transparency' in df.columns:
                df_germany = df[['utc_timestamp', 'DE_load_actual_entsoe_transparency']].copy()
                df_germany.columns = ['timestamp', 'load_mw']
                df_germany = df_germany.dropna()
                
                # Convertir a GWh (desde MW)
                df_germany['load_gwh'] = df_germany['load_mw'] / 1000
                
                print(f"✅ Datos de Alemania extraídos: {len(df_germany)} registros horarios")
                
                # Limpiar archivo temporal
                os.remove(temp_file)
                
                return df_germany
            else:
                print("⚠️  Columna de Alemania no encontrada. Generando datos sintéticos...")
                return self.generate_synthetic_data()
                
        except Exception as e:
            print(f"⚠️  Error al descargar datos: {e}")
            print("📊 Generando datos sintéticos realistas como alternativa...")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """
        Genera datos sintéticos realistas de consumo eléctrico alemán.
        Simula: tendencia + estacionalidad + ruido
        """
        print("\n📊 Generando datos sintéticos de consumo eléctrico alemán...")
        
        # Generar 60 meses de datos horarios (2013-2017)
        date_range = pd.date_range(start='2013-01-01', end='2017-12-31 23:00:00', freq='H')
        
        # Número de horas
        n_hours = len(date_range)
        t = np.arange(n_hours)
        
        # Componentes del modelo
        # 1. Tendencia lineal leve (crecimiento del consumo)
        trend = 50 + 0.0001 * t
        
        # 2. Estacionalidad anual (verano bajo, invierno alto)
        annual_seasonality = 10 * np.sin(2 * np.pi * t / (365.25 * 24) + np.pi/2)
        
        # 3. Estacionalidad semanal (fines de semana bajo)
        weekly_seasonality = 5 * np.sin(2 * np.pi * t / (7 * 24))
        
        # 4. Estacionalidad diaria (noche bajo, día alto)
        daily_seasonality = 8 * np.sin(2 * np.pi * t / 24 + np.pi/2)
        
        # 5. Ruido aleatorio
        np.random.seed(42)
        noise = np.random.normal(0, 2, n_hours)
        
        # Combinar componentes
        load_gwh = trend + annual_seasonality + weekly_seasonality + daily_seasonality + noise
        
        # Asegurar valores positivos y realistas (35-75 GWh típico para Alemania)
        load_gwh = np.clip(load_gwh, 35, 75)
        
        df = pd.DataFrame({
            'timestamp': date_range,
            'load_gwh': load_gwh
        })
        
        print(f"✅ {len(df)} registros horarios sintéticos generados")
        print(f"   Rango de fechas: {df['timestamp'].min()} a {df['timestamp'].max()}")
        print(f"   Rango de carga: {df['load_gwh'].min():.2f} - {df['load_gwh'].max():.2f} GWh")
        
        return df
    
    def convert_to_monthly(self, df):
        """
        Convierte datos horarios a mensuales (suma total mensual).
        """
        print("\n🔄 Convirtiendo datos horarios a mensuales...")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resampleo mensual (suma total del mes)
        df_monthly = df.resample('MS').sum()
        
        # CORRECCIÓN: Tomar los primeros 60 meses disponibles (independiente del año)
        if len(df_monthly) >= 60:
            df_monthly = df_monthly.iloc[:60]
        
        print(f"✅ {len(df_monthly)} meses de datos generados")
        print(f"   Período: {df_monthly.index[0].strftime('%Y-%m')} a {df_monthly.index[-1].strftime('%Y-%m')}")
        print(f"   Consumo promedio mensual: {df_monthly['load_gwh'].mean():.2f} GWh")
        
        return df_monthly
    
    def split_train_val_test(self, df):
        """
        Divide datos en conjuntos train/val/test según especificaciones:
        - Train: 48 meses (80%)
        - Validation: 6 meses (10%)
        - Test: 6 meses (10%)
        """
        print("\n✂️  Dividiendo datos en train/val/test...")
        
        # Verificar que tenemos 60 meses
        assert len(df) == 60, f"Error: Se esperaban 60 meses, se obtuvieron {len(df)}"
        
        # División
        train = df.iloc[:48]      # Primeros 48 meses
        val = df.iloc[48:54]      # Siguientes 6 meses
        test = df.iloc[54:60]     # Últimos 6 meses
        
        print(f"✅ División completada:")
        print(f"   📚 Train:      {len(train)} meses ({train.index[0].strftime('%Y-%m')} a {train.index[-1].strftime('%Y-%m')})")
        print(f"   🔍 Validation: {len(val)} meses ({val.index[0].strftime('%Y-%m')} a {val.index[-1].strftime('%Y-%m')})")
        print(f"   🧪 Test:       {len(test)} meses ({test.index[0].strftime('%Y-%m')} a {test.index[-1].strftime('%Y-%m')})")
        
        return train, val, test
    
    def save_data(self, df_monthly, train, val, test):
        """
        Guarda todos los datasets en archivos CSV.
        """
        print("\n💾 Guardando archivos CSV...")
        
        # Archivo principal (60 meses completos)
        main_file = os.path.join(self.output_dir, 'germany_monthly_power.csv')
        df_monthly.to_csv(main_file)
        print(f"   ✅ {main_file}")
        
        # Conjuntos individuales
        train_file = os.path.join(self.output_dir, 'train.csv')
        train.to_csv(train_file)
        print(f"   ✅ {train_file}")
        
        val_file = os.path.join(self.output_dir, 'validation.csv')
        val.to_csv(val_file)
        print(f"   ✅ {val_file}")
        
        test_file = os.path.join(self.output_dir, 'test.csv')
        test.to_csv(test_file)
        print(f"   ✅ {test_file}")
        
        # Archivo de metadatos
        metadata = {
            'total_months': len(df_monthly),
            'train_months': len(train),
            'val_months': len(val),
            'test_months': len(test),
            'date_range': f"{df_monthly.index[0].strftime('%Y-%m')} to {df_monthly.index[-1].strftime('%Y-%m')}",
            'mean_consumption_gwh': float(df_monthly['load_gwh'].mean()),
            'std_consumption_gwh': float(df_monthly['load_gwh'].std()),
            'min_consumption_gwh': float(df_monthly['load_gwh'].min()),
            'max_consumption_gwh': float(df_monthly['load_gwh'].max()),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = os.path.join(self.output_dir, 'metadata.txt')
        with open(metadata_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"   ✅ {metadata_file}")
        
        print("\n🎉 Todos los archivos guardados exitosamente!")
        
    def run(self):
        """
        Ejecuta el pipeline completo de descarga y procesamiento de datos.
        """
        print("\n" + "=" * 80)
        print("🚀 INICIANDO PIPELINE DE DATOS")
        print("=" * 80)
        
        # Paso 1: Descargar/generar datos
        df_hourly = self.download_opsd_data()
        
        # Paso 2: Convertir a mensual
        df_monthly = self.convert_to_monthly(df_hourly)
        
        # Paso 3: Dividir en train/val/test
        train, val, test = self.split_train_val_test(df_monthly)
        
        # Paso 4: Guardar archivos
        self.save_data(df_monthly, train, val, test)
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print("\n📊 Archivos generados en directorio 'data/':")
        print("   - germany_monthly_power.csv  (60 meses completos)")
        print("   - train.csv                  (48 meses)")
        print("   - validation.csv             (6 meses)")
        print("   - test.csv                   (6 meses)")
        print("   - metadata.txt               (información del dataset)")
        print("\n🎯 Siguiente paso: Entrenar agente RL o ejecutar aplicación Streamlit")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.run()
