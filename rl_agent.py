#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: rl_agent.py
# Descripción: Agente PPO Robustecido + Filtro Anti-Infinitos
# ============================================================================

import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym  
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from src.arima_env import ARIMAHyperparamEnv
from src.data_processor import TimeSeriesProcessor


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class SanitizeObsWrapper(gym.ObservationWrapper):
    """
    Filtro de Seguridad:
    Intercepta las observaciones del entorno antes de que lleguen al agente.
    Si encuentra 'Infinito' (porque no hay métricas previas), lo cambia por un número fijo.
    Esto evita que la Red Neuronal colapse con NaNs.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def observation(self, obs):

        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

class ARIMAAgent:
    """
    Agente RL PPO para optimización ARIMA.
    """

    def __init__(self, train_data, val_data, config=None):
        self.train_data = train_data
        self.val_data = val_data

        
        if config is None:
            self.config = {
                'p_max': 5,
                'd_max': 2,
                'q_max': 4,
                'max_steps': 50,
                'learning_rate': 1e-4,  
                'n_steps': 128,         
                'batch_size': 32,
                'n_epochs': 5,
                'gamma': 0.95,
                'policy_kwargs': {
                    'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
                }
            }
        else:
            self.config = config

        self.env = self._create_env()
        self.model = None

    def _create_env(self):
        """
        Crea el entorno con múltiples capas de protección (Wrappers).
        """
        
        env = ARIMAHyperparamEnv(
            train_data=self.train_data,
            val_data=self.val_data,
            p_max=self.config['p_max'],
            d_max=self.config['d_max'],
            q_max=self.config['q_max'],
            max_steps=self.config['max_steps']
        )

        
        env = SanitizeObsWrapper(env)

        
        env = Monitor(env)

        
        env = DummyVecEnv([lambda: env])

        
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        return env

    def train(self, total_timesteps=1000, save_path='models/arima_dqn_agent',
              tensorboard_log='models/tensorboard_logs', save_freq=5000):
        
        print("\n" + "=" * 80)
        print("🚀 INICIANDO ENTRENAMIENTO (PPO + SANITIZACIÓN)")
        print("=" * 80)

        # Crear directorios
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(tensorboard_log, exist_ok=True)

        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            policy_kwargs=self.config['policy_kwargs'],
            tensorboard_log=tensorboard_log,
            verbose=1,
            device='cpu' 
        )

        print(f"\n✅ Modelo PPO inicializado")

        # Entrenar
        print(f"\n🏃 Entrenando agente...")
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)

        # Guardar modelo
        self.model.save(save_path)
        self.env.save(save_path + "_vecnormalize.pkl")
        
        print(f"\n✅ Modelo guardado en: {save_path}.zip")
        return self.model

    def load(self, model_path='models/arima_dqn_agent'):
        """Carga modelo y entorno normalizado"""
        print(f"📂 Cargando modelo desde: {model_path}")
        
        
        stats_path = model_path + "_vecnormalize.pkl"
        
        
        if model_path.endswith('.zip'):
             stats_path = model_path.replace('.zip', '_vecnormalize.pkl')
        
        if os.path.exists(stats_path):
            self.env = VecNormalize.load(stats_path, self.env)
            self.env.training = False 
            self.env.norm_reward = False
            print("   ✅ Estadísticas de normalización cargadas")
        else:
            print("   ⚠️ No se encontraron estadísticas de normalización (usando defaults)")

        
        if not model_path.endswith('.zip'):
            model_path += ".zip"
            
        self.model = PPO.load(model_path, env=self.env)
        print("✅ Modelo cargado exitosamente")
        return self.model

    def predict_best_config(self):
        if self.model is None:
            raise ValueError("Modelo no cargado.")

        
        obs = self.env.reset()
        
        
        action, _ = self.model.predict(obs, deterministic=True)
        
        
        if isinstance(action, np.ndarray):
            if len(action.shape) > 1:
                p, d, q = action[0]
            else:
                p, d, q = action
        else:
             p, d, q = 1, 1, 1 

        print(f"\n🤖 Agente predice: (p={p}, d={d}, q={q})")
        return (int(p), int(d), int(q))

# ============================================================================
# UTILS
# ============================================================================

def train_agent_from_file(data_path='data/germany_monthly_power.csv',
                          timesteps=1000,
                          output_dir='models'):
    
    print("📂 Cargando datos...")
    processor = TimeSeriesProcessor(data_path)
    processor.load_data()
    processor.split_data()

    train_data = processor.train['value'].values
    val_data = processor.val['value'].values

    
    agent = ARIMAAgent(train_data, val_data)
    
    save_path = os.path.join(output_dir, 'arima_dqn_agent')
    tensorboard_log = os.path.join(output_dir, 'tensorboard_logs')

    agent.train(total_timesteps=timesteps, save_path=save_path, tensorboard_log=tensorboard_log)
    
    return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--data', type=str, default='data/germany_monthly_power.csv')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='models')
    
    args = parser.parse_args()

    if args.train:
        train_agent_from_file(args.data, args.timesteps, args.output_dir)

if __name__ == "__main__":
    main()