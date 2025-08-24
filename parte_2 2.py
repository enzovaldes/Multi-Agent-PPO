"""
Tarea 2 - Parte 2: Variación de cantidad de arqueros y caballeros en KAZ (PettingZoo)

Este script reutiliza el de la Parte 1 y ejecuta los 4 escenarios pedidos:
  2.1  Doble de arqueros y doble de caballeros
  2.2  Doble de arqueros, caballeros por defecto
  2.3  Doble de caballeros, arqueros por defecto
  2.4  Mitad de arqueros y mitad de caballeros

Luego imprime un resumen y una conclusión para 2.5:
  ¿Qué tipo de agente es más efectivo para repeler zombies?
"""

from __future__ import annotations

import glob
import os
import time
from typing import Tuple, Dict

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from pettingzoo.butterfly import knights_archers_zombies_v10

# ============================
# Reutilización de funciones
# ============================

def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs) -> str:
    # Entrena un único modelo que juega como cada agente (entorno AEC -> vectorizado)
    env = env_fn.parallel_env(**env_kwargs)

    # Mantener número constante de agentes
    env = ss.black_death_v3(env)

    # Pre-proceso con SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])} with kwargs={env_kwargs}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Política según tipo de observación
    model = PPO(CnnPolicy if visual_observation else MlpPolicy, env, verbose=3, batch_size=256)
    model.learn(total_timesteps=steps)

    # MODIFICACION: nombre lleva steps y configuración
    tag = f"na{env_kwargs.get('num_archers')}_nk{env_kwargs.get('num_knights')}"
    model_path = f"{env.unwrapped.metadata.get('name')}_ppo_{steps}_steps_{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_path)
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()
    return model_path  # MODIFICACION: devolver ruta del modelo


def eval(env_fn, num_games: int = 20, render_mode: str | None = None, model_path: str | None = None, **env_kwargs) -> Tuple[float, Dict[str, float]]:
    # Evalúa un modelo entrenado vs un agente aleatorio en AEC
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode}) with kwargs={env_kwargs}")

    try:
        if model_path is None:
            latest_policy = max(glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime)
        else:
            latest_policy = model_path + ".zip"
    except ValueError:
        print("Policy not found.")
        raise SystemExit(1)

    model = PPO.load(latest_policy)

    rewards = {agent: 0.0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards)
    avg_reward_per_agent = {agent: r / num_games for agent, r in rewards.items()}
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards (acumuladas): ", rewards)
    return float(avg_reward), {k: float(v) for k, v in avg_reward_per_agent.items()}


# ============================
# Bloque principal Parte 2
# ============================

if __name__ == "__main__":
    env_fn = knights_archers_zombies_v10

    # Base de la config (igual a Parte 1 salvo num_archers/num_knights que iremos cambiando)
    base_kwargs = dict(
        max_cycles=100,
        max_zombies=4,
        vector_state=True,   # observación vectorial (más rápido)
        # num_archers y num_knights se setean por escenario
    )

    # MODIFICACION: usar los mejores steps de la Parte 1
    BEST_STEPS = 100_000

    # Valores por defecto del entorno KAZ: num_archers=2, num_knights=2
    # MODIFICACION: escenarios Parte 2
    scenarios = {
        "2.1_doble_ambos": dict(num_archers=4, num_knights=4),
        "2.2_doble_archers": dict(num_archers=4, num_knights=2),
        "2.3_doble_knights": dict(num_archers=2, num_knights=4),
        "2.4_mitad_ambos": dict(num_archers=1, num_knights=1),
    }

    resultados = []

    for name, overrides in scenarios.items():
        env_kwargs = {**base_kwargs, **overrides}
        print(f"\n================= ESCENARIO {name} | {env_kwargs} =================")
        model_path = train(env_fn, steps=BEST_STEPS, seed=0, **env_kwargs)
        avg_reward, avg_reward_per_agent = eval(
            env_fn,
            num_games=20,
            render_mode=None,
            model_path=model_path,
            **env_kwargs
        )

        # MODIFICACION: sumar por tipo de agente para 2.5
        archer_keys = [k for k in avg_reward_per_agent.keys() if k.startswith("archer_")]
        knight_keys = [k for k in avg_reward_per_agent.keys() if k.startswith("knight_")]
        archer_avg_sum = sum(avg_reward_per_agent[k] for k in archer_keys)
        knight_avg_sum = sum(avg_reward_per_agent[k] for k in knight_keys)

        resultados.append({
            "escenario": name,
            "kwargs": env_kwargs,
            "avg_reward_global": avg_reward,
            "avg_reward_archers_sum": archer_avg_sum,
            "avg_reward_knights_sum": knight_avg_sum,
            "detalle_por_agente": avg_reward_per_agent
        })

    # ============================
    # Resumen + Conclusión 2.5
    # ============================
    print("\n================= RESUMEN PARTE 2 =================")
    for r in resultados:
        print(f"- {r['escenario']} | global={r['avg_reward_global']:.4f} | archers_sum={r['avg_reward_archers_sum']:.4f} | knights_sum={r['avg_reward_knights_sum']:.4f} | cfg={r['kwargs']}")

    # Mejor escenario por recompensa global
    best = max(resultados, key=lambda x: x["avg_reward_global"])
    # ¿Quién sumó más recompensa por tipo de agente en el mejor escenario?
    tipo_mas_efectivo = "arqueros" if best["avg_reward_archers_sum"] >= best["avg_reward_knights_sum"] else "caballeros"

    print("\n>>> CONCLUSION 2.5")
    print(f"Mejor escenario: {best['escenario']} con recompensa global promedio {best['avg_reward_global']:.4f}.")
    print(f"En el mejor escenario, el tipo de agente más efectivo (mayor recompensa agregada) fue: {tipo_mas_efectivo.upper()}.")
    print("Revisa 'detalle_por_agente' por escenario para ver contribuciones individuales.")