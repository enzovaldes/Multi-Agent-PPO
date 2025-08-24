"""
Tarea 2 - Parte 1: Entrenamiento por cantidad de steps en KAZ (PettingZoo)

Este script entrena PPO en el entorno Knights-Archers-Zombies (KAZ) y evalúa el desempeño
después de cada entrenamiento, de acuerdo a lo solicitado en la Parte 1.

Ejecuta los siguientes experimentos:
  - 50,000 steps
  - 100,000 steps
  - 150,000 steps
  - 200,000 steps

Para cada modelo entrenado:
  - Evalúa 20 juegos (episodios)
  - Calcula e imprime la recompensa promedio global y el promedio por agente
  - Guarda cada política con el número de steps en el nombre del archivo

Al final, imprime un resumen y responde explícitamente:
  - ¿Hay un cambio en la recompensa promedio al entrenar más tiempo?
  - ¿Cuál es la máxima recompensa promedio obtenida y con cuántos steps?

Notas:
  - Está pensado para ejecución local con Python 3.10.18.
  - Usa observación vectorial (vector_state=True) para entrenar más rápido.
  - Si se desea visualizar partidas, cambiar render_mode=None por "human" (requiere pygame).
"""

#librerias
from __future__ import annotations

import glob
import os
import time
from typing import Tuple, Dict

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from pettingzoo.butterfly import knights_archers_zombies_v10


def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs) -> str:
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state 
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Use a CNN policy if the observation space is visual
    model = PPO(
        CnnPolicy if visual_observation else MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    # MODIFICACION: guardar el modelo con steps en el nombre y devolver la ruta
    model_path = f"{env.unwrapped.metadata.get('name')}_ppo_{steps}_steps_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_path)
    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    return model_path  # MODIFICACION: devolver path explícitamente


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, model_path: str | None = None, **env_kwargs) -> Tuple[float, Dict[str, float]]:
    # MODIFICACION: nuevo parámetro model_path para evaluar el modelo específico entrenado recién
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        if model_path is None:  # compatibilidad con comportamiento original
            latest_policy = max(
                glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
            )
        else:
            latest_policy = model_path + ".zip"  # MODIFICACION: usar el modelo recién guardado
    except ValueError:
        print("Policy not found.")
        raise SystemExit(1)

    model = PPO.load(latest_policy)

    rewards = {agent: 0.0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
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

    avg_reward = sum(rewards.values()) / len(reards := rewards.values())  # evita recalcular len
    avg_reward_per_agent = {agent: r / num_games for agent, r in rewards.items()}
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return float(avg_reward), {k: float(v) for k, v in avg_reward_per_agent.items()}  # MODIFICACION: devolver métricas para el resumen


if __name__ == "__main__":
    env_fn = knights_archers_zombies_v10

    # Set vector_state to false in order to use visual observations (significantly longer training time)
    env_kwargs = dict(max_cycles=100, max_zombies=4, vector_state=True)

    # MODIFICACION: lista de steps solicitados por la Parte 1
    steps_list = [50_000, 100_000, 150_000, 200_000]

    resultados = []  # MODIFICACION: almacenar resultados de cada corrida

    for steps in steps_list:  # MODIFICACION: entrenar y evaluar en loop
        print(f"\n================= EXPERIMENTO: {steps:,} steps =================")
        model_path = train(env_fn, steps=steps, seed=0, **env_kwargs)
        avg_reward, avg_reward_per_agent = eval(
            env_fn,
            num_games=20,                 # MODIFICACION: 20 juegos
            render_mode=None,             # en local sin GUI; usa "human" si quieres ver partidas
            model_path=model_path,
            **env_kwargs
        )
        resultados.append({
            "steps": steps,
            "model_path": model_path,
            "avg_reward": avg_reward,
            "avg_reward_per_agent": avg_reward_per_agent
        })

    # MODIFICACION: resumen final para responder la pregunta 1
    print("\n================= RESUMEN PARTE 1 =================")
    for r in resultados:
        print(f"- {r['steps']:,} steps | avg_reward={r['avg_reward']:.4f} | model={os.path.basename(r['model_path'])}.zip")

    # detectar la máxima recompensa promedio
    best = max(resultados, key=lambda x: x["avg_reward"])
    print(f"\n¿Hay cambio en la recompensa promedio con más steps? -> Observa la tendencia anterior.")
    print(f"Máxima recompensa promedio: {best['avg_reward']:.4f} obtenida con {best['steps']:,} steps (modelo {os.path.basename(best['model_path'])}.zip)")

    