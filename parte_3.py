"""
Tarea 2 - Parte 3: PPO en Cooperative Pong v5 (PettingZoo)

Objetivo:
  Adaptar el código de KAZ a otro ambiente cooperativo de PettingZoo:
  'butterfly/cooperative_pong_v5' y reportar la recompensa alcanzada
  por el sistema multiagente.

Ejecución:
  python parte_3.py
"""

from __future__ import annotations

import glob
import os
import time
from typing import Tuple, Dict

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from pettingzoo.butterfly import cooperative_pong_v5


def _is_visual(env) -> bool:
    """Detecta si el entorno es visual (no vector_state). Más robusto que asumir atributo."""
    try:
        return not bool(getattr(env.unwrapped, "vector_state", False))
    except Exception:
        # Si no existe el atributo, asumimos que es visual (cooperative_pong lo es).
        return True


def train(env_fn, steps: int = 100_000, seed: int | None = 0, **env_kwargs) -> str:
    """
    Entrena PPO en un entorno paralelo de PettingZoo con wrappers de SuperSuit
    y devuelve la ruta del modelo guardado.
    """
    env = env_fn.parallel_env(**env_kwargs)

    # Mantener número constante de agentes si aplica
    # (cooperative_pong no tiene 'muertes', pero black_death no daña)
    env = ss.black_death_v3(env)

    # Pre-proceso con SuperSuit (visual)
    visual_observation = _is_visual(env)
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])} with kwargs={env_kwargs}.")

    # Vectorizar para SB3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Política según tipo de observación
    policy_cls = CnnPolicy if visual_observation else MlpPolicy
    model = PPO(policy_cls, env, verbose=3, batch_size=256)

    model.learn(total_timesteps=steps)

    model_path = f"{env_fn.__name__}_ppo_{steps}_steps_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_path)
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()
    return model_path


def eval(env_fn, num_games: int = 20, render_mode: str | None = None,
         model_path: str | None = None, **env_kwargs) -> Tuple[float, Dict[str, float]]:
    """
    Evalúa un modelo entrenado por num_games y retorna:
      - promedio global de recompensa por agente
      - promedio por agente (dict)
    """
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    visual_observation = _is_visual(env)
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(f"\nStarting evaluation on {str(env.metadata['name'])} "
          f"(num_games={num_games}, render_mode={render_mode}) with kwargs={env_kwargs}")

    try:
        if model_path is None:
            latest_policy = max(glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime)
        else:
            cand = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
            if os.path.exists(cand):
                latest_policy = cand
            elif os.path.exists(model_path.rstrip(".zip")):
                latest_policy = model_path.rstrip(".zip")
            else:
                # fallback: toma el .zip más reciente que empiece con el nombre del env
                candidates = glob.glob(f"{env.metadata['name']}*.zip")
                if not candidates:
                    raise ValueError("No se encontraron modelos .zip coincidentes.")
                latest_policy = max(candidates, key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        raise SystemExit(1)

    model = PPO.load(latest_policy)

    rewards = {agent: 0.0 for agent in env.possible_agents}

    # AEC para evaluación
    for i in range(num_games):
        env.reset(seed=i)
        # Estrategia: primer agente aleatorio como baseline de cooperación mínima
        if env.possible_agents:
            env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # acumular recompensas de todos los agentes en este paso
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

    avg_reward = sum(rewards.values()) / max(len(rewards), 1)
    avg_reward_per_agent = {a: r / num_games for a, r in rewards.items()}
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards (acumuladas): ", rewards)
    return float(avg_reward), {k: float(v) for k, v in avg_reward_per_agent.items()}


if __name__ == "__main__":
    env_fn = cooperative_pong_v5
    
    env_kwargs = dict(
        max_cycles=500,   # episodios más largos que KAZ
        # opponent_policy=None,        # <-- REMOVIDO (#MODIFICACION)
        # bounce_randomness=False,     # (si esto también diera error, quítalo)
    )

    STEPS = 100_000
    SEED = 0

    print("\n================= ENTRENAMIENTO PARTE 3 =================")
    model_path = train(env_fn, steps=STEPS, seed=SEED, **env_kwargs)

    print("\n================= EVALUACION PARTE 3 =================")
    avg_reward, avg_reward_per_agent = eval(
        env_fn,
        num_games=20,
        render_mode=None,
        model_path=model_path,
        **env_kwargs
    )

    print("\n================= RESUMEN PARTE 3 =================")
    print(f"Recompensa promedio (20 juegos): {avg_reward:.4f}")
    for k, v in avg_reward_per_agent.items():
        print(f"  {k}: {v:.4f}")