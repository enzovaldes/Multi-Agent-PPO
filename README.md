# SIMULACIÓN BASADA EN AGENTES Sec.2 FIC

# Alumnos:

# Enzo Valdés
# Jorge Martinez
# Cesar Bustamantes
# Cristian Arroyo

# Parte 1: Multi-Agent PPO en Knights-Archers-Zombies (PettingZoo)

## Contexto
El código se ejecuta **en local** usando **Python 3.10.18** y reproduce el notebook de clase adaptado a los enunciados propuestos.  
Se entrena PPO en knights_archers_zombies_v10 y se evalúa con 20 juegos tras cada entrenamiento.

---

## Requisitos

- **Python**: 3.10.18
- Pre-requisitos y modo de ejecución.
  - `python3.10 -m venv venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`

- **Ejecución**

  - `python parte_1.py`

---

## 1. Primera Pregunta:

Los experimentos arrojaron los siguientes promedios de recompensa:

- Con **50,000 pasos de entrenamiento** el promedio fue **2.00**.  
- Con **100,000 pasos de entrenamiento** el promedio fue **8.00**.  
- Con **150,000 pasos de entrenamiento** el promedio fue **2.50**.  
- Con **200,000 pasos de entrenamiento** el promedio fue **6.25**.  

En los resultados detallados por agente se observó que las recompensas provienen casi exclusivamente de los **arqueros**, mientras que los **caballeros** se mantuvieron con valores nulos.

**Preguntas:**
- ¿Hay un cambio en la recompensa promedio al entrenar más tiempo?  
  Sí. A medida que se incrementan los pasos de entrenamiento, la recompensa promedio varía, pero no de forma lineal ni estable. Se observa que al llegar a **100,000 pasos** el rendimiento del modelo mejora de manera clara, pero después de ese punto los resultados fluctúan e incluso disminuyen en algunos casos (150k y 200k pasos).  

- ¿Cuál es la máxima recompensa promedio alcanzada?  
  La máxima recompensa promedio fue de **8.00**, alcanzada con **100,000 pasos** de entrenamiento.  

---

# Parte 2: Variación de cantidad de arqueros y caballeros en KAZ (PettingZoo)

Este experimento reutiliza el código de la Parte 1 y evalúa el desempeño del sistema multiagente (PPO) en **Knights-Archers-Zombies (KAZ)** al **modificar la cantidad de arqueros y caballeros**, usando como presupuesto de entrenamiento los **100,000 steps** identificados como mejores en la Parte 1.  
Tras cada entrenamiento se **evalúan 20 juegos** y se calcula la recompensa promedio.

- **Ejecución**

	- `python parte_2.py ` 

Se entrenó y evaluó en los 4 escenarios solicitados:

	1.	2.1 – Doble de arqueros y doble de caballeros

	•	Recompensa promedio global: 2.75
	•	Suma promedio arqueros: 1.10
	•	Suma promedio caballeros: 0.00
	
	2.	2.2 – Doble de arqueros; caballeros por defecto

	•	Recompensa promedio global: 5.00
	•	Suma promedio arqueros: 1.35
	•	Suma promedio caballeros: 0.15
	
	3.	2.3 – Doble de caballeros; arqueros por defecto

	•	Recompensa promedio global: 4.00
	•	Suma promedio arqueros: 1.20
	•	Suma promedio caballeros: 0.00
	
	4.	2.4 – Mitad de arqueros y mitad de caballeros

	•	Recompensa promedio global: 8.00
	•	Suma promedio arqueros: 0.80
	•	Suma promedio caballeros: 0.00

Conclusiones:

	•	Duplicar ambos tipos de agentes (2.1) no genera un buen resultado.
	•	Duplicar solo arqueros (2.2) mejora el desempeño respecto a 2.1.
	•	Duplicar solo caballeros (2.3) no aporta valor adicional y es inferior a duplicar arqueros.
	•	Reducir a la mitad ambos tipos (2.4) entrega el mejor resultado global con 8.00.

**Preguntas:**

2.5 – ¿Qué tipo de agente es más efectivo para repeler zombies?

En todos los escenarios los arqueros concentran prácticamente toda la recompensa, mientras que los caballeros mantienen valores nulos o muy bajos.
Por lo tanto, los arqueros son los agentes más efectivos para repeler zombies.

---

# Parte 3: PPO en Cooperative Pong v5 (PettingZoo)

Este experimento adapta el pipeline de entrenamiento de la Parte 1 al entorno cooperativo **Cooperative Pong v5** de PettingZoo.  
Se entrena un modelo PPO con observación visual (grises + resize 84×84 + frame stacking) y se evalúa en **20 juegos** para medir la recompensa alcanzada por el sistema multiagente.

- **Ejecución**

  - `python parte_3.py`

  	•	Recompensa promedio global (20 juegos): 41.40
	•	Promedio por agente, por juego:
	•	paddle_0: 2.07
	•	paddle_1: 2.07

**Preguntas:**

Se observa que ambos agentes cooperativos (las dos paletas) reciben recompensas similares, lo que indica que el modelo logra un comportamiento simétrico en el entorno.

Conclusión – Parte 3

El modelo PPO entrenado en Cooperative Pong v5 alcanzó un desempeño promedio de 41.4 puntos acumulados en 20 juegos, distribuidos equitativamente entre los dos agentes.
Esto refleja un comportamiento cooperativo efectivo en el que ambos agentes contribuyen de manera balanceada al objetivo de mantener la pelota en juego. Ambos agentes (paddle_0 y paddle_1) recibieron prácticamente la misma recompensa, lo que muestra que el comportamiento aprendido fue cooperativo y simétrico.