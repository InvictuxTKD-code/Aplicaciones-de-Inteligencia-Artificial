# Memorice con Agente A* 🎮

Este proyecto implementa el clásico juego de **Memorice (Concentration)** 
con un agente automático que utiliza el algoritmo de búsqueda **A*** 
para resolverlo.

## 🚀 Características
- Tablero 6x6 generado aleatoriamente.
- Visualización interactiva en **Pygame**.
- Agente con memoria parcial:
  - Usa **A*** para planificar jugadas.
  - Política *greedy* como respaldo si A* se estanca.
- Estadísticas finales: número de turnos y tiempo de ejecución.

## 📂 Estructura del código
- `MemoriceGame`: lógica principal del juego.
- `astar_plan`: implementación del algoritmo A*.
- `greedy_policy`: política de respaldo.
- `draw_board`: renderizado visual en Pygame.
- `main`: bucle principal del juego.

## 🧩 Algoritmo
El agente utiliza una búsqueda A* donde:
- El estado es una tupla con las cartas emparejadas y la memoria de las vistas.
- La heurística corresponde al número de pares faltantes por resolver.
- Se priorizan jugadas con pares ya conocidos antes de explorar nuevas cartas.

## ⏱️ Resultados
En promedio, el agente resuelve un tablero de 6x6 en:
- **X turnos** (según ejecución).
- **Y segundos** en máquina de prueba.

## 🛠️ Requisitos
- Python 3.10+
- `pygame`

Instalación:
```bash
pip install pygame
```
Proyecto desarrollado por **Andrés Jaramillo** como parte de la evaluación de algoritmos de búsqueda.

