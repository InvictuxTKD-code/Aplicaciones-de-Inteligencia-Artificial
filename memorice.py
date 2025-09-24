"""
Memorice con Agente A* (Pygame)
--------------------------------
Este programa implementa un juego de Memorice (Memory / Concentration) de 6x6
donde un agente automático resuelve el tablero usando el algoritmo de búsqueda A*.
Si A* no logra encontrar una solución completa dentro de los límites de expansión,
se utiliza una política greedy de respaldo.

Características:
- Representación visual del tablero usando Pygame.
- El agente recuerda cartas vistas y busca pares conocidos.
- El algoritmo A* genera un plan de jugadas óptimo aproximado.
- Se muestran estadísticas finales: número de turnos y tiempo total.

Requisitos:
- Python 3.10+
- Librerías: pygame, collections, heapq, random, time

Ejecución:
    python memorice_pygame_astar.py
"""
# memorice_pygame_astar.py
import pygame
import random
import time
import heapq
from collections import defaultdict, namedtuple

# -------------------------
# Configuración del tablero y ventana
# -------------------------
BOARD_SIZE = 6                           # Número de filas y columnas del tablero (6x6)
CELL = 80                                # Tamaño de cada celda en pixeles (ancho y alto)
MARGIN = 8                               # Margen en pixeles alrededor del tablero y entre celdas
WIDTH = BOARD_SIZE * CELL + 2 * MARGIN   # Ancho de la ventana: tablero + márgenes laterales
HEIGHT = BOARD_SIZE * CELL + 120         # Alto de la ventana: tablero + espacio extra para mostrar info
FPS = 60                                 # Frames per second: velocidad de actualización de la ventana

# -------------------------
# Colores (RGB) usados en el juego
# -------------------------
BG = (18, 18, 24)               # Color de fondo de la ventana
CARD_BACK = (40, 40, 60)        # Color de las cartas cuando están boca abajo
CARD_FRONT = (230, 230, 230)    # Color de las cartas cuando se revelan
TEXT_COLOR = (220, 220, 220)    # Color del texto general
MATCH_COLOR = (160, 220, 160)   # Color de las cartas que ya fueron emparejadas
HIGHLIGHT = (255, 220, 120)     # Color para resaltar cartas seleccionadas o acciones

# -------------------------
# Clases y utilidades del juego
# -------------------------
class MemoriceGame:
    """
    Clase que modela la lógica del juego de Memorice (Memoria).
    
    Funcionalidades:
    - Genera un tablero cuadrado con pares de valores mezclados aleatoriamente.
    - Controla las posiciones emparejadas, el número de turnos y las cartas vistas.
    - Permite aplicar un turno y verificar si el juego ha terminado.
    """
    
    def __init__(self, size=BOARD_SIZE):
        """
        Inicializa el juego.
        
        Args:
            size (int): tamaño del tablero (size x size)
        """
        self.size = size
        self.total_pairs = (size * size) // 2   # cantidad total de pares
        self._generate_board()                  # genera tablero mezclado
        self.matched = set()                    # posiciones ya emparejadas
        self.moves = 0                          # contador de turnos realizados
        self.seen = {}                          # memoria de cartas vistas (pos -> valor)

    def _generate_board(self):
        """
        Genera el tablero con pares duplicados y mezclados aleatoriamente.
        """
        vals = list(range(1, (self.size * self.size // 2) + 1)) * 2
        random.shuffle(vals)
        self.board = [vals[i*self.size:(i+1)*self.size] for i in range(self.size)]

    def value_at(self, pos):
        """
        Devuelve el valor de la carta en la posición dada.
        
        Args:
            pos (tuple): (fila, columna)
            
        Returns:
            int: valor de la carta
        """
        r, c = pos
        return self.board[r][c]

    def all_positions(self):
        """
        Generador que devuelve todas las posiciones del tablero.
        """
        for r in range(self.size):
            for c in range(self.size):
                yield (r, c)

    def is_matched(self, pos):
        """
        Verifica si una posición ya fue emparejada.
        """
        return pos in self.matched

    def unmatched_positions(self):
        """
        Devuelve una lista de posiciones aún no emparejadas.
        """
        return [p for p in self.all_positions() if p not in self.matched]

    def seen_pairs(self):
        """
        Devuelve un diccionario: valor -> lista de posiciones vistas
        que aún no han sido emparejadas.
        """
        mp = defaultdict(list)
        for pos, val in self.seen.items():
            if pos not in self.matched:
                mp[val].append(pos)
        return mp

    def apply_turn(self, pos1, pos2):
        """
        Aplica un turno: voltea dos posiciones.
        
        Args:
            pos1, pos2 (tuple): posiciones a voltear
            
        Returns:
            bool: True si las posiciones forman un par, False en caso contrario
        """
        self.moves += 1
        v1 = self.value_at(pos1)
        v2 = self.value_at(pos2)
        # almacenar en memoria
        if pos1 not in self.matched:
            self.seen[pos1] = v1
        if pos2 not in self.matched:
            self.seen[pos2] = v2
        # si forman par
        if v1 == v2 and pos1 != pos2:
            self.matched.add(pos1)
            self.matched.add(pos2)
            self.seen.pop(pos1, None)
            self.seen.pop(pos2, None)
            return True
        return False

    def is_finished(self):
        """
        Verifica si se han encontrado todos los pares del tablero.
        """
        return len(self.matched) // 2 == self.total_pairs

# -------------------------
# Nodo para A* 
# Representa un estado en la búsqueda A*
# -------------------------
# Cada nodo contiene:
#   f      : valor f = g + h (prioridad en la cola de A*)
#   g      : costo acumulado desde el inicio hasta este nodo
#   state  : estado del juego (tupla de matched y seen)
#   parent : nodo padre (para reconstruir el camino)
#   action : acción que llevó a este estado desde el padre
Node = namedtuple("Node", ["f", "g", "state", "parent", "action"])

def state_to_key(state):
    """
    Convierte un estado del juego en una clave hashable para usar en diccionarios.
    
    state: tuple
        - matched_tuple: tupla de posiciones emparejadas (ordenadas)
        - seen_items_tuple: tupla de (valor, tupla de posiciones ordenadas)
    
    Retorna:
        (matched_key, seen_key): ambos tuplas ordenadas que representan el estado
    """
    matched, seen = state
    matched_key = tuple(sorted(matched))  # ordenar posiciones emparejadas

    # Convertir 'seen' a tupla ordenada de (valor, posiciones)
    if isinstance(seen, dict):
        seen_key = tuple(sorted((val, tuple(sorted(pos_list))) for val, pos_list in seen.items()))
    else:
        # si 'seen' ya es tupla de (val, tuple(positions)), solo ordenamos
        seen_key = tuple(sorted((val, tuple(sorted(pos_list))) for val, pos_list in seen))

    return (matched_key, seen_key)

# -------------------------
# Heurística
# -------------------------
def heuristic_state(state, total_pairs):
    """
    Heurística para A*:
    - Calcula cuántos pares faltan por emparejar.
    - Cada turno puede resolver como máximo 1 par,
      por lo que el valor mínimo de turnos restantes
      es (total_pairs - pares_actuales).
    """
    matched, seen = state
    matched_count = len(matched) // 2
    return total_pairs - matched_count  # turnos mínimos restantes (cada turno puede coincidir como máximo con 1 par)

# -------------------------
# Generador de sucesores
# Produce los posibles estados futuros a partir del estado actual del juego
# Cada acción corresponde a un turno que voltea dos cartas
#
# Estrategia:
# 1) Si existen pares conocidos en memoria (seen), generar únicamente esas acciones (prioridad)
# 2) Si no hay pares conocidos, generar acciones de exploración combinando:
#    - Posiciones aún no vistas (preferiblemente)
#    - Algunas posiciones ya vistas
# Se limita la ramificación para mantener la búsqueda factible.
# -------------------------
def successors_state(state, game):
    matched, seen = state
    matched_set = set(matched)

    # Convertir 'seen' a un diccionario uniforme
    if isinstance(seen, dict):
        seen_map = {val: list(poses)[:] for val, poses in seen.items()}
    else:
        # 'seen' viene como tupla de (valor, tupla(posiciones))
        seen_map = {val: list(poses)[:] for val, poses in seen}

    actions = []

    # 1) Generar acciones de pares conocidos en memoria
    for val, poses in seen_map.items():
        if len(poses) >= 2:
            # Seleccionar las dos primeras posiciones no emparejadas
            available = [p for p in poses if p not in matched_set]
            if len(available) >= 2:
                p1, p2 = available[0], available[1]
                new_matched = set(matched_set)
                new_matched.add(p1); new_matched.add(p2)

                # Actualizar 'seen' eliminando posiciones ya emparejadas
                new_seen = {k: [x for x in v if x not in {p1, p2}] for k, v in seen_map.items()}
                new_seen = {k: v for k, v in new_seen.items() if v}  # eliminar listas vacías

                actions.append((
                    ("match", (p1, p2)),
                    (tuple(sorted(new_matched)), tuple(sorted((k, tuple(sorted(v))) for k, v in new_seen.items()))),
                    1
                ))

    if actions:
        return actions  # Priorizar emparejamientos conocidos

    # 2) Generar acciones de exploración si no hay pares conocidos
    unmatched = [p for p in game.all_positions() if p not in matched_set]
    unseen = [p for p in unmatched if p not in game.seen and p not in {item for sl in seen_map.values() for item in sl}]
    if not unseen:
        unseen = unmatched[:]  # fallback si no hay posiciones invisibles

    K = 6  # límite de candidatos para combinar (controla ramificación)
    candidates = unseen[:K]
    seen_positions = [p for sl in seen_map.values() for p in sl if p not in matched_set]

    # Construir combinaciones de pares (c, d)
    for i, c in enumerate(candidates):
        # Priorizar combinaciones con posiciones ya vistas (para explotar memoria)
        for d in seen_positions:
            if d == c: continue
            new_matched = set(matched_set)
            new_seen = {k: list(v)[:] for k, v in seen_map.items()}

            # Revelar c y d: agregar a 'seen'
            v_c = game.value_at(c)
            v_d = game.value_at(d)
            new_seen.setdefault(v_c, [])
            if c not in new_seen[v_c]:
                new_seen[v_c].append(c)
            new_seen.setdefault(v_d, [])
            if d not in new_seen[v_d]:
                new_seen[v_d].append(d)

            # Si coinciden, marcar como emparejadas
            if v_c == v_d and c != d:
                new_matched.add(c); new_matched.add(d)
                new_seen[v_c] = [x for x in new_seen[v_c] if x not in {c, d}]
                if not new_seen[v_c]:
                    del new_seen[v_c]

            actions.append((
                ("turn", (c, d)),
                (tuple(sorted(new_matched)), tuple(sorted((k, tuple(sorted(v))) for k, v in new_seen.items()))),
                1
            ))

        # Combinar con otras posiciones no vistas/no emparejadas (limitar a K)
        others = [p for p in unmatched if p != c][:K]
        for d in others[:K]:
            new_matched = set(matched_set)
            new_seen = {k: list(v)[:] for k, v in seen_map.items()}
            v_c = game.value_at(c)
            v_d = game.value_at(d)
            new_seen.setdefault(v_c, [])
            if c not in new_seen[v_c]:
                new_seen[v_c].append(c)
            new_seen.setdefault(v_d, [])
            if d not in new_seen[v_d]:
                new_seen[v_d].append(d)
            if v_c == v_d and c != d:
                new_matched.add(c); new_matched.add(d)
                new_seen[v_c] = [x for x in new_seen[v_c] if x not in {c, d}]
                if not new_seen[v_c]:
                    del new_seen[v_c]

            actions.append((
                ("turn", (c, d)),
                (tuple(sorted(new_matched)), tuple(sorted((k, tuple(sorted(v))) for k, v in new_seen.items()))),
                1
            ))

    return actions

# -------------------------
# Búsqueda A* para planificar los movimientos del juego
# Devuelve una secuencia de acciones para llegar al objetivo (todas las cartas emparejadas)
# Se incluye un límite de expansiones para evitar búsquedas infinitas
# -------------------------
def astar_plan(initial_state, game, max_expansions=200000):
    total_pairs = game.total_pairs

    # Inicialización de la frontera (priority queue basada en f = g + h)
    frontier = []
    start_key = state_to_key(initial_state)
    start_h = heuristic_state(initial_state, total_pairs)
    # Nodo inicial: f = heurística, g = 0 (costo acumulado)
    heapq.heappush(frontier, Node(f=start_h, g=0, state=initial_state, parent=None, action=None))

    # Diccionario para reconstruir el camino (state_key -> (parent_key, acción que llevó aquí))
    came_from = {state_to_key(initial_state): (None, None)}
    # gscore: costo mínimo para llegar a cada estado
    gscore = {state_to_key(initial_state): 0}

    expansions = 0  # contador de nodos expandidos

    while frontier:
        node = heapq.heappop(frontier)
        state = node.state
        key = state_to_key(state)
        g = node.g

        # Verificar si alcanzamos el objetivo: todas las parejas emparejadas
        matched_tuple, _ = state
        if len(matched_tuple) // 2 == total_pairs:
            # Reconstruir la secuencia de acciones desde el estado inicial hasta el objetivo
            path = []
            cur_key = key
            while True:
                parent_key, action = came_from[cur_key]
                if parent_key is None:
                    break
                path.append(action)
                cur_key = parent_key
            path.reverse()
            return path  # retorna lista de acciones ("turn", (p1,p2)) o ("match",(p1,p2))

        # Expandir sucesores
        expansions += 1
        if expansions > max_expansions:
            return None  # si se supera el límite de expansiones, se abandona la búsqueda

        succs = successors_state(state, game)
        for (action, new_state, cost) in succs:
            new_key = state_to_key(new_state)
            tentative_g = g + cost
            # Solo actualizar si encontramos un camino más corto hacia new_state
            if tentative_g < gscore.get(new_key, float("inf")):
                gscore[new_key] = tentative_g
                h = heuristic_state(new_state, total_pairs)
                f = tentative_g + h
                # Añadir nuevo nodo a la frontera
                heapq.heappush(frontier, Node(f=f, g=tentative_g, state=new_state, parent=key, action=action))
                came_from[new_key] = (key, action)

    return None  # no se encontró solución dentro del límite de expansiones


# -------------------------
# Política "greedy" de respaldo
# Se usa si A* falla o es demasiado lento.
# Estrategia:
# 1. Si hay algún par conocido en la memoria (seen) -> emparejarlo.
# 2. Si no, explorar:
#    a) Elegir una carta no vista al azar.
#    b) Si el valor ya se ha visto antes en otra posición -> emparejar con esa posición.
#    c) Si no, elegir otra carta aleatoria no vista para voltear.
# -------------------------
def greedy_policy(game):
    # Revisar si hay pares conocidos en memoria
    seen_pairs = game.seen_pairs()
    for val, poses in seen_pairs.items():
        if len(poses) >= 2:
            # retornar acción de voltear las dos posiciones conocidas
            p1, p2 = poses[0], poses[1]
            return ("turn", (p1, p2))
    
    # Si no hay pares conocidos, explorar
    unmatched = [p for p in game.all_positions() if not game.is_matched(p)]
    unseen = [p for p in unmatched if p not in game.seen]
    if not unseen:
        unseen = unmatched  # fallback si todas las cartas han sido vistas

    # Elegir una carta aleatoria no vista
    p1 = random.choice(unseen)
    v1 = game.value_at(p1)

    # Verificar si ya vimos otra carta con el mismo valor
    for pos, val in game.seen.items():
        if val == v1 and pos != p1 and pos not in game.matched:
            return ("turn", (p1, pos))

    # Si no se encontró pareja conocida, elegir otra carta aleatoria
    remaining = [p for p in unmatched if p != p1]
    if not remaining:
        return ("turn", (p1, p1))  # fallback: repetir la misma carta
    p2 = random.choice(remaining)
    return ("turn", (p1, p2))


# -------------------------
# Función de renderizado con Pygame
# Dibuja el tablero y la información del juego en pantalla
# Parámetros:
#  - screen: superficie de Pygame donde se dibuja
#  - game: instancia de MemoriceGame con el estado actual
#  - reveal_preview: conjunto de posiciones a mostrar temporalmente (para animar volteos)
# -------------------------
def draw_board(screen, game, reveal_preview=None):
    # Fondo
    screen.fill(BG)

    # Fuentes para textos
    font = pygame.font.SysFont("Consolas", 20)
    small = pygame.font.SysFont("Consolas", 16)

    # Título del juego
    title = font.render("Memorice (6x6) — A* agent", True, TEXT_COLOR)
    screen.blit(title, (MARGIN, HEIGHT - 110))

    # Información: número de turnos y pares encontrados
    info = small.render(f"Turns: {game.moves}    Matches: {len(game.matched)//2}/{game.total_pairs}", True, TEXT_COLOR)
    screen.blit(info, (MARGIN, HEIGHT - 80))

    # Dibujar cada celda del tablero
    for r in range(game.size):
        for c in range(game.size):
            x = MARGIN + c * CELL
            y = MARGIN + r * CELL
            pos = (r, c)
            rect = pygame.Rect(x, y, CELL - 4, CELL - 4)

            if pos in game.matched:
                # Carta ya encontrada: verde con valor visible
                pygame.draw.rect(screen, MATCH_COLOR, rect, border_radius=8)
                val = game.value_at(pos)
                txt = font.render(str(val), True, (10, 10, 10))
                screen.blit(txt, (x + CELL//2 - txt.get_width()//2 - 2,
                                  y + CELL//2 - txt.get_height()//2 - 4))
            elif reveal_preview and pos in reveal_preview:
                # Carta volteada temporalmente para mostrar su valor
                pygame.draw.rect(screen, CARD_FRONT, rect, border_radius=8)
                val = game.value_at(pos)
                txt = font.render(str(val), True, (10, 10, 10))
                screen.blit(txt, (x + CELL//2 - txt.get_width()//2 - 2,
                                  y + CELL//2 - txt.get_height()//2 - 4))
            else:
                # Carta no volteada: mostrar reverso
                pygame.draw.rect(screen, CARD_BACK, rect, border_radius=8)

    # Actualizar pantalla
    pygame.display.flip()

# -------------------------
# Función principal del juego
# Maneja la inicialización de Pygame, el bucle principal del juego y la ejecución del agente A*
# -------------------------
def main():
    # Inicializar Pygame y ventana
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Memorice A* Agent")
    clock = pygame.time.Clock()  # Controla la velocidad de actualización de frames

    # Crear instancia del juego
    game = MemoriceGame()
    auto_run = True  # El juego se resolverá automáticamente
    running = True   # Controla el bucle principal
    solving = False  # Marca si el agente ha empezado a jugar
    time_start = None
    time_end = None

    # Estado del agente
    planned_actions = []  # Lista de acciones planificadas por A*
    last_plan_time = None

    # Delays para mostrar animaciones
    flip_delay = 0.45  # segundos que se muestra cada carta al voltear
    between_turn_delay = 0.25  # pausa entre turnos

    # Bucle principal del juego
    while running:
        # Manejo de eventos (cerrar ventana)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Comenzar resolución automática si no ha iniciado
        if not solving:
            solving = True
            time_start = time.time()

        if solving and not game.is_finished():
            # Si no hay plan actual, generar plan con A* desde el estado actual
            if not planned_actions:
                matched_tuple = tuple(sorted(game.matched))
                # Agrupar posiciones ya vistas por valor
                seen_map = defaultdict(list)
                for pos, v in game.seen.items():
                    if pos not in game.matched:
                        seen_map[v].append(pos)
                seen_state = tuple(sorted((v, tuple(sorted(ps))) for v, ps in seen_map.items()))
                initial_state = (matched_tuple, seen_state)

                # Generar plan con A* (máximo número de expansiones)
                plan = astar_plan(initial_state, game, max_expansions=150000)
                last_plan_time = time.time()
                if plan is None:
                    # Si A* falla, usar política voraz (greedy)
                    action = greedy_policy(game)
                    planned_actions = [action]
                else:
                    planned_actions = plan

            # Ejecutar la primera acción planificada
            if planned_actions:
                action = planned_actions.pop(0)
                act_type, (p1, p2) = action

                # Mostrar volteo de cartas en pantalla
                draw_board(screen, game, reveal_preview={p1})
                pygame.time.delay(int(flip_delay * 1000))
                draw_board(screen, game, reveal_preview={p1, p2})
                pygame.time.delay(int(flip_delay * 1000))

                # Aplicar turno al estado del juego
                matched = game.apply_turn(p1, p2)

                # Mostrar resultado del turno
                draw_board(screen, game, reveal_preview={p1, p2})
                pygame.time.delay(int(between_turn_delay * 1000))

            # Control de frames
            clock.tick(FPS)

        else:
            # Juego terminado
            if time_end is None:
                time_end = time.time()
            draw_board(screen, game, reveal_preview=None)

            # Mostrar estadísticas finales en pantalla
            font = pygame.font.SysFont("Consolas", 18)
            small = pygame.font.SysFont("Consolas", 16)
            summary = font.render("Finished!", True, TEXT_COLOR)
            screen.blit(summary, (MARGIN + 250, HEIGHT - 110))
            stats = small.render(f"Turns: {game.moves}    Time: {round(time_end - time_start, 4)}s", True, TEXT_COLOR)
            screen.blit(stats, (MARGIN + 250, HEIGHT - 80))
            pygame.display.flip()

            # Pausa corta antes de cerrar
            pygame.time.delay(2000)
            running = False

    # Cerrar Pygame
    pygame.quit()

    # Mostrar resultado final en la terminal
    print("=== RESULTADO ===")
    print("Turns:", game.moves)
    if time_start and time_end:
        print("Tiempo (s):", round(time_end - time_start, 4))

# Ejecutar el juego si este archivo se corre directamente
if __name__ == "__main__":
    main()
