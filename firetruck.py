from mesa import Agent
from tree import Tree
import math

class Firetruck(Agent):
    def __init__(self, unique_id, model, speed, strategy="base", production_rate=1.0, safe_distance=2):
        """
        Parámetros:
          - speed: cuántas celdas se pueden mover por step.
          - strategy: "base" o "direct_attack".
          - production_rate: velocidad (en metros/segundo) para construir la línea de fuego,
                              relevante para el ataque directo.
        """
        super().__init__(unique_id, model)
        self.speed = speed
        self.strategy = strategy
        self.production_rate = production_rate
        self.safe_distance = safe_distance

    def get_closest_burning_tree(self):
        """
        Scans all agents in the schedule for burning trees and returns the one
        closest to this firetruck (using Euclidean distance).
        """
        # Get all agents with a burning status.
        burning_trees = [
            agent for agent in self.model.schedule.agents
            if isinstance(agent, Tree) and agent.status == "burning"
        ]
        if not burning_trees:
            return None

        # Helper to compute Euclidean distance.
        def distance(pos1, pos2):
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        # Find and return the burning tree with the minimum distance.
        closest_tree = min(burning_trees, key=lambda tree: distance(self.pos, tree.pos))
        return closest_tree
    
    def calculate_next_pos(self, target_pos):
        """
        Calcula la posición a la que se moverá el firetruck hacia el objetivo,
        similar a tu método move_towards pero sin mover todavía al agente.
        """
        current_x, current_y = self.pos
        target_x, target_y = target_pos

        dx = target_x - current_x
        dy = target_y - current_y

        distance = math.sqrt(dx**2 + dy**2)
        if distance == 0:
            return self.pos

        step_distance = min(self.speed, distance)
        step_x = int(round((dx / distance) * step_distance))
        step_y = int(round((dy / distance) * step_distance))

        new_x = current_x + step_x
        new_y = current_y + step_y

        return (new_x, new_y)

    #################################
    # Funciones para ataque directo #
    #################################
    def get_local_grid(self, radius=5):
        """
        Extrae una subárea (diccionario de posiciones y su estado) centrada en la posición
        actual del agente. Se recorre el grid dentro del radio indicado.
        """
        local_grid = {}
        x, y = self.pos
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                pos = (x + dx, y + dy)
                if not self.model.grid.out_of_bounds(pos):
                    # Por simplicidad: se determina el estado a partir de los agentes en la celda.
                    state = "unburned"
                    agents = self.model.grid.get_cell_list_contents(pos)
                    for agent in agents:
                        if isinstance(agent, Tree):
                            if agent.status == "burning":
                                state = "burning"
                            elif agent.status == "burned":
                                state = "burned"
                            elif agent.status == "suppressed":
                                state = "suppressed"
                    local_grid[pos] = state
        return local_grid

    def simulate_fire(self, local_grid, T_lookahead):
        """
        Simula la propagación del fuego en la subárea local durante T_lookahead segundos.
        Se utiliza un paso de tiempo dt = 1 (segundo) y se asume que:
          - Los árboles "burning" pueden prender a los vecinos "healthy"
            con una probabilidad dependiente de la tasa de propagación.
          - Los estados "burnt" o "suppressed" no cambian.
        Retorna un diccionario con el estado final de cada celda.
        """
        dt = 1  # paso de tiempo en segundos
        n_steps = max(1, int(math.ceil(T_lookahead)))
        simulated_grid = local_grid.copy()
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),           (0, 1),
                            (1, -1),  (1, 0),  (1, 1)]
        
        for _ in range(n_steps):
            new_grid = simulated_grid.copy()
            # Recorrer todas las celdas en la grilla simulada
            for pos, state in simulated_grid.items():
                if state == "burning":
                    x, y = pos
                    for dx, dy in neighbor_offsets:
                        neighbor_pos = (x + dx, y + dy)
                        if neighbor_pos in simulated_grid and simulated_grid[neighbor_pos] == "healthy":
                            # Calcular el spread_angle de la celda burning al vecino healthy
                            # Se utiliza la misma fórmula que en el método burn de Tree
                            # Notar que: spread_angle = arctan2( (y - (y+dy)), ((x+dx) - x) )
                            spread_angle = np.arctan2(-dy, dx)
                            print("SPREAD ANGLE", spread_angle)
                            # Obtener los parámetros de combustible por defecto del modelo
                            fuel_params = self.model.default_fuel_params
                            ros = compute_rate_of_spread(
                                w_0=fuel_params["w_0"],
                                delta=fuel_params["delta"],
                                M_x=fuel_params["M_x"],
                                sigma=fuel_params["sigma"],
                                h=fuel_params["h"],
                                S_T=fuel_params["S_T"],
                                S_e=fuel_params["S_e"],
                                p_p=fuel_params["p_p"],
                                M_f=fuel_params["M_f"],
                                U=fuel_params["U"],
                                U_dir=fuel_params["U_dir"],
                                slope_mag=fuel_params["slope_mag"],
                                slope_dir=fuel_params["slope_dir"],
                                spread_angle=spread_angle
                            )
                            probability = min(ros / 10, 1)
                            if random.random() < probability:
                                new_grid[neighbor_pos] = "burning"
            simulated_grid = new_grid
        return simulated_grid

    def scan_fireline(self, current_pos, predicted_grid, stage):
        """
        Aplica el proceso de “scan” para buscar, a partir de la posición actual,
        la primera celda en estado "burning" siguiendo un recorrido circular.
        
        Parámetros:
          - stage: "non_diagonal" rechaza celdas diagonales; "diagonal" las acepta.
        """
        # Orden de direcciones: norte, noreste, este, sureste, sur, suroeste, oeste, noroeste.
        directions = [(0, -1), (1, -1), (1, 0), (1, 1),
                      (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        for dx, dy in directions:
            candidate = (current_pos[0] + dx, current_pos[1] + dy)
            if candidate in predicted_grid and predicted_grid[candidate] == "burning":
                if stage == "non_diagonal" and abs(dx) == 1 and abs(dy) == 1:
                    # En la etapa 1 se aborta si la celda es diagonal.
                    return None
                return candidate
        return None

    def direct_attack_strategy(self):
        """
        Implementa la estrategia de ataque directo mediante el esquema
        “predict-and-scan” de dos etapas.
        """
        cell_size = self.model.cell_size

        # ETAPA 1: vecinos no diagonales
        T_lookahead = cell_size / self.production_rate
        local_grid = self.get_local_grid(radius=2)
        predicted_grid = self.simulate_fire(local_grid, T_lookahead)
        destination = self.scan_fireline(self.pos, predicted_grid, stage="non_diagonal")
        print("PRIMERA ETAPA")
        # Si la celda obtenida es diagonal (o no se encontró), se pasa a la ETAPA 2.
        if destination is None:
            T_lookahead = math.sqrt(2) * cell_size / self.production_rate
            predicted_grid = self.simulate_fire(local_grid, T_lookahead)
            destination = self.scan_fireline(self.pos, predicted_grid, stage="diagonal")
            print("SEGUNDA ETAPA")
        if destination is not None:
            # Se intenta reservar la celda destino
            if self.model.can_reserve_cell(destination, self):
                self.model.reserve_cell(destination, self)
                self.model.grid.move_agent(self, destination)
                self.model.suppress_cell(destination)
            else:
                alternative = self.find_alternative(destination)
                if alternative and self.model.can_reserve_cell(alternative, self):
                    self.model.reserve_cell(alternative, self)
                    self.model.grid.move_agent(self, alternative)
                    self.model.suppress_cell(alternative)
                else:
                    self.base_strategy()
        else:
            self.base_strategy()


    ##############################
    # Funciones para ataque base #
    ##############################

    def base_strategy(self):
        nearby_agents = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True)
        burning_trees = [ agent for agent in nearby_agents
                         if isinstance(agent, Tree) and agent.status == "burning"]

        # Attack the first burning tree found in the neighborhood    
        if burning_trees:
            target_tree = burning_trees[0]
            target_tree.extinguish_steps -= 1
            if target_tree.extinguish_steps <= 0:
                target_tree.extinguish_steps = 0
                target_tree.status = "extinguished"
        else:
            closest_fire = self.get_closest_burning_tree()
            if closest_fire is not None:
                desired_pos = self.calculate_next_pos(closest_fire.pos)
                # Intentar reservar la celda deseada en el modelo.
                if self.model.can_reserve_cell(desired_pos, self):
                    self.model.reserve_cell(desired_pos, self)
                    self.model.grid.move_agent(self, desired_pos)
                else:
                    # Si la celda ya está reservada, se puede implementar una estrategia
                    # alternativa, por ejemplo, buscar una celda adyacente libre.
                    alternative_pos = self.find_alternative(desired_pos)
                    if alternative_pos and self.model.can_reserve_cell(alternative_pos, self):
                        self.model.reserve_cell(alternative_pos, self)
                        self.model.grid.move_agent(self, alternative_pos)

    ##################################
    # Funciones para ataque paralelo #
    ##################################
    def parallel_attack_strategy(self):
        """
        Implementa la estrategia de Parallel Attack.
        El agente simula la propagación en una subárea delimitada por la distancia de seguridad (safe_distance)
        y escanea sus vecinos inmediatos. Para cada vecino se calcula la distancia (D_fire) a la celda burning
        más cercana en la predicción; se selecciona el primer vecino cuyo D_fire esté cercano a safe_distance.
        """
        cell_size = self.model.cell_size
        safe_distance = self.safe_distance
        # Se usa el peor caso: celdas diagonales
        T_lookahead = math.sqrt(2) * cell_size / self.production_rate
        # Se define el radio de la subárea como el número de celdas que equivale a safe_distance
        distance_bound = int(math.ceil(safe_distance / cell_size))
        
        local_grid = self.get_local_grid(radius=distance_bound)
        predicted_grid = self.simulate_fire(local_grid, T_lookahead)
        
        tolerance = 0.5  # Tolerancia para considerar "cercano" a safe_distance
        candidate_destination = None
        # Escanear las celdas vecinas inmediatas (8 direcciones)
        directions = [(0, -1), (1, -1), (1, 0), (1, 1),
                      (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        for dx, dy in directions:
            candidate = (self.pos[0] + dx, self.pos[1] + dy)
            if candidate not in predicted_grid:
                continue
            # Calcular D_fire: distancia desde la celda candidata a la celda burning más cercana en predicted_grid
            d_fire = None
            for pos, state in predicted_grid.items():
                if state == "burning":
                    dist = math.sqrt((candidate[0] - pos[0])**2 + (candidate[1] - pos[1])**2)
                    if d_fire is None or dist < d_fire:
                        d_fire = dist
            if d_fire is None:
                continue
            if abs(d_fire - safe_distance) <= tolerance:
                candidate_destination = candidate
                break
        
        if candidate_destination is not None:
            # Intentar reservar la celda destino
            if self.model.can_reserve_cell(candidate_destination, self):
                self.model.reserve_cell(candidate_destination, self)
                self.model.grid.move_agent(self, candidate_destination)
                self.model.suppress_cell(candidate_destination)
            else:
                alternative = self.find_alternative(candidate_destination)
                if alternative and self.model.can_reserve_cell(alternative, self):
                    self.model.reserve_cell(alternative, self)
                    self.model.grid.move_agent(self, alternative)
                    self.model.suppress_cell(alternative)
                else:
                    self.base_strategy()
        else:
            self.base_strategy()

    def step(self):
        # Según la estrategia asignada al crearse el agente, ejecuta la acción correspondiente.
        if self.strategy == "base":
            self.base_strategy()
        elif self.strategy == "direct_attack":
            self.direct_attack_strategy()
        elif self.strategy == "parallel_attack":
            self.parallel_attack_strategy()
        else:
            # Por defecto, se utiliza la estrategia base.
            self.base_strategy()

    def find_alternative(self, pos):
        """
        Método opcional para buscar una celda alternativa cercana a 'pos'.
        Aquí podrías implementar la lógica para elegir otra celda
        (por ejemplo, probando las celdas adyacentes).
        """
        x, y = pos
        # Ejemplo: probar celdas adyacentes en 4 direcciones.
        alternativas = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        # Filtra las que estén dentro de los límites del grid
        alternativas = [p for p in alternativas if self.model.grid.out_of_bounds(p) is False]
        # Retorna la primera que se encuentre libre (esta lógica se puede mejorar)
        for alt in alternativas:
            if self.model.can_reserve_cell(alt, self):
                return alt
        return None