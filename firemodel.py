from mesa import Model
from mesa.time import SimultaneousActivation, RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from tree import Tree
from firetruck import Firetruck
import random
import numpy as np

attack_strategies= {
    "Base": "base",
    "Direct Attack": "direct_attack",
    "Parallel Attack": "parallel_attack"
}

wind_directions = {
        "North": 180,
        "South": 0,
        "East": 90,
        "West": 270,
        "North-East": 135,
        "South-East": 45,
        "South-West": 315,
        "North-West": 225,
    }

class FireModel(Model):
    def __init__(self, 
                 width, height,  
                 num_firetrucks, truck_speed, 
                 tree_density, extinguish_steps,
                 wind_direction, wind_speed,
                 attack_strategy = "base",
                 cell_size=1, random_seed =42,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.attack_strategy = attack_strategies[attack_strategy]
        self.cell_size = 1
        self.cell_reservations = {}
        self.suppressed_cells = set()
        self.num_firetrucks = num_firetrucks
        self.truck_speed = truck_speed
        self.tree_density = tree_density
        self.extinguish_steps = extinguish_steps
        self.wind_direction = wind_directions[wind_direction]
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Set up the parameters of the fuel
        fuel_params = {
            "w_0": 0.5, "delta": 0.5, "M_x": 0.3, "sigma": 1500,
            "h": 8000, "S_T": 0.0555, "S_e": 0.01,
            "p_p": 32, "M_f": 0.1, "U": wind_speed,
            "U_dir": self.wind_direction, "slope_mag": 0.1, "slope_dir": 0
        }

        tree_params = {
            "extinguish_steps": extinguish_steps
        }

        # Place a tree in the center of the grid
        center_x, center_y = width // 2, height // 2
        tree = Tree((center_x, center_y), self, fuel_params, tree_params)
        self.grid.place_agent(tree, (center_x, center_y))
        self.schedule.add(tree)

        # Populate grid with trees
        for x in range(width):
            for y in range(height):
                if x == center_x and y == center_y:
                    continue
                if random.random() < self.tree_density:
                    tree = Tree((x, y), self, fuel_params, tree_params)
                    self.grid.place_agent(tree, (x, y))
                    self.schedule.add(tree)

        # Start fire at the center
        center_tree = self.grid.get_cell_list_contents([(center_x, center_y)])[0]
        center_tree.status = "burning"

        # Add firetrucks at the edges
        posiciones_borde = ([(0, y) for y in range(height)] +
                    [(width-1, y) for y in range(height)] +
                    [(x, 0) for x in range(width)] +
                    [(x, height-1) for x in range(width)])
        random.shuffle(posiciones_borde)
        for i in range(num_firetrucks):
            if posiciones_borde:
                x, y = posiciones_borde.pop()
                firetruck = Firetruck(i + width * height, self, truck_speed, strategy=self.attack_strategy)
                self.grid.place_agent(firetruck, (x, y))
                self.schedule.add(firetruck)

        # Data collector for updating the chart
        self.datacollector = DataCollector(model_reporters={
            "Healthy": self.count_healthy_trees,
            "Burning": self.count_burning_trees,
            "Burnt": self.count_burnt_trees,
            "Extinguished": self.count_extinguished_trees
        })
        self.datacollector.collect(self)

    def step(self):
        self.clear_reservations()
        self.update_cell_reservations()
        self.datacollector.collect(self)
        # Stop execution when no trees are burning
        if self.count_burning_trees() == 0:
            self.running = False
        self.schedule.step()
    
    def can_reserve_cell(self, pos, agent):
        """
        Verifica si la celda 'pos' está libre para reservar.
        Opcionalmente, se puede implementar un sistema de prioridades.
        """
        return pos not in self.cell_reservations

    def reserve_cell(self, pos, agent):
        """Reserva la celda 'pos' para el agente."""
        self.cell_reservations[pos] = agent

    def clear_reservations(self):
        """Limpia las reservas al inicio de cada paso."""
        self.cell_reservations = {}

    def update_cell_reservations(self):
        """Reserva las celdas ocupadas por los firetrucks antes de calcular movimientos."""
        for agent in self.schedule.agents:
            if isinstance(agent, Firetruck):
                self.cell_reservations[agent.pos] = agent
    
    def suppress_cell(self, pos):
        """
        Marca la celda como suprimida, por ejemplo actualizando el estado
        de los árboles que se encuentren allí.
        """
        self.suppressed_cells.add(pos)
        agents = self.grid.get_cell_list_contents(pos)
        for agent in agents:
            if isinstance(agent, Tree):
                agent.status = "suppressed"

    def count_healthy_trees(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Tree) and a.status == "healthy")
    
    def count_burning_trees(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Tree) and a.status == "burning")
    
    def count_burnt_trees(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Tree) and a.status == "burnt")

    def count_extinguished_trees(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Tree) and (a.status == "extinguished" or a.status == "supressed"))