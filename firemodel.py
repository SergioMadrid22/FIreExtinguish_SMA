from mesa import Model
from mesa.time import SimultaneousActivation, RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from tree import Tree
from firetruck import Firetruck
import random

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
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.num_firetrucks = num_firetrucks
        self.tree_density = tree_density
        self.extinguish_steps = extinguish_steps
        self.wind_direction = wind_directions[wind_direction]
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
        for i in range(num_firetrucks):
            x, y = random.choice([(0, y) for y in range(height)] + [(width-1, y) for y in range(height)] +
                                  [(x, 0) for x in range(width)] + [(x, height-1) for x in range(width)])
            
            firetruck = Firetruck(i + width * height, self, truck_speed)
            self.grid.place_agent(firetruck, (x, y))
            self.schedule.add(firetruck)

        # Data collector for updating the chart
        self.datacollector = DataCollector(model_reporters={
            "Healthy": self.count_healthy_trees,
            "Burning": self.count_burning_trees,
            "Burnt": self.count_burnt_trees,
        })
        self.datacollector.collect(self)

    def step(self):
        self.datacollector.collect(self)
        # Stop execution when no trees are burning
        if self.count_burning_trees() == 0:
            self.running = False
        self.schedule.step()
    
    def count_healthy_trees(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Tree) and a.status == "healthy")
    
    def count_burning_trees(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Tree) and a.status == "burning")
    
    def count_burnt_trees(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Tree) and a.status == "burnt")

