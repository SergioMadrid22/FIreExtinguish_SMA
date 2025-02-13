from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization import Slider, Choice
from firemodel import FireModel
from tree import Tree
from firetruck import Firetruck

# Function for drawing the agents
def agent_portrayal(agent):
    if isinstance(agent, Tree):
        if agent.status == "healthy":
            color = "green"
        elif agent.status == "burning":
            color = "red"
        elif agent.status == "burnt":
            color = "black"
        elif agent.status == "suppressed":
            color = "brown"
        elif agent.status == "extinguished":
            color = "brown"
        return {"Shape": "rect", "Filled": "true", "Color": color, "Layer": 0, "w": 1, "h": 1}
    
    elif isinstance(agent, Firetruck):
        return {"Shape": "rect", "Filled": "true", "Color": "blue", "Layer": 1, "w": 1, "h": 1}


grid = CanvasGrid(agent_portrayal, 60, 60, 500, 500)
tree_stats = ChartModule([
                {"Label": "Healthy", "Color": "Green"},
                {"Label": "Burning", "Color": "Red"},
                {"Label": "Burnt", "Color": "Black"}],
                data_collector_name="datacollector"
            )
extinguished_stats = ChartModule([
                {"Label": "Extinguished", "Color": "Brown"}],
                data_collector_name="datacollector")

attack_strategy = Choice("Select firetrucks attack strategy", value="Base",
                            choices=["Base", "Direct Attack", "Parallel Attack"])
density_slider = Slider("Tree density", 0.8, 0.1, 1.0, 0.05)
extinguish_steps_slider = Slider("Number of steps needed to extinguish fire", 1, 1, 5, 1)
number_firetrucks_slider = Slider("Number of firetrucks to spawn", 30, 0, 100, 1)
truck_speed_slider = Slider("Speed of the firetrucks", 3, 1, 15, 1)
wind_direction_choice = Choice("Select wind direction", value="North",
                            choices=["North", "South", "East", "West",
                                        "North-East", "North-West", "South-East", "South-West"])
wind_speed_slider = Slider("Wind speed", 60, 0, 200, 2)

# Define the parameters of the model
parameters = {
    "attack_strategy": attack_strategy,
    "width": 60,
    "height": 60,
    "num_firetrucks": number_firetrucks_slider,
    "tree_density": density_slider,
    "extinguish_steps": extinguish_steps_slider,
    "truck_speed": truck_speed_slider,
    "wind_direction": wind_direction_choice,
    "wind_speed": wind_speed_slider,
}

# Start server execution 
server = ModularServer(FireModel, [grid, tree_stats, extinguished_stats], "Firefighting Simulation", parameters)
server.port = 8521
