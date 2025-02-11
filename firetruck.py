from mesa import Agent
from tree import Tree
import math

class Firetruck(Agent):
    def __init__(self, unique_id, model, speed):
        super().__init__(unique_id, model)
        self.speed = speed

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
    

    def move_towards(self, target_pos):
        """
        Moves the firetruck towards the target position. The distance moved in one
        time step is determined by the firetruck's speed.
        """
        current_x, current_y = self.pos
        target_x, target_y = target_pos

        # Calculate the difference in positions.
        dx = target_x - current_x
        dy = target_y - current_y

        # Compute the Euclidean distance between current and target positions.
        distance = math.sqrt(dx**2 + dy**2)
        if distance == 0:
            # Already at the target.
            return

        # Determine the actual step distance (don't overshoot the target).
        step_distance = min(self.speed, distance)

        # Calculate the proportional step components.
        # These might be fractional, so we round to the nearest integer.
        step_x = int(round((dx / distance) * step_distance))
        step_y = int(round((dy / distance) * step_distance))

        new_x = current_x + step_x
        new_y = current_y + step_y

        new_pos = (new_x, new_y)
        self.model.grid.move_agent(self, new_pos)

    def step(self):
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
                self.move_towards(closest_fire.pos)
    