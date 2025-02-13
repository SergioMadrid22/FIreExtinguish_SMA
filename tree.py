from mesa import Agent
import random
import numpy as np

# Function for computing the rate of spread of a neighbor using Rothermel fire model
def compute_rate_of_spread(w_0, delta, M_x, sigma, h, S_T, S_e, p_p, M_f, U, U_dir, slope_mag, slope_dir, spread_angle):
    eta_S = np.minimum(0.174 * S_e**-0.19, 1)
    r_M = np.minimum(M_f / M_x, 1)
    eta_M = 1 - 2.59 * r_M + 5.11 * r_M**2 - 3.52 * r_M**3
    w_n = w_0 * (1 - S_T)
    p_b = w_0 / delta
    B = p_b / p_p
    B_op = 3.348 * sigma**-0.8189
    gamma_prime_max = sigma**1.5 / (495 + 0.0594 * sigma**1.5)
    A = 133 * sigma**-0.7913
    gamma_prime = gamma_prime_max * (B / B_op) ** A * np.exp(A * (1 - B / B_op))
    I_R = gamma_prime * w_n * h * eta_M * eta_S
    xi = np.exp((0.792 + 0.681 * sigma**0.5) * (B + 0.1)) / (192 + 0.2595 * sigma)
    c = 7.47 * np.exp(-0.133 * sigma**0.55)
    b = 0.02526 * sigma**0.54
    e = 0.715 * np.exp(-3.59e-4 * sigma)
    
    # Convert wind direction to a standard angle (radians)
    wind_angle = np.radians(90 - U_dir)
    # Compute the wind component along the fire spread direction:
    wind_along_spread = U * np.cos(wind_angle - spread_angle)
    wind_along_spread = max(wind_along_spread, 0)
    
    phi_w = c * wind_along_spread**b * (B / B_op) ** -e
    # For slope, we use the squared magnitude as before:
    phi_s = 5.275 * B**-0.3 * slope_mag**2
    epsilon = np.exp(-138 / sigma)
    Q_ig = 250 + 1116 * M_f
    R = ((I_R * xi) * (1 + phi_w + phi_s)) / (p_b * epsilon * Q_ig)
    return max(R, 0)

class Tree(Agent):
    def __init__(self, unique_id, model, fuel_params, tree_params):
        """
        tree_params should include:
          - "extinguish_steps": integer number of steps required to extinguish the tree (if attacked)
          - "burning_cooldown": integer number of steps to wait before the tree can spread the fire again
        """
        super().__init__(unique_id, model)
        self.status = "healthy"
        self.health = 10
        self.extinguish_steps = tree_params["extinguish_steps"]
        self.fuel_params = fuel_params
        self.burning_cooldown = tree_params.get("burning_cooldown", 0)
        # Initialize the cooldown counter to 0 (ready to spread fire immediately once burning)
        self.cooldown_counter = 1

    def step(self):
        if self.status == "burning":
            self.burn()

    def burn(self):
        # Reduce health as the tree burns.
        self.health -= 1
        if self.health <= 0:
            self.status = "burnt"
            return

        # Only attempt to spread fire if the cooldown has expired.
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        # Try to spread fire to healthy neighbors.
        for neighbor in self.model.grid.get_neighbors(self.pos, moore=True, radius=1):
            if isinstance(neighbor, Tree) and neighbor.status == "healthy":
                # Calculate the spread angle from self.pos to neighbor.pos
                delta_y = self.pos[1] - neighbor.pos[1]
                delta_x = neighbor.pos[0] - self.pos[0]
                spread_angle = np.arctan2(delta_y, delta_x)
                
                # Compute the rate of spread using the Rothermel model.
                ros = compute_rate_of_spread(
                    spread_angle=spread_angle,
                    **self.fuel_params,
                )
                # Use a normalized probability to ignite the neighbor.
                if random.random() < min(ros / 10, 1):
                    neighbor.status = "burning"
        
        # After spreading fire, reset the cooldown counter.
        self.cooldown_counter = self.burning_cooldown
