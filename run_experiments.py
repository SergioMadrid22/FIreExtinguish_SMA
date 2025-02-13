from mesa.batchrunner import batch_run
from firemodel import FireModel
import random
import pandas as pd
import matplotlib.pyplot as plt

# Function to configure the experiment
def run_batch_experiment():
    # Create a list of parameter values to experiment with
    attack_strategies = ["Base", "Direct Attack", "Parallel Attack"]
    #fuel_types = ["Tall grass", "Chaparral", "Litter and understory", "Logging slash"]
    #tree_densities = [0.5, 0.6, 0.7, 0.8, 1.0]
    #num_firetrucks_values = [10, 20, 30, 50, 70, 100]
    #truck_speeds = [3, 5, 10, 15]
    #wind_speeds = [30, 60, 100, 150]
    #wind_directions = ["North", "South", "West", "East"]

    # Set the parameters of the model for the experiment
    parameter_combinations = {
        "attack_strategy": attack_strategies,
        "fuel_type": "Chaparral",
        "tree_density": 0.8,
        "num_firetrucks": 20,
        "truck_speed": 1,
        "wind_speed": 60,
        "wind_direction": "North",
        "width": 60,
        "height": 60,
        "extinguish_steps": 1,

    }

    # Create the BatchRunner with the FireModel and parameters to run the simulations
    results = batch_run(
        FireModel,
        parameter_combinations,
        iterations=2,  # Number of runs for each combination of parameters
        max_steps=1000,  # Maximum steps in each run
        data_collection_period=10,  # Collect data every 10 steps
        display_progress=True
    )


    # After the batch run, you can gather data from the experiment
    # Get the results in a pandas DataFrame

    return results


def plot_batch_results(experiment_results):
    # Convert results to a DataFrame
    results_df = pd.DataFrame(experiment_results)

    # Filter relevant columns (e.g., 'Step', 'attack_strategy', and 'Extinguished')
    plot_data = results_df[['Step', 'attack_strategy', 'Extinguished']]

    # Create a plot for each attack strategy
    plt.figure(figsize=(10, 6))
    strategies = plot_data['attack_strategy'].unique()

    for strategy in strategies:
        strategy_data = plot_data[plot_data['attack_strategy'] == strategy]
        plt.plot(strategy_data['Step'], strategy_data['Extinguished'], label=strategy)

    # Set labels and title
    plt.xlabel('Time (Steps)')
    plt.ylabel('Number of Trees Extinguished')
    plt.title('Trees Extinguished over Time for Different Attack Strategies')

    # Display the legend
    plt.legend(title="Attack Strategy")

    # Show the plot
    plt.show()

# Assuming the experiment_results is already returned from the batch run
experiment_results = run_batch_experiment()  # Your batch experiment function

# Plot the results
plot_batch_results(experiment_results)

