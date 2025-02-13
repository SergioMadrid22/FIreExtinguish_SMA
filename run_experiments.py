from mesa.batchrunner import batch_run
from firemodel import FireModel
import random
import pandas as pd
import matplotlib.pyplot as plt

# Function to configure the experiment
def run_batch_experiment():
    # Create a list of parameter values to experiment with
    attack_strategies = ["Base", "Direct Attack", "Parallel Attack"]

    # Set the parameters of the model for the experiment
    parameter_combinations = {
        "attack_strategy": attack_strategies,
        "fuel_type": "Chaparral",
        "tree_density": 0.65,
        "num_firetrucks": 20,
        "truck_speed": 3,
        "wind_speed": 10,
        "wind_direction": "North",
        "width": 60,
        "height": 60,
        "extinguish_steps": 1,
    }

    # Create the BatchRunner with the FireModel and parameters to run the simulations
    results = batch_run(
        FireModel,
        parameter_combinations,
        iterations=2,      # Number of runs for each combination
        max_steps=1000,    # Maximum steps in each run
        data_collection_period=10,  # Collect data every 10 steps
        display_progress=True
    )

    return results


def plot_batch_results(experiment_results):
    # Convert results to a DataFrame
    results_df = pd.DataFrame(experiment_results)

    # === 1) Plot for Trees Extinguished ===
    # Filter relevant columns: 'Step', 'attack_strategy', and 'Extinguished'
    plot_data_ext = results_df[['Step', 'attack_strategy', 'Extinguished']]

    # Create a plot for each attack strategy
    plt.figure(figsize=(10, 6))
    strategies = plot_data_ext['attack_strategy'].unique()

    for strategy in strategies:
        strategy_data = plot_data_ext[plot_data_ext['attack_strategy'] == strategy]
        plt.plot(strategy_data['Step'], strategy_data['Extinguished'], label=strategy)

    # Set labels and title
    plt.xlabel('Time (Steps)')
    plt.ylabel('Number of Trees Extinguished')
    plt.title('Trees Extinguished over Time for Different Attack Strategies')
    plt.legend(title="Attack Strategy")
    plt.show()

    # === 2) Plot for Alive Trees ===
    # Make sure you have "Healthy" recorded in your DataCollector in FireModel
    if 'Healthy' in results_df.columns:
        plot_data_alive = results_df[['Step', 'attack_strategy', 'Healthy']]

        plt.figure(figsize=(10, 6))
        for strategy in strategies:
            strategy_data = plot_data_alive[plot_data_alive['attack_strategy'] == strategy]
            plt.plot(strategy_data['Step'], strategy_data['Healthy'], label=strategy)

        # Set labels and title
        plt.xlabel('Time (Steps)')
        plt.ylabel('Number of Alive Trees')
        plt.title('Alive Trees over Time for Different Attack Strategies')
        plt.legend(title="Attack Strategy")
        plt.show()
    else:
        print("No 'Healthy' data found. Make sure your model collects this variable.")

# Run the batch experiment
experiment_results = run_batch_experiment()

# Plot the results (extinguished and alive trees)
plot_batch_results(experiment_results)
