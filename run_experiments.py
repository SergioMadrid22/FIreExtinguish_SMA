import os
from mesa.batchrunner import batch_run
from firemodel import FireModel
import pandas as pd
import matplotlib.pyplot as plt

# Create an "images" folder if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

# ---------------------------
# Run the batch experiment
# ---------------------------
def run_batch_experiment():
    # Create a list of parameter values to experiment with
    attack_strategies = ["Base", "Direct Attack", "Parallel Attack"]
    fuel_types = ["Tall grass", "Chaparral", "Litter and understory", "Logging slash"]

    # Set the parameters of the model for the experiment
    parameter_combinations = {
        "attack_strategy": attack_strategies,
        "fuel_type": fuel_types,
        "tree_density": 0.60,
        "num_firetrucks": 15,
        "truck_speed": 3,
        "wind_speed": 15,
        "wind_direction": "North",
        "width": 60,
        "height": 60,
        "extinguish_steps": 2,
    }

    results = batch_run(
        FireModel,
        parameter_combinations,
        iterations=1,      # Number of runs for each combination
        max_steps=1000,    # Maximum steps in each run
        data_collection_period=1,  # Collect data every step
        display_progress=True
    )

    return results

# ---------------------------
# Plot for Trees Extinguished
# ---------------------------
def plot_extinguished(results_df):
    # Ensure a RunId column exists
    if "RunId" not in results_df.columns:
        results_df["RunId"] = 1

    # Extract data for extinguished trees
    plot_data_ext = results_df[['RunId', 'Step', 'attack_strategy', 'Extinguished']]
    plt.figure(figsize=(10, 6))
    
    # Create a color mapping for each attack strategy
    strategies = sorted(plot_data_ext['attack_strategy'].unique())
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {strategy: color_cycle[i % len(color_cycle)] for i, strategy in enumerate(strategies)}

    # Plot each run's extinguished trees time series using the same color for the same strategy.
    for strategy in strategies:
        strat_data = plot_data_ext[plot_data_ext['attack_strategy'] == strategy]
        run_ids = strat_data['RunId'].unique()
        for j, run in enumerate(run_ids):
            run_data = strat_data[strat_data['RunId'] == run]
            label = strategy if j == 0 else None  # Avoid duplicate legend labels
            plt.plot(run_data['Step'], run_data['Extinguished'],
                     label=label, color=colors[strategy])
    
    plt.xlabel('Time (Steps)')
    plt.ylabel('Number of Trees Extinguished')
    plt.title('Trees Extinguished over Time for Different Attack Strategies')
    plt.legend(title="Attack Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure to the images folder
    plt.savefig("images/extinguished_trees.png")
    plt.close()  # Close the figure

# ---------------------------
# Plot Alive Trees by Fuel Type
# ---------------------------
def plot_runs_by_fuel_type(results_df):
    # Ensure RunId exists
    if "RunId" not in results_df.columns:
        results_df["RunId"] = 1

    # Check required columns
    if 'fuel_type' not in results_df.columns:
        print("No 'fuel_type' column in the results.")
        return
    if 'Good' not in results_df.columns:
        print("No 'Good' data found. Ensure your model collects this variable.")
        return

    # Create a color mapping for each attack strategy (to be reused in every fuel-type plot)
    strategies = sorted(results_df['attack_strategy'].unique())
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {strategy: color_cycle[i % len(color_cycle)] for i, strategy in enumerate(strategies)}

    # Get unique fuel types and create separate plots for each.
    fuel_types = sorted(results_df['fuel_type'].unique())
    for fuel in fuel_types:
        fuel_data = results_df[results_df['fuel_type'] == fuel]
        plt.figure(figsize=(10, 6))
        plt.title(f"Alive Trees over Time for Fuel Type: {fuel}")

        # Determine overall x-axis limits for this fuel type.
        overall_x_min = fuel_data['Step'].min()
        overall_x_max = fuel_data['Step'].max()

        # To avoid duplicate legend entries for the same strategy in this fuel-type plot.
        labeled_strategies = set()

        # Plot each run (grouped by RunId) for this fuel type.
        run_ids = sorted(fuel_data['RunId'].unique())
        for run in run_ids:
            run_data = fuel_data[fuel_data['RunId'] == run]
            # Get the attack strategy for this run
            strategy = run_data['attack_strategy'].iloc[0]
            color = colors[strategy]
            label = strategy if strategy not in labeled_strategies else None
            if label is not None:
                labeled_strategies.add(strategy)
            plt.plot(run_data['Step'], run_data['Good'], label=label, color=color)
            # Draw a horizontal dashed line (across the full x-axis) at the run's final Good value.
            final_good = run_data['Good'].iloc[-1]
            plt.hlines(y=final_good,
                       xmin=overall_x_min,
                       xmax=overall_x_max,
                       colors=color,
                       linestyles='--',
                       alpha=0.7)
            # Annotate the final value at the right end.
            plt.text(overall_x_max, final_good, f' {final_good}',
                     verticalalignment='center', color=color)
        
        plt.xlabel('Time (Steps)')
        plt.ylabel('Number of Alive Trees')
        plt.legend(title="Attack Strategy")
        plt.tight_layout()
        
        # Save each fuel type plot to the images folder with the fuel name in the filename.
        filename = f"images/alive_trees_fuel_{fuel.replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()

# ---------------------------
# Main Execution
# ---------------------------
experiment_results = run_batch_experiment()
results_df = pd.DataFrame(experiment_results)

# Plot 1: Trees Extinguished over Time (all runs together)
plot_extinguished(results_df)

# Plot 2: Alive Trees by Fuel Type, with runs colored by attack strategy
plot_runs_by_fuel_type(results_df)

# ---------------------------
# Final Summary Print
# ---------------------------
# For each run, get the row with the maximum 'Step' value (i.e., the final time step).
# Then print a summary with the final number of extinguished and healthy (Good) trees.
if "RunId" not in results_df.columns:
    results_df["RunId"] = 1

final_summary = results_df.loc[results_df.groupby("RunId")["Step"].idxmax(),
                               ["RunId", "attack_strategy", "fuel_type", "Extinguished", "Good"]]

print("\nFinal Summary of Experiment Results (per run):")
print(final_summary.to_string(index=False))