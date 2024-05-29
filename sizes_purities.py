import numpy as np
import matplotlib.pyplot as plt
import inquarting

sizes = [(5, 5, 5), (10, 10, 10), (15, 15, 15), (20, 20, 20)]
gold_ratios = [0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8]
impurity_ratio = 0.01
num_simulations = 10

purity_results = {size: {gold_ratio: [] for gold_ratio in gold_ratios} for size in sizes}

# run the simulation
for size in sizes:
    for gold_ratio in gold_ratios:
        for _ in range(num_simulations):
            alloy_mixture = inquarting.create_alloy_array(size, gold_ratio, 1 - gold_ratio - impurity_ratio, impurity_ratio)
            dissolved_alloy = inquarting.simulate_nitric_acid(alloy_mixture.copy())
            num_gold = np.count_nonzero(dissolved_alloy == 1)
            num_silver_dissolved = np.count_nonzero(dissolved_alloy == 2)
            num_silver = np.count_nonzero(alloy_mixture == 0)
            num_remaining = num_gold + num_silver - num_silver_dissolved
            final_purity = num_gold / num_remaining
            purity_results[size][gold_ratio].append((num_gold, num_remaining, final_purity))

# Calculate and print the average purity for the alloys at each gold concentration
for size in sizes:
    print(f"Average purity for the {size} alloy at each gold concentration:")
    for gold_ratio in gold_ratios:
        avg_num_gold = np.mean([result[0] for result in purity_results[size][gold_ratio]])
        avg_num_remaining = np.mean([result[1] for result in purity_results[size][gold_ratio]])
        avg_final_purity = np.mean([result[2] for result in purity_results[size][gold_ratio]])
        print(f"Gold ratio {gold_ratio:.2f}: Average final purity {avg_final_purity:.4f} (Gold: {avg_num_gold:.2f}, Total Remaining: {avg_num_remaining:.2f})")
    print()

# Function to plot purity vs starting gold concentration for a given size
def plot_purity_vs_gold_concentration(ax, size, gold_ratios, purity_results):
    gold_concentrations = []
    final_purities = []
    for gold_ratio in gold_ratios:
        avg_final_purity = np.mean([result[2] for result in purity_results[size][gold_ratio]])
        gold_concentrations.append(gold_ratio)
        final_purities.append(avg_final_purity)
    bars = ax.bar(gold_concentrations, final_purities, width=0.05)
    ax.set_xlabel('Starting Gold Concentration')
    ax.set_ylabel('Final Purity (Average)')
    ax.set_title(f'Final Purity vs Starting Gold Concentration\nfor {size} Alloy')
    ax.grid(True)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', ha='center', va='bottom')

# Create subplots for each size
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for ax, size in zip(axes, sizes):
    plot_purity_vs_gold_concentration(ax, size, gold_ratios, purity_results)

plt.tight_layout()
plt.show()
