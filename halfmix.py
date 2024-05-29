import numpy as np
import matplotlib.pyplot as plt
import inquarting

gold_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
impurity_ratio = 0.01
num_simulations = 10

purity_results_half_mixed = {gold_ratio: [] for gold_ratio in gold_ratios}

# Perform the analysis for the 20x20x20 alloy with half-mixed alloy at each gold concentration
for gold_ratio in gold_ratios:
    for _ in range(num_simulations):
        alloy_mixture = inquarting.create_half_mixed_alloy((20, 20, 20), gold_ratio, 1 - gold_ratio - impurity_ratio, impurity_ratio)
        dissolved_alloy = inquarting.simulate_nitric_acid(alloy_mixture.copy())
        num_gold = np.count_nonzero(dissolved_alloy == 1)
        num_silver_dissolved = np.count_nonzero(dissolved_alloy == 2)
        num_silver = np.count_nonzero(alloy_mixture == 0)
        num_remaining = num_gold + num_silver - num_silver_dissolved
        final_purity = num_gold / num_remaining
        purity_results_half_mixed[gold_ratio].append((num_gold, num_remaining, final_purity))

# Calculate and print the average purity for the 20x20x20 alloy with half-mixed alloy at each gold concentration
print(f"Average purity for the 20x20x20 alloy with half-mixed alloy at each gold concentration:")
for gold_ratio in gold_ratios:
    avg_num_gold = np.mean([result[0] for result in purity_results_half_mixed[gold_ratio]])
    avg_num_remaining = np.mean([result[1] for result in purity_results_half_mixed[gold_ratio]])
    avg_final_purity = np.mean([result[2] for result in purity_results_half_mixed[gold_ratio]])
    print(f"Gold ratio {gold_ratio:.2f}: Average final purity {avg_final_purity:.4f} (Gold: {avg_num_gold:.2f}, Total Remaining: {avg_num_remaining:.2f})")


# Plot purity vs starting gold concentration for the 20x20x20 half-mixed alloy
fig, ax = plt.subplots(figsize=(7, 6))
gold_concentrations = []
final_purities = []
for gold_ratio in gold_ratios:
    avg_final_purity = np.mean([result[2] for result in purity_results_half_mixed[gold_ratio]])
    gold_concentrations.append(gold_ratio)
    final_purities.append(avg_final_purity)
bars = ax.bar(gold_concentrations, final_purities, width=0.05)
ax.set_xlabel('Starting Gold Concentration')
ax.set_ylabel('Final Purity')

plt.show()
