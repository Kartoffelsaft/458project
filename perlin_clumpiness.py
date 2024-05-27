import inquarting
import numpy as np
import matplotlib.pyplot as plt

clump_sizes = [2, 4, 6, 8, 12]
SAMPLES_PER_CLUMP_SIZE = 20
ALLOY_SIZE = 48
ALLOY_SHAPE = (ALLOY_SIZE, ALLOY_SIZE, ALLOY_SIZE)
samples = np.zeros((len(clump_sizes), SAMPLES_PER_CLUMP_SIZE) + ALLOY_SHAPE)

for clump_i in range(len(clump_sizes)):
    print(clump_sizes[clump_i])
    for sample_i in range(SAMPLES_PER_CLUMP_SIZE):
        samples[clump_i][sample_i] = inquarting.create_alloy_array_perlin(ALLOY_SHAPE, 0.45, 0.55, clump_sizes[clump_i], 2)
        samples[clump_i][sample_i] = inquarting.simulate_nitric_acid(samples[clump_i][sample_i])

total_remaining_by_sample = np.sum(samples != inquarting.MATERIAL_DISSOLVED_SILVER, axis=(2, 3, 4))
print(total_remaining_by_sample)
total_gold_by_sample = np.sum(samples == inquarting.MATERIAL_GOLD, axis=(2, 3, 4))
print(total_gold_by_sample)
sample_purities = total_gold_by_sample / total_remaining_by_sample

scatter_xs = np.repeat(clump_sizes, SAMPLES_PER_CLUMP_SIZE)
scatter_ys = sample_purities.flatten()

plt.scatter(scatter_xs, scatter_ys)
plt.show()

