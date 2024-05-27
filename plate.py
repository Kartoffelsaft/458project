import inquarting
import numpy as np
import matplotlib.pyplot as plt

THICKNESS = 16
lengths = list(range(16, 200))
purities = []

for length_i in range(len(lengths)):
    plate = inquarting.create_alloy_array((THICKNESS, length_i, lengths[length_i]), 0.4, 0.6, 0.0)
    plate = inquarting.simulate_nitric_acid(plate)
    purity = np.sum(plate == inquarting.MATERIAL_GOLD) / np.sum(plate != inquarting.MATERIAL_DISSOLVED_SILVER)
    purities.append(purity)

plt.scatter(lengths, purities)
plt.show()

