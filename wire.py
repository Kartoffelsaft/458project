import inquarting
import numpy as np
import matplotlib.pyplot as plt

DIAMETER = 16
lengths = list(range(16, 200))
purities = []

for length_i in range(len(lengths)):
    wire = inquarting.create_alloy_array((DIAMETER, DIAMETER, lengths[length_i]), 0.4, 0.6, 0.0)
    wire = inquarting.simulate_nitric_acid(wire)
    purity = np.sum(wire == inquarting.MATERIAL_GOLD) / np.sum(wire != inquarting.MATERIAL_DISSOLVED_SILVER)
    purities.append(purity)

plt.scatter(lengths, purities)
plt.show()

