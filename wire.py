import inquarting
import numpy as np
import matplotlib.pyplot as plt

DIAMETER = 6
START_PURITY = 0.45
lengths = list(range(DIAMETER, 200))
purities = []

for length_i in range(len(lengths)):
    wire = inquarting.create_alloy_array((DIAMETER, DIAMETER, lengths[length_i]), START_PURITY, 1-START_PURITY, 0.0)
    wire = inquarting.simulate_nitric_acid(wire)
    purity = np.sum(wire == inquarting.MATERIAL_GOLD) / np.sum(wire != inquarting.MATERIAL_DISSOLVED_SILVER)
    purities.append(purity)

plt.scatter(lengths, purities)
plt.xlabel("Length of wire")
plt.ylabel("Result purity")
plt.title(f"Purity by Wire Length (Start purity: {START_PURITY}, Diameter: {DIAMETER})")
plt.show()

