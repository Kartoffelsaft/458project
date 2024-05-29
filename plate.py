import inquarting
import numpy as np
import matplotlib.pyplot as plt

THICKNESS = 6
START_PURITY = 0.45
lengths = list(range(THICKNESS, 200))
purities = []

for length_i in range(len(lengths)):
    plate = inquarting.create_alloy_array((THICKNESS, length_i, lengths[length_i]), START_PURITY, 1-START_PURITY, 0.0)
    plate = inquarting.simulate_nitric_acid(plate)
    purity = np.sum(plate == inquarting.MATERIAL_GOLD) / np.sum(plate != inquarting.MATERIAL_DISSOLVED_SILVER)
    purities.append(purity)

plt.scatter(lengths, purities)
plt.xlabel("Length/Width of plate")
plt.ylabel("Result purity")
plt.title(f"Purity by Wire Length (Start purity: {START_PURITY}, Thickness: {THICKNESS})")
plt.show()

