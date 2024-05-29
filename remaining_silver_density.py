import inquarting
import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 800
ALLOY_SIZE = 16
ALLOY_SHAPE = (ALLOY_SIZE, ALLOY_SIZE, ALLOY_SIZE)
START_PURITY = 0.8

alloys = np.zeros((SAMPLES,) + ALLOY_SHAPE)

for i in range(SAMPLES):
    alloys[i] = inquarting.create_alloy_array(ALLOY_SHAPE, START_PURITY, 1-START_PURITY, 0.0)
    alloys[i] = inquarting.simulate_nitric_acid(alloys[i])

silver_by_pos = np.sum(alloys == inquarting.MATERIAL_SILVER, axis=0)
plt.imshow(silver_by_pos[ALLOY_SIZE//2, :, :] / SAMPLES, cmap='gray')
plt.colorbar()
plt.title(f"Likelihood of silver for a given position (Start purity: {START_PURITY}, {SAMPLES} samples")
plt.show()
