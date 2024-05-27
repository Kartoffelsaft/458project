import inquarting
import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 800
ALLOY_SIZE = 16
ALLOY_SHAPE = (ALLOY_SIZE, ALLOY_SIZE, ALLOY_SIZE)

alloys = np.zeros((SAMPLES,) + ALLOY_SHAPE)

for i in range(SAMPLES):
    alloys[i] = inquarting.create_alloy_array(ALLOY_SHAPE, 0.8, 0.2, 0.0)
    alloys[i] = inquarting.simulate_nitric_acid(alloys[i])

silver_by_pos = np.sum(alloys == inquarting.MATERIAL_SILVER, axis=0)
plt.imshow(silver_by_pos[ALLOY_SIZE//2, :, :])
plt.show()
