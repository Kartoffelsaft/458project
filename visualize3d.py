import inquarting
import matplotlib.pyplot as plt
import matplotlib.widgets as widget
import numpy as np

#alloy = inquarting.create_alloy_array((20, 20, 20), 0.50, 0.45, 0.05)
alloy = inquarting.create_alloy_array_perlin((32, 32, 32), 0.8, 0.19, 4)
processed = inquarting.simulate_nitric_acid(alloy)

voxels = np.zeros(processed.shape + (3,))

voxels[:,:,:,0] = np.where(processed == inquarting.MATERIAL_GOLD, 1.0, voxels[:,:,:,0])
voxels[:,:,:,1] = np.where(processed == inquarting.MATERIAL_GOLD, 1.0, voxels[:,:,:,1])

voxels[:,:,:,0] = np.where(processed == inquarting.MATERIAL_IMPURITY, 0.3, voxels[:,:,:,0])
voxels[:,:,:,1] = np.where(processed == inquarting.MATERIAL_IMPURITY, 0.3, voxels[:,:,:,1])
voxels[:,:,:,2] = np.where(processed == inquarting.MATERIAL_IMPURITY, 0.3, voxels[:,:,:,2])

voxels[:,:,:,0] = np.where(processed == inquarting.MATERIAL_SILVER, 0.8, voxels[:,:,:,0])
voxels[:,:,:,1] = np.where(processed == inquarting.MATERIAL_SILVER, 0.8, voxels[:,:,:,1])
voxels[:,:,:,2] = np.where(processed == inquarting.MATERIAL_SILVER, 0.8, voxels[:,:,:,2])


fig = plt.figure()
ax = fig.add_axes((0, 0.2, 1.0, 0.8), projection='3d')

def update(amount):
    mask = np.zeros(processed.shape, dtype='bool')
    mask[0:amount,:,:] = True
    filled = processed != inquarting.MATERIAL_DISSOLVED_SILVER
    filled = np.logical_and(filled, mask)
    ax.clear()
    ax.voxels(filled, facecolors=voxels)

update(processed.shape[0])

cut_slider_ax = fig.add_axes((0, 0, 1.0, 0.2))
cut_slider = widget.Slider(
    ax=cut_slider_ax,
    label="cut",
    valmin=0,
    valmax=processed.shape[0],
    valinit=processed.shape[0],
    valstep=1,
)
cut_slider.on_changed(update)

plt.show()
