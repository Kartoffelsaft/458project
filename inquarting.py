import numpy as np  # Importing NumPy for array operations
from perlin_numpy import generate_fractal_noise_3d
from collections import deque  # Importing deque for efficient queue operations
import matplotlib.pyplot as plt

MATERIAL_SILVER = 0
MATERIAL_GOLD = 1
MATERIAL_DISSOLVED_SILVER = 2
MATERIAL_IMPURITY = -1

def create_alloy_array(shape, gold_ratio, silver_ratio, impurity_ratio=0.01):
    if not np.isclose(gold_ratio + silver_ratio + impurity_ratio, 1.0):
        raise ValueError("The sum of the ratios must be 1.0.")
    total_elements = np.prod(shape)  # Calculate total number of elements in the array

    num_gold = int(total_elements * gold_ratio)  # Calculate number of gold elements based on ratio
    num_silver = int(total_elements * silver_ratio)  # Calculate number of silver elements based on ratio
    num_impurity = total_elements - num_gold - num_silver  # Calculate number of impurity elements
    elements = np.array([MATERIAL_GOLD] * num_gold + [MATERIAL_SILVER] * num_silver + [MATERIAL_IMPURITY] * num_impurity)  # Create array with correct proportions
    np.random.shuffle(elements)  # Shuffle the array to randomize positions of elements

    alloy_mixture = elements.reshape(shape)  # Reshape the array to the desired shape

    return alloy_mixture # Return the created alloy mixture

def create_half_mixed_alloy(shape, gold_ratio, silver_ratio, impurity_ratio=0.01):
    if not np.isclose(gold_ratio + silver_ratio + impurity_ratio, 1.0):
        raise ValueError("The sum of the ratios must be 1.0.")
    
    total_elements = np.prod(shape)
    num_gold = int(total_elements * gold_ratio)
    num_silver = int(total_elements * silver_ratio)
    num_impurity = total_elements - num_gold - num_silver
    
    elements = np.array([1] * num_gold + [0] * num_silver + [-1] * num_impurity)
    
    # Divide the alloy into two halves
    half_elements = total_elements // 2
    
    # Fully mix one half
    np.random.shuffle(elements[:half_elements])
    
    # Leave the other half unmixed
    # (The elements are already in the correct proportion, so we don't need to do anything)
    
    alloy_mixture = elements.reshape(shape)
    return alloy_mixture

def create_alloy_array_perlin(shape, gold_ratio, silver_ratio, clump_size, octaves=1):
    clump_period = (int(shape[0] / clump_size), int(shape[1] / clump_size), int(shape[2] / clump_size))
    def gen_clump_positions(threshold):
        if threshold >= 1.0:
            return np.ones(shape)
        elif threshold <= 0:
            return np.zeros(shape)
        values = generate_fractal_noise_3d(shape, clump_period, octaves)
        value_threshold = np.sort(values.flatten())[int(threshold * values.size)]
        return values < value_threshold

    alloy = np.full(shape, MATERIAL_IMPURITY)
    alloy = np.where(gen_clump_positions(gold_ratio), MATERIAL_GOLD, alloy)
    if gold_ratio == 1:
        return alloy

    silver_threshold = silver_ratio / (1.0 - gold_ratio)
    alloy = np.where(np.logical_and(alloy != MATERIAL_GOLD, gen_clump_positions(silver_threshold)), MATERIAL_SILVER, alloy)

    return alloy

def simulate_nitric_acid(alloy):
    shape = alloy.shape
    visited = np.zeros(shape, dtype=bool)
    queue = deque()

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if alloy[x, y, z] == 0:
                    if x == 0 or x == shape[0] - 1 or y == 0 or y == shape[1] - 1 or z == 0 or z == shape[2] - 1:
                        queue.append((x, y, z))
                        visited[x, y, z] = True

    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    while queue:
        x, y, z = queue.popleft()
        if alloy[x, y, z] == 0:
            alloy[x, y, z] = 2
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if not visited[nx, ny, nz] and alloy[nx, ny, nz] == 0:
                    visited[nx, ny, nz] = True
                    queue.append((nx, ny, nz))
    return alloy


if __name__ == "__main__":
    sizes = [(5, 5, 5), (10, 10, 10), (15, 15, 15), (20, 20, 20)]
    gold_ratios = [0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8]
    impurity_ratio = 0.01
    num_simulations = 10

    purity_results = {size: {gold_ratio: [] for gold_ratio in gold_ratios} for size in sizes}
    purity_results_half_mixed = {gold_ratio: [] for gold_ratio in gold_ratios}

    for size in sizes:
        for gold_ratio in gold_ratios:
            for _ in range(num_simulations):
                alloy_mixture = create_alloy_array(size, gold_ratio, 1 - gold_ratio - impurity_ratio, impurity_ratio)
                dissolved_alloy = simulate_nitric_acid(alloy_mixture.copy())
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

    # Perform the analysis for the 10x10x10 alloy with half-mixed alloy at each gold concentration
    for gold_ratio in gold_ratios:
        for _ in range(num_simulations):
            alloy_mixture = create_half_mixed_alloy((10, 10, 10), gold_ratio, 1 - gold_ratio - impurity_ratio, impurity_ratio)
            dissolved_alloy = simulate_nitric_acid(alloy_mixture.copy())
            num_gold = np.count_nonzero(dissolved_alloy == 1)
            num_silver_dissolved = np.count_nonzero(dissolved_alloy == 2)
            num_silver = np.count_nonzero(alloy_mixture == 0)
            num_remaining = num_gold + num_silver - num_silver_dissolved
            final_purity = num_gold / num_remaining
            purity_results_half_mixed[gold_ratio].append((num_gold, num_remaining, final_purity))

    # Calculate and print the average purity for the 20x20x20 alloy with half-mixed alloy at each gold concentration
    print(f"Average purity for the 10x10x10 alloy with half-mixed alloy at each gold concentration:")
    for gold_ratio in gold_ratios:
        avg_num_gold = np.mean([result[0] for result in purity_results_half_mixed[gold_ratio]])
        avg_num_remaining = np.mean([result[1] for result in purity_results_half_mixed[gold_ratio]])
        avg_final_purity = np.mean([result[2] for result in purity_results_half_mixed[gold_ratio]])
        print(f"Gold ratio {gold_ratio:.2f}: Average final purity {avg_final_purity:.4f} (Gold: {avg_num_gold:.2f}, Total Remaining: {avg_num_remaining:.2f})")

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

    # Plot purity vs starting gold concentration for the 10x10x10 half-mixed alloy
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




########### Tests

def test_alloy_generation_accurate_probs(log):
    alloy = create_alloy_array((50, 50, 50), 0.6, 0.35, 0.05)
    log.write(alloy)

    portion_gold     = np.sum(alloy == MATERIAL_GOLD    ) / alloy.size
    portion_silver   = np.sum(alloy == MATERIAL_SILVER  ) / alloy.size
    portion_impurity = np.sum(alloy == MATERIAL_IMPURITY) / alloy.size

    log.write(f"gold actual: {portion_gold} | expected: {0.6}")
    log.write(f"silver actual: {portion_silver} | expected: {0.6}")
    log.write(f"impurity actual: {portion_impurity} | expected: {0.6}")

    return np.isclose(portion_gold, 0.6, atol=0.05) \
       and np.isclose(portion_silver, 0.35, atol=0.05) \
       and np.isclose(portion_impurity, 0.05, atol=0.02)

def test_alloy_generation_perlin_accurate_probs(log):
    alloy = create_alloy_array_perlin((64, 64, 64), 0.6, 0.35, 4)
    log.write(alloy)

    portion_gold     = np.sum(alloy == MATERIAL_GOLD    ) / alloy.size
    portion_silver   = np.sum(alloy == MATERIAL_SILVER  ) / alloy.size
    portion_impurity = np.sum(alloy == MATERIAL_IMPURITY) / alloy.size

    log.write(f"gold actual: {portion_gold} | expected: {0.6}")
    log.write(f"silver actual: {portion_silver} | expected: {0.6}")
    log.write(f"impurity actual: {portion_impurity} | expected: {0.6}")

    return np.isclose(portion_gold, 0.6, atol=0.05) \
       and np.isclose(portion_silver, 0.35, atol=0.05) \
       and np.isclose(portion_impurity, 0.05, atol=0.02)

def test_alloy_generation_perlin_pure_gold(log):
    alloy = create_alloy_array_perlin((64, 64, 64), 1.0, 0.0, 4)
    log.write(alloy)

    log.write(f"Total gold:     {np.sum(alloy == MATERIAL_GOLD    )}")
    log.write(f"Total silver:   {np.sum(alloy == MATERIAL_SILVER  )}")
    log.write(f"Total impurity: {np.sum(alloy == MATERIAL_IMPURITY)}")

    return (alloy == MATERIAL_GOLD).all()

def test_alloy_generation_perlin_pure_impurity(log):
    alloy = create_alloy_array_perlin((64, 64, 64), 0.0, 0.0, 4)
    log.write(alloy)

    log.write(f"Total gold:     {np.sum(alloy == MATERIAL_GOLD    )}")
    log.write(f"Total silver:   {np.sum(alloy == MATERIAL_SILVER  )}")
    log.write(f"Total impurity: {np.sum(alloy == MATERIAL_IMPURITY)}")

    return (alloy == MATERIAL_IMPURITY).all()

def test_simple_dissolve(log):
    alloy_before = np.array([
        [[MATERIAL_SILVER, MATERIAL_GOLD  , MATERIAL_GOLD  ],
         [MATERIAL_GOLD  , MATERIAL_GOLD  , MATERIAL_GOLD  ],
         [MATERIAL_SILVER, MATERIAL_SILVER, MATERIAL_SILVER]],

        [[MATERIAL_GOLD  , MATERIAL_GOLD  , MATERIAL_GOLD  ],
         [MATERIAL_GOLD  , MATERIAL_SILVER, MATERIAL_GOLD  ],
         [MATERIAL_SILVER, MATERIAL_GOLD  , MATERIAL_SILVER]],

        [[MATERIAL_GOLD  , MATERIAL_SILVER, MATERIAL_GOLD  ],
         [MATERIAL_GOLD  , MATERIAL_GOLD  , MATERIAL_GOLD  ],
         [MATERIAL_GOLD  , MATERIAL_GOLD  , MATERIAL_SILVER]],
    ])

    alloy_after = simulate_nitric_acid(alloy_before)

    log.write(alloy_before)
    log.write(alloy_after)

    return np.array_equal(alloy_after, np.array([
        [[MATERIAL_DISSOLVED_SILVER, MATERIAL_GOLD            , MATERIAL_GOLD            ],
         [MATERIAL_GOLD            , MATERIAL_GOLD            , MATERIAL_GOLD            ],
         [MATERIAL_DISSOLVED_SILVER, MATERIAL_DISSOLVED_SILVER, MATERIAL_DISSOLVED_SILVER]],

        [[MATERIAL_GOLD            , MATERIAL_GOLD            , MATERIAL_GOLD            ],
         [MATERIAL_GOLD            , MATERIAL_SILVER          , MATERIAL_GOLD            ],
         [MATERIAL_DISSOLVED_SILVER, MATERIAL_GOLD            , MATERIAL_DISSOLVED_SILVER]],

        [[MATERIAL_GOLD            , MATERIAL_DISSOLVED_SILVER, MATERIAL_GOLD            ],
         [MATERIAL_GOLD            , MATERIAL_GOLD            , MATERIAL_GOLD            ],
         [MATERIAL_GOLD            , MATERIAL_GOLD            , MATERIAL_DISSOLVED_SILVER]],
    ]))

def test_simple_dissolve_impurity_untouched(log):
    alloy_before = np.full((4, 4, 4), MATERIAL_IMPURITY)
    alloy_after = simulate_nitric_acid(alloy_before)

    log.write(alloy_before)
    log.write(alloy_after)

    return (alloy_after == MATERIAL_IMPURITY).all()

def test_simple_dissolve_impurity_constant(log):
    alloy_before = create_alloy_array((10, 10, 10), 0.33, 0.33, 0.34)
    alloy_after = simulate_nitric_acid(alloy_before)

    log.write(alloy_before)
    log.write(alloy_after)

    return np.array_equal((alloy_before == MATERIAL_IMPURITY), (alloy_after == MATERIAL_IMPURITY))


def test_simple_dissolve_encased_silver(log):
    alloy_before = create_alloy_array((3, 3, 3), 0.5, 0.0, 0.5)
    alloy_before[1, 1, 1] = MATERIAL_SILVER

    alloy_after = simulate_nitric_acid(alloy_before)

    log.write(alloy_before)
    log.write(alloy_after)

    return alloy_after[1, 1, 1] == MATERIAL_SILVER

def test_simple_dissolve_full_silver(log):
    alloy_before = np.full((4, 4, 4), MATERIAL_SILVER)
    alloy_after = simulate_nitric_acid(alloy_before)

    log.write(alloy_before)
    log.write(alloy_after)

    return (alloy_before == MATERIAL_SILVER).all() and (alloy_after == MATERIAL_DISSOLVED_SILVER).all()
    
