import numpy as np  # Importing NumPy for array operations
from collections import deque  # Importing deque for efficient queue operations

MATERIAL_SILVER = 0
MATERIAL_GOLD = 1
MATERIAL_IMPURITY = -1
MATERIAL_EMPTY = -2

DISSOLVE_RATE = 0.3
ACID_DIFFUSION_RATE = 0.7

def create_alloy_array(shape, gold_ratio, silver_ratio, impurity_ratio=0.01):
    # Function to create the initial alloy mixture based on specified ratios
    if not np.isclose(gold_ratio + silver_ratio + impurity_ratio, 1.0):  # Check if ratios sum to 1
        raise ValueError("The sum of the ratios must be 1.")  # Raise an error if ratios are invalid

    total_elements = np.prod(shape)  # Calculate total number of elements in the array

    num_gold = int(total_elements * gold_ratio)  # Calculate number of gold elements based on ratio
    num_silver = int(total_elements * silver_ratio)  # Calculate number of silver elements based on ratio
    num_impurity = total_elements - num_gold - num_silver  # Calculate number of impurity elements
    elements = np.array([MATERIAL_GOLD] * num_gold + [MATERIAL_SILVER] * num_silver + [MATERIAL_IMPURITY] * num_impurity)  # Create array with correct proportions
    np.random.shuffle(elements)  # Shuffle the array to randomize positions of elements

    alloy_mixture = elements.reshape(shape)  # Reshape the array to the desired shape

    return alloy_mixture, np.zeros(shape), np.zeros(shape)  # Return the created alloy mixture

def generate_boundraries(alloy, acid_content):
    if alloy.shape != acid_content.shape:
        raise ValueError("simulation variables should be the same shape")
    b_alloy = np.full([d+2 for d in alloy.shape], MATERIAL_EMPTY)
    b_alloy[1:-1, 1:-1, 1:-1] = alloy
    b_acid = np.ones([d+2 for d in acid_content.shape])
    b_acid[1:-1, 1:-1, 1:-1] = acid_content

    return b_alloy, b_acid

def simulate_nitric_acid_step(alloy, acid_content, dissolusion):
    return np.array([]), np.array([]), np.array([]) # TODO

def simulate_nitric_acid(alloy):
    # Function to simulate nitric acid dissolving silver in the alloy
    shape = alloy.shape  # Get the shape of the alloy array
    visited = np.zeros(shape, dtype=bool)  # Create an array to track visited positions
    queue = deque()  # Initialize a deque to store positions for BFS traversal

    # Initialize the queue with boundary silver positions
    for x in range(shape[0]):  # Iterate over x-axis
        for y in range(shape[1]):  # Iterate over y-axis
            for z in range(shape[2]):  # Iterate over z-axis
                if alloy[x, y, z] == MATERIAL_SILVER:  # If position contains silver (0)
                    if x == 0 or x == shape[0] - 1 or y == 0 or y == shape[1] - 1 or z == 0 or z == shape[2] - 1:
                        # Check if position is at the boundary
                        queue.append((x, y, z))  # Add boundary silver position to queue
                        visited[x, y, z] = True  # Mark position as visited

    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]  # Define possible directions for movement

    while queue:  # Continue while queue is not empty
        x, y, z = queue.popleft()  # Get position from the front of the queue

        if alloy[x, y, z] == MATERIAL_SILVER:  # If position contains silver
            alloy[x, y, z] = 2  # Mark the silver as dissolved (2 represents dissolved silver)

        for dx, dy, dz in directions:  # Iterate over possible directions
            nx, ny, nz = x + dx, y + dy, z + dz  # Calculate new position in the direction
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                # Check if new position is within bounds
                if not visited[nx, ny, nz] and alloy[nx, ny, nz] == MATERIAL_SILVER:
                    # Check if new position is unvisited and contains silver
                    visited[nx, ny, nz] = True  # Mark new position as visited
                    queue.append((nx, ny, nz))  # Add new position to queue for further processing

    return alloy  # Return the alloy after simulation

if __name__ == "__main__":
    # Define the shape of the array
    shape = (20, 20, 20)

    # Define the gold ratios to test, keeping 1% impurity and adjusting silver
    gold_ratios = [0.10, 0.25, 0.50, 0.75, 0.90]
    impurity_ratio = 0.01  # Impurity ratio is fixed at 1%

    # Test different ratios
    for gold_ratio in gold_ratios:  # Loop through each gold ratio
        silver_ratio = 1 - gold_ratio - impurity_ratio  # Calculate silver ratio based on gold and impurity
        alloy_mixture, acid_content, dissolusion = create_alloy_array(shape, gold_ratio, silver_ratio, impurity_ratio)  # Create alloy mixture
        dissolved_alloy = simulate_nitric_acid(alloy_mixture.copy())  # Simulate nitric acid dissolution
        
        # Print initial and dissolved alloy mixtures for each ratio
        print(f"Initial Alloy Mixture (1 for Gold, 0 for Silver, -1 for Impurity) with ratios {gold_ratio}, {silver_ratio}, {impurity_ratio}:")
        print(alloy_mixture)
        print()
        
        print(f"Dissolved Alloy Mixture (1 for Gold, 2 for Dissolved Silver, -1 for Impurity) with ratios {gold_ratio}, {silver_ratio}, {impurity_ratio}:")
        print(dissolved_alloy)
        print()  # Add a newline for readability between different ratio outputs



########### Tests

def test_alloy_generation_accurate_probs(log):
    alloy, _, _ = create_alloy_array((50, 50, 50), 0.6, 0.35, 0.05)
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

def test_generate_boundraries(log):
    alloy = np.array([
        [[ 0, 1, -1],
         [-1, 0, -1],
         [ 0, 1,  1]],
        [[ 0, 1, -1],
         [-1, 0, -1],
         [ 0, 1,  1]],
        [[ 0, 1, -1],
         [-1, 0, -1],
         [ 0, 1,  1]],
    ])
    acid = np.array([
        [[1.0, 1.0, 1.0],
         [1.0, 1.0, 0.5],
         [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0],
         [1.0, 1.0, 0.5],
         [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0],
         [1.0, 1.0, 0.5],
         [1.0, 1.0, 1.0]],
    ])

    b_alloy, b_acid = generate_boundraries(alloy, acid)

    log.write(alloy)
    log.write(b_alloy)
    log.write(acid)
    log.write(b_acid)

    alloy_correct = np.array_equal(b_alloy, np.array([
        [[-2, -2, -2, -2, -2],
         [-2, -2, -2, -2, -2],
         [-2, -2, -2, -2, -2],
         [-2, -2, -2, -2, -2],
         [-2, -2, -2, -2, -2]],
        [[-2, -2, -2, -2, -2],
         [-2,  0,  1, -1, -2],
         [-2, -1,  0, -1, -2],
         [-2,  0,  1,  1, -2],
         [-2, -2, -2, -2, -2]],
        [[-2, -2, -2, -2, -2],
         [-2,  0,  1, -1, -2],
         [-2, -1,  0, -1, -2],
         [-2,  0,  1,  1, -2],
         [-2, -2, -2, -2, -2]],
        [[-2, -2, -2, -2, -2],
         [-2,  0,  1, -1, -2],
         [-2, -1,  0, -1, -2],
         [-2,  0,  1,  1, -2],
         [-2, -2, -2, -2, -2]],
        [[-2, -2, -2, -2, -2],
         [-2, -2, -2, -2, -2],
         [-2, -2, -2, -2, -2],
         [-2, -2, -2, -2, -2],
         [-2, -2, -2, -2, -2]],
    ]))
    acid_correct = np.allclose(b_acid, np.array([
        [[1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 0.5, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 0.5, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 0.5, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0]],
    ]))

    return alloy_correct and acid_correct

def test_dissolve_line(log):
    # Situation where we have an ingot like:
    #
    # IIII
    # ISSE
    # IIII
    #
    # and acid is starting to get in (via right side in illustration). This 
    # tests whether the rightmost silver starts to dissolve appropriately, and
    # whether the empty tile starts to get filled with acid from outside.

    alloy = np.full((3, 3, 4), MATERIAL_IMPURITY)
    alloy[1, 1, 1:3] = MATERIAL_SILVER
    alloy[1, 1, 3] = MATERIAL_EMPTY

    acid = np.zeros((3, 3, 4))
    acid[1, 1, 3] = 0.5

    dissolusion = np.zeros((3, 3, 4))
    dissolusion[1, 1, 3] = 1.0

    n_alloy, n_acid, n_dissolusion = simulate_nitric_acid_step(alloy, acid, dissolusion)

    log.write(f"Alloy before: {alloy}")
    log.write(f"Alloy after: {n_alloy}")
    log.write(f"Acid before: {acid}")
    log.write(f"Acid after: {n_acid}")
    log.write(f"Dissolusion before: {dissolusion}")
    log.write(f"Dissolusion after: {n_dissolusion}")

    # there is 0.5 acid adjacent to this silver tile, dissolves relative to it
    silver_dissolving_correctly = np.isclose(n_dissolusion[1, 1, 2], DISSOLVE_RATE * 0.5)

    # acid should lose the amount that reacted to the silver, but gain some
    # amount from the outside (would theoretically lose some to diffusion too
    # but all adjacent tiles either don't have space or have full acid content
    acid_diffusing_correctly = np.isclose(n_acid[1, 1, 3], acid[1, 1, 3] - DISSOLVE_RATE*0.5 + ACID_DIFFUSION_RATE*1.0)

    return silver_dissolving_correctly and acid_diffusing_correctly
