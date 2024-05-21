import numpy as np  # Importing NumPy for array operations
from collections import deque  # Importing deque for efficient queue operations

MATERIAL_SILVER = 0
MATERIAL_GOLD = 1
MATERIAL_IMPURITY = -1

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
    return alloy_mixture  # Return the created alloy mixture

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

# Define the shape of the array
shape = (20, 20, 20)

# Define the gold ratios to test, keeping 1% impurity and adjusting silver
gold_ratios = [0.10, 0.25, 0.50, 0.75, 0.90]
impurity_ratio = 0.01  # Impurity ratio is fixed at 1%

# Test different ratios
for gold_ratio in gold_ratios:  # Loop through each gold ratio
    silver_ratio = 1 - gold_ratio - impurity_ratio  # Calculate silver ratio based on gold and impurity
    alloy_mixture = create_alloy_array(shape, gold_ratio, silver_ratio, impurity_ratio)  # Create alloy mixture
    dissolved_alloy = simulate_nitric_acid(alloy_mixture.copy())  # Simulate nitric acid dissolution
    
    # Print initial and dissolved alloy mixtures for each ratio
    print(f"Initial Alloy Mixture (1 for Gold, 0 for Silver, -1 for Impurity) with ratios {gold_ratio}, {silver_ratio}, {impurity_ratio}:")
    print(alloy_mixture)
    print()
    
    print(f"Dissolved Alloy Mixture (1 for Gold, 2 for Dissolved Silver, -1 for Impurity) with ratios {gold_ratio}, {silver_ratio}, {impurity_ratio}:")
    print(dissolved_alloy)
    print()  # Add a newline for readability between different ratio outputs
