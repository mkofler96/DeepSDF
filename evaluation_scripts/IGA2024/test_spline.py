import splinepy as sp

test = sp.BSpline(
    degrees=[0],
    control_points=[
        [0],[1],[2]
    ],
    knot_vectors=[[0, .33, 0.66, 1]]
)

import numpy as np

# Create the middle part with linspace and add the repeated values at the ends
array = np.linspace(-0.5, 0.5, 3)
custom_array = np.concatenate(([-1, -1], array, [1, 1]))

print(custom_array)

test.elevate_degrees([0])