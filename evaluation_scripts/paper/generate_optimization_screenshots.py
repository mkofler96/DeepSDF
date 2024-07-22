
import numpy as np
import gustaf as gus
import splinepy as sp
import matplotlib.pyplot as plt

init_mesh = gus.io.meshio.load("simulations/optimization_mimi/simulation_0/volume.inp")
vedo_showable = gus.show(init_mesh, interactive=False, c="#EDEDED", lw=1)
vedo_showable.screenshot("evaluation_scripts/wccm/initial_mesh.png")

init_mesh = gus.io.meshio.load("simulations/optimization_mimi/simulation_218/volume218.inp")
vedo_showable = gus.show(init_mesh, interactive=False, c="#EDEDED", lw=1)
vedo_showable.screenshot("evaluation_scripts/wccm/optimized_mesh.png")

import re

# Regular expression pattern to extract compliance and volume
pattern = r"compliance ([\d.]+) and volume ([\d.]+)"

# Lists to store extracted values
compliance_values = []
volume_values = []

# Open the file and process each line
with open('simulations/optimization_mimi/optimization_logs.log', 'r') as file:
    for line in file:
        # Find matches
        matches = re.search(pattern, line)
        if matches:
            # Extract and store the compliance and volume values
            compliance = float(matches.group(1))
            volume = float(matches.group(2))
            compliance_values.append(compliance)
            volume_values.append(volume)

# Print the extracted values
fig, ax = plt.subplots(1,1, figsize=(12/2.5,8/2.5))

ax.plot(np.array(compliance_values[::2])/compliance_values[0], c="#007E71")
ax.plot(np.array(volume_values[::2])/6, c="#E18922")
ax.legend(["Objective (Compliance)", "Constraint (Volume)"])
ax.grid()
ax.set_xlabel("Iteration")
ax.set_ylabel("Normalized Objective/Constraint")
ax.set_ylim(0,1.1)
plt.savefig("evaluation_scripts/wccm/optimization_progress.png", dpi=300, bbox_inches='tight')