import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read the CSV file
df = pd.read_csv('sequences_summary.csv')

# Step 2: Directly use "Start Action" for pivoting
pivot_table = df.pivot(index='Start Action', columns='Sequence Length', values='Number of Sequences').fillna(0)

# Preparing data for the plot
X = np.arange(len(pivot_table.columns))
Y = np.arange(len(pivot_table.index))
X, Y = np.meshgrid(X, Y)
Z = pivot_table.to_numpy()

#Z_log = np.log10(Z + 1)
# Step 4: Create the 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#surf = ax.plot_surface(X, Y, Z_log, cmap='coolwarm')
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
# Labeling
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Start Action')
#ax.set_zlabel(' Log10 Number of Sequences')
ax.set_zlabel('Number of Sequences')
ax.set_title('3D Surface Plot of Sequences Summary')

# Adjusting the y-axis to show string labels
ax.set_yticks(np.arange(len(pivot_table.index)))
ax.set_yticklabels(pivot_table.index)

# Adjusting the x-axis to show integer labels for Sequence Length
sequence_lengths = pivot_table.columns.astype(int)
ax.set_xticks(np.arange(len(sequence_lengths)))
ax.set_xticklabels(sequence_lengths)

# Add a color bar to indicate the values of Z
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()