"""
This script generate model points which is neccessary for geometric pose estimation methods.
This script produces:
    1. data/task_1/processed/model_xyzs.npdata.npy: model for icp
    2. data/task_1/processed/box_model_xyzs.npdata.npy: box model for visualize
"""

import numpy as np
from pydrake.all import PointCloud, StartMeshcat, Rgba

# Box dimensions
width = 0.2963  # x-direction (m)
length = 0.2963  # y-direction (m)
height = 0.35   # z-direction (m)

# Sampling density (~0.001 m spacing)
nx = int(np.ceil(width / 0.001))  # ~30 points
ny = int(np.ceil(length / 0.001)) # ~30 points
nz = int(np.ceil(height / 0.001)) # ~35 points

# Box boundaries (centered at origin)
x_min, x_max = -width / 2, width / 2   # [-0.14815, 0.14815]
y_min, y_max = -length / 2, length / 2 # [-0.14815, 0.14815]
z_min, z_max = -height / 2, height / 2 # [-0.175, 0.175]

# Generate points for each face
points = []
points_model = []

# Top face (z = z_max)
x_top = np.linspace(x_min, x_max, nx)
y_top = np.linspace(y_min, y_max, ny)
X_top, Y_top = np.meshgrid(x_top, y_top)
Z_top = np.full_like(X_top, z_max)
points_model.append(np.vstack([X_top.ravel(), Y_top.ravel(), Z_top.ravel()]))

# Bottom face (z = z_min)
Z_bottom = np.full_like(X_top, z_min)
points.append(np.vstack([X_top.ravel(), Y_top.ravel(), Z_bottom.ravel()]))
points_model.append(np.vstack([X_top.ravel(), Y_top.ravel(), Z_bottom.ravel()]))

# Front face (y = y_max)
x_front = np.linspace(x_min, x_max, nx)
z_front = np.linspace(z_min, z_max, nz)
X_front, Z_front = np.meshgrid(x_front, z_front)
Y_front = np.full_like(X_front, y_max)
points_model.append(np.vstack([X_front.ravel(), Y_front.ravel(), Z_front.ravel()]))

# Back face (y = y_min)
Y_back = np.full_like(X_front, y_min)
points_model.append(np.vstack([X_front.ravel(), Y_back.ravel(), Z_front.ravel()]))

# Left face (x = x_min)
y_left = np.linspace(y_min, y_max, ny)
z_left = np.linspace(z_min, z_max, nz)
Y_left, Z_left = np.meshgrid(y_left, z_left)
X_left = np.full_like(Y_left, x_min)
points_model.append(np.vstack([X_left.ravel(), Y_left.ravel(), Z_left.ravel()]))

# Right face (x = x_max)
X_right = np.full_like(Y_left, x_max)
points_model.append(np.vstack([X_right.ravel(), Y_left.ravel(), Z_left.ravel()]))

# Combine all points into a single (3, N) array
model_xyzs = np.hstack(points)
box_model_xyzs = np.hstack(points_model)

# Save to file
np.save("data/task_1/processed/model_xyzs.npdata.npy", model_xyzs)
np.save("data/task_1/processed/box_model_xyzs.npdata.npy", box_model_xyzs)