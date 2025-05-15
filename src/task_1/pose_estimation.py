import numpy as np
import open3d as o3d

from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from pydrake.all import DiagramBuilder, StartMeshcat, PointCloud, RigidTransform, Rgba, RotationMatrix

def least_squares_transform(scene, model) -> RigidTransform:
    """
    Calculates the least-squares best-fit transform that maps corresponding
    points scene to model.
    Args:
      scene: 3xN numpy array of corresponding points
      model: 3xN numpy array of corresponding points
    Returns:
      X_BA: A RigidTransform object that maps point_cloud_A on to point_cloud_B
            such that
                        X_BA.multiply(model) ~= scene,
    """
    # For convenction, reshape the input if necessary
    if scene.shape[0] == 3:
        scene = scene.T  # [Nx3]
    if model.shape[0] == 3:
        model = model.T  # [Nx3]

    # Central points
    scene_bar = np.mean(scene, axis=0)  # [3x1]
    model_bar = np.mean(model, axis=0)  # [3x1]

    # Shift points (centralized point clouds)
    scene = scene - scene_bar  # [Nx3]
    model = model - model_bar  # [Nx3]

    # Cross-covariance matrix
    W = scene.T @ model  # [3x3]

    # Optimized roration matrix R*
    U, _, Vt = np.linalg.svd(W)
    R = U @ Vt  # [3x3]
    # Check for reflection
    if np.linalg.det(R) < 0:
        # print("Reflection detected -> flipping sign of last column of Vt")
        Vt[-1, :] *= -1
        R = U @ Vt

    # Optimized translation p*
    p = scene_bar - R @ model_bar

    return RigidTransform(RotationMatrix(R), p)

def nearest_neighbors(scene, model):
    """
    Find the nearest (Euclidean) neighbor in model for each
    point in scene
    Args:
        scene: 3xN numpy array of points
        model: 3xN numpy array of points
    Returns:
        distances: (N, ) numpy array of Euclidean distances from each point in
            scene to its nearest neighbor in model.
        indices: (N, ) numpy array of the indices in model of each
            scene point's nearest neighbor - these are the c_i's
    """
    kdtree = KDTree(model.T)

    distances, indices = kdtree.query(scene.T, k=1)

    return distances.flatten(), indices.flatten()

def icp(scene, model, max_iterations=1000, tolerance=1e-10000):
    """
    Perform ICP to return the correct relative transform between two set of points.
    Args:
        scene: 3xN numpy array of points
        model: 3xN numpy array of points
        max_iterations: max amount of iterations the algorithm can perform.
        tolerance: tolerance before the algorithm converges.
    Returns:
      X_BA: A RigidTransform object that maps point_cloud_A on to point_cloud_B
            such that
                        X_BA.multiply(point_cloud_A) ~= point_cloud_B,
      mean_error: Mean of all pairwise distances.
      num_iters: Number of iterations it took the ICP to converge.
    """
    X_BA = RigidTransform()

    mean_error = 0
    num_iters = 0
    prev_error = 0

    transformed_model = model.copy()

    while True:
        num_iters += 1
        
        # Step 1: Find correspondences using the transformed model
        distances, c_hat = nearest_neighbors(scene, transformed_model)
        # c_hat should return indices of model points corresponding to scene points
        corresponding_points = transformed_model[:, c_hat]

        # Step 2: Compute the transform between scene and corresponding points
        delta_X = least_squares_transform(scene, corresponding_points)

        # Step 3: Update the transformed model
        X_BA = delta_X @ X_BA
        # X_BA = delta_X

        # Step 4: Apply the updated transform to the model points
        transformed_model = X_BA @ model

        # Step 5: Compute mean error based on correspondences
        mean_error = np.mean(np.linalg.norm(scene - corresponding_points, axis=0))

        # Break condition
        if abs(mean_error - prev_error) < tolerance or num_iters >= max_iterations:
            break

        prev_error = mean_error

    return X_BA, mean_error, num_iters


# ---------------------------------------------------------------------
# Pose estimation using geometry method
# ---------------------------------------------------------------------
# Load model xyzs
model_xyzs = np.load("data/task_1/processed/model_xyzs.npdata.npy")
model_cloud = PointCloud(model_xyzs.shape[1])
model_cloud.mutable_xyzs()[:] = model_xyzs

# Load scene xyzs
scene_xyzs = np.load("data/task_1/processed/refined_scene_xyzs.npdata.npy")
scene_cloud = PointCloud(scene_xyzs.shape[1])
scene_cloud.mutable_xyzs()[:] = scene_xyzs

# Outlier removal
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scene_xyzs.T)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
scene_xyzs = np.asarray(pcd.points).T

# Voxelize down-sample.
down_sampled_scene_cloud = scene_cloud.VoxelizedDownSample(voxel_size=0.003)
down_sampled_model_cloud = model_cloud.VoxelizedDownSample(voxel_size=0.003)

# Start ICP algorithm
X_BA, mean_error, num_iters = icp(down_sampled_scene_cloud.xyzs(), down_sampled_model_cloud.xyzs())
print("Tranformation from model point cloud to scene point cloud:", X_BA)
print("Max iterations:", num_iters)
print("Tolerance:", mean_error)

# ---------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------
# Start the visualizer.
meshcat = StartMeshcat()

# Load the box model
box_model_xyzs = np.load("data/task_1/processed/box_model_xyzs.npdata.npy")
box_model_cloud = PointCloud(box_model_xyzs.shape[1])

# Load the scene points
scene_xyzs = np.load("data/task_1/processed/scene_xyzs.npdata.npy")
scene_cloud = PointCloud(box_model_xyzs.shape[1])


meshcat.Delete()
meshcat.SetProperty("/Background", "visible", False)
meshcat.SetProperty("/Grid", "visible", False)
meshcat.SetObject("scene_cloud", scene_cloud, point_size=0.01, rgba=Rgba(1.0, 0, 0))
meshcat.SetObject("model_cloud", box_model_cloud, point_size=0.01, rgba=Rgba(0, 0, 1.0))
meshcat.SetTransform('model_cloud', X_BA)