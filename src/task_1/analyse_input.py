"""
This script:
    1. Analyzes the input depth map and print out the following information:
        1.1. Box top surface mean depth
        1.2. Base mean depth
        1.3. Floor mean depth
        1.4. Box top surface area
        1.5. Density of points
    2. Generates images for demonstration
    3. Extracts 3D points (xyzs) of interest for pose estimation
This script produces:
    1. pictures/task1_box_depth_clustered.png: picture demonstrates the feature with box's height
    2. pictures/task1_box_top_surface.png: picture demonstrates the box mask
    3. pictures/task1_depth_clustered_image.png: picture demonstrates depth clusters
    4. pictures/task1_depth_image.png: visualize of input depth image
    5. data/task_1/processed/refined_scene_xyzs.npdata.npy: points of interest in the scene (from sensor)
    6. data/task_1/processed/scene_xyzs.npdata.npy: points extracted from the depth image for visualize
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------
# Load and preprocess the data
# ---------------------------------------------------------------------
# Load depth image
depth_image = np.load("data/task_1/input/one-box.depth.npdata.npy")
# print("Depth image shape:", depth_image.shape)  # 1544 x 2064

# Load intrinsics
intrinsics_matrix = np.load("data/task_1/input/intrinsics.npy")
# print("Intrinsics Matrix:", intrinsics_matrix)
f_x = intrinsics_matrix[0, 0]
f_y = intrinsics_matrix[1, 1]
c_x = intrinsics_matrix[0, 2]
c_y = intrinsics_matrix[1, 2]
# print(f"Focal Lengths: f_x = {f_x}, f_y = {f_y}")
# print(f"Principal Point: c_x = {c_x}, c_y = {c_y}")

# Load extrinsics
extrinsics_matrix = np.load("data/task_1/input/extrinsics.npy")
# print("Extrinsics Matrix:", extrinsics_matrix)

# Visualize depth image
plt.close('all')
plt.imshow(depth_image, cmap='viridis')
plt.colorbar(label='Depth (meters)')
plt.title('Depth Image')
plt.savefig("pictures/task1_depth_image.png")

# ---------------------------------------------------------------------
# Apply K-mean to cluster the depth
# ---------------------------------------------------------------------
# Step 1: Reshape for K-means clustering
height, width = depth_image.shape
pixels = depth_image.reshape(-1, 1)  # Shape: (1544*2064, 1)

# Step 2: Apply K-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(pixels)
# print(labels.shape)

# Step 3: Reshape labels to image shape
clustered_image = labels.reshape(height, width)

# Step 4: Select the cluster containing the box
cluster_centers = kmeans.cluster_centers_.flatten()
cluster_depths = [(i, center) for i, center in enumerate(cluster_centers)]
cluster_depths.sort(key=lambda x: x[1])  # Sort by depth
# print("Sorted clusters (index, depth):", cluster_depths)
box_cluster_id = cluster_depths[1][0]
box_depth = cluster_depths[1][1]
# print("Box depth: ", box_depth)

# Step 5: Visualize as image
plt.close('all')
plt.imshow(clustered_image, cmap='tab10')
cbar = plt.colorbar(ticks=range(n_clusters))
cbar.set_label('Cluster ID')
plt.title('Clustered Depth Image')
plt.xlabel('Width')
plt.ylabel('Height')
plt.savefig("pictures/task1_depth_clustered_image.png")

# ---------------------------------------------------------------------
# Extract top area of the box
# ---------------------------------------------------------------------
# Step 1: Contour detection on the cluster mask
# Create mask for the selected cluster
box_cluster_mask = (clustered_image == box_cluster_id)
# Convert mask to uint8 for contour detection
cluster_mask_uint8 = (box_cluster_mask * 255).astype(np.uint8)
# Find contours in the cluster
contours, _ = cv2.findContours(cluster_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 2: Filter contours to find the square-shaped box
square_contour = None
best_square_score = float('inf')  # Lower is better (closer to square)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if w > h else h / w  # Ratio of longer to shorter side
    area = cv2.contourArea(contour)
    
    # Check if contour is square-like and large enough
    if 0.9 <= aspect_ratio <= 1.1 and area > 100:
        square_score = abs(1 - aspect_ratio)
        if square_score < best_square_score:
            best_square_score = square_score
            square_contour = contour
# print("Square Contour Shape:", square_contour.shape)

# Create mask for the square box
box_mask = np.zeros_like(depth_image, dtype=np.uint8)
cv2.drawContours(box_mask, [square_contour], -1, 1, -1)
pixel_positions = np.where(box_mask == 1)
# print(pixel_positions)


# Step 3: Calculate mean depth of the box's top surface
box_depth_values = depth_image[box_mask.astype(bool)]
mean_depth = box_depth_values.mean()
# print(f"Mean Depth of Box Top Surface: {mean_depth:.4f} units")

# Step 4: Calculate area of the box's top surface
area_pixels = cv2.contourArea(square_contour)
# print(f"Area of Box Top Surface: {area_pixels} pixels")

# Convert area to physical units
scale_x = mean_depth / f_x
scale_y = mean_depth / f_y
scale_meters_per_pixel = (scale_x + scale_y) / 2  # Assuming the scale is the same in both directions (average)
if scale_meters_per_pixel:
    area_meters = area_pixels * (scale_meters_per_pixel ** 2)
    # print(f"Area of Box Top Surface: {area_meters:.4f} square meters")

# Visualize the detected box
plt.close('all')
plt.figure(figsize=(10, 8))
plt.imshow(depth_image, cmap='gray')
plt.imshow(box_mask, cmap='Reds', alpha=0.5)
plt.axis('off')
plt.title('Depth Image with Square Box Top Surface Highlighted')
plt.savefig('pictures/task1_box_top_surface.png')

# Visualize the selected cluster for reference
plt.close('all')
plt.figure(figsize=(10, 8))
plt.imshow(depth_image, cmap='gray')
plt.imshow(box_cluster_mask, cmap='Blues', alpha=0.5)
plt.axis('off')
plt.title('Selected Cluster (Before Contour Filtering)')
plt.savefig('pictures/task1_box_depth_clustered.png')

# ---------------------------------------------------------------------
# Extract points (xyzs) belong to top area of the box
# ---------------------------------------------------------------------
# Step 1: Compute 3D coordinates
height, width = depth_image.shape
u, v = np.meshgrid(np.arange(width), np.arange(height))
z = depth_image  # Depth values
x = (u - c_x) * z / f_x
y = (v - c_y) * z / f_y

# Step 2: Convert depth image to xyzs
scene_xyzs = np.stack((x, y, z), axis=0)

# Step 3: Crop area of interest
xyzs_of_interest = scene_xyzs[:, pixel_positions[0], pixel_positions[1]]
# print(xyzs_of_interest.shape)

# Step 4: Save files
np.save("data/task_1/processed/scene_xyzs.npdata.npy", scene_xyzs)
np.save("data/task_1/processed/refined_scene_xyzs.npdata.npy", xyzs_of_interest)