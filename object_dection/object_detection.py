import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def detect_nodes_and_adjacency(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")
    
    # Invert the image if necessary (assuming white lines on black background)
    if np.mean(img) > 127:
        img = 255 - img
    
    # Apply slight Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Use adaptive thresholding
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to find nodes
    nodes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 10 < area < 200:  # Adjust these thresholds as needed
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                nodes.append((cX, cY))
    
    # Remove duplicate nodes (if any)
    nodes = list(set(nodes))
    
    print(f"Number of nodes detected: {len(nodes)}")
    
    # Create a visualization image
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw detected nodes
    for (x, y) in nodes:
        cv2.circle(vis_img, (x, y), 5, RED, 2)
    
    # Create adjacency matrix
    n = len(nodes)
    adj_matrix = np.zeros((n, n), dtype=int)
    
    # Check for connections between nodes and draw lines
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            # Check if there's a line between these two points
            points = np.linspace((x1, y1), (x2, y2), num=20).astype(int)
            if np.all(img[points[:, 1], points[:, 0]] > 45):  # Adjust threshold as needed
                adj_matrix[i, j] = adj_matrix[j, i] = 1
                # Draw a line between nodes
                cv2.line(vis_img, (x1, y1), (x2, y2), GREEN, 2)
    
    cv2.imwrite('detected_nodes_with_edges.png', vis_img)
    
    return adj_matrix

# Usage
image_path = 'image_0.png'
try:
    adjacency_matrix = detect_nodes_and_adjacency(image_path)
    np.save('predict_matrix.npy', adjacency_matrix)
    # print(adjacency_matrix)
except Exception as e:
    print(f"An error occurred: {e}")
