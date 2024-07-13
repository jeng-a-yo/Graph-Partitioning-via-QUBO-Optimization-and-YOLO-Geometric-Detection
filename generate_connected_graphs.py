import numpy as np
import time
import os
import sys
import math
from copy import deepcopy
from typing import List, Tuple
from PIL import Image, ImageDraw
from random import uniform, sample
from collections import deque

np.set_printoptions(threshold=sys.maxsize)

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

def count_time(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[Info] Data Set Generated")
        print(f"[Info] Spend Time: {round(end_time - start_time, 4)} seconds")
        return result
    return wrapper

def calculate_distance_of_nodes(node1: Tuple[float, float], node2: Tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two nodes."""
    return ((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2) ** 0.5

def remove_zeros(lst):
    """
    Remove all zero elements from a list.

    Parameters:
    lst (list): The input list.

    Returns:
    list: A new list with all zero elements removed.
    """
    return [element for element in lst if element != 0]



def distance_point_to_line(point: Tuple[float, float], line: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """Calculate the perpendicular distance from a point to a line segment."""
    # Unpack points
    (x0, y0), (x1, y1) = line
    x, y = point

    # Line vector
    dx, dy = x1 - x0, y1 - y0

    # Normalize line vector
    mag = (dx ** 2 + dy ** 2) ** 0.5
    if mag == 0:
        return float('inf')

    dx, dy = dx / mag, dy / mag

    # Project point onto line
    t = dx * (x - x0) + dy * (y - y0)
    nearest = (x0 + t * dx, y0 + t * dy)

    # Clamp nearest point to the segment
    if t < 0:
        nearest = (x0, y0)
    elif t > mag:
        nearest = (x1, y1)

    # Return distance from point to the nearest point on segment
    return calculate_distance_of_nodes(point, nearest)

def angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors in degrees.

    Parameters:
    v1 (array-like): First vector.
    v2 (array-like): Second vector.

    Returns:
    float: Angle between the vectors in degrees.
    """
    # Convert to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Compute the dot product
    dot_product = np.dot(v1, v2)
    
    # Compute the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Clip the value to the range [-1, 1] to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Compute the angle in radians
    angle_radians = np.arccos(cos_theta)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


def draw_circle(draw, position, radius, fill_color, outline_color):
    """Draw a circle on the image."""
    diameter = radius ** 0.5
    draw.ellipse(
        (position[0] - diameter, position[1] - diameter, position[0] + diameter, position[1] + diameter),
        fill=fill_color, outline=outline_color, width=3
    )

def generate_points(points_quantity, image_width, image_length, edge, point_radius, distance_threshold):
    """Generate a list of points ensuring minimum distance between them."""
    min_value = edge + 2 * edge
    length_max_value = image_length - edge - 2 * edge
    width_max_value = image_width - edge - 2 * edge

    points = []

    while len(points) < points_quantity:
        new_point = (uniform(min_value, length_max_value), uniform(min_value, width_max_value))
        if all(calculate_distance_of_nodes(new_point, point) >= distance_threshold for point in points):
            points.append(tuple(map(round, new_point)))

    return points

def generate_adjacency_matrix(nodes: List[Tuple[float, float]], connections_per_node: int, 
                              unit_vector_length: float, point_radius: float, 
                              draw: ImageDraw.ImageDraw) -> np.ndarray:
    """Generate adjacency matrix and draw connections."""
    num_nodes = len(nodes)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i, node1 in enumerate(nodes):
        # Calculate distances from node1 to all other nodes
        distances = {j: calculate_distance_of_nodes(node1, node2) for j, node2 in enumerate(nodes) if i != j}
        # Find the indices of the nearest nodes
        nearest_nodes = sorted(distances, key=distances.get)[:connections_per_node]

        for j in nearest_nodes:
            node2 = nodes[j]
            vector = (node2[0] - node1[0], node2[1] - node1[1])
            vector_length = calculate_distance_of_nodes(node1, node2)
            unit_vector = tuple(coord * unit_vector_length / vector_length for coord in vector)

            # Adjust node positions to create spacing for the lines
            adjusted_node1 = (node1[0] + unit_vector[0], node1[1] + unit_vector[1])
            adjusted_node2 = (node2[0] - unit_vector[0], node2[1] - unit_vector[1])

            # Check if the line intersects with any nodes
            if all(remove_zeros(distance_point_to_line(node, (node1, node2)) > point_radius + 60 for node in nodes)):
                # Update adjacency matrix
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
                # Draw the line
                draw.line([adjusted_node1, adjusted_node2], fill=BLACK, width=3)

    return adjacency_matrix


def is_connected(adjacency_matrix):
    """Check if the graph is connected using BFS."""
    points_quantity = len(adjacency_matrix)
    visited = [False] * points_quantity
    queue = deque([0])
    visited[0] = True
    while queue:
        node = queue.popleft()
        for neighbor, connected in enumerate(adjacency_matrix[node]):
            if connected and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return all(visited)

@count_time
def main(num_images=1, points_quantity=25, image_width=800, image_length=800, edge=20, point_radius=130, 
         dots_distance_threshold=50, connect_quantity=2, unit_vector_length=12, 
         base_path="dataset"):
    """Main function to generate points, adjacency matrix, and save the image."""
    images_path = os.path.join(base_path, "images")
    matrices_path = os.path.join(base_path, "adjacency_matrices")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(matrices_path, exist_ok=True)

    for i in range(num_images):
        while True:
            image = Image.new("RGB", (image_width, image_length), WHITE)
            draw = ImageDraw.Draw(image)

            nodes = generate_points(points_quantity, image_width, image_length, edge, point_radius, dots_distance_threshold)

            for node in nodes:
                draw_circle(draw, node, point_radius, BLUE, BLACK)

            adjacency_matrix = generate_adjacency_matrix(nodes, connect_quantity, unit_vector_length, point_radius, draw)

            if is_connected(adjacency_matrix):
                break
            print(f"[Info] Generated graph is not connected. Retrying...")

        image.save(os.path.join(images_path, f"image_{i}.png"))
        np.save(os.path.join(matrices_path, f"adjacency_matrix_{i}.npy"), adjacency_matrix)

        print(f"[Info] Image and adjacency matrix {i} saved in {images_path} and {matrices_path}")

if __name__ == "__main__":
    main(num_images=10)
