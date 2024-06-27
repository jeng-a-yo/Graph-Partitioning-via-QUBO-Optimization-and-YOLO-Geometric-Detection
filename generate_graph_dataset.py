import numpy as np
import time
import os
import sys
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

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) ** 0.5

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
        if all(calculate_distance(new_point, point) >= distance_threshold for point in points):
            points.append(tuple(map(round, new_point)))

    return points

def generate_adjacency_matrix(points, connect_quantity, unit_vector_length, draw):
    """Generate adjacency matrix and draw connections."""
    points_quantity = len(points)
    adjacency_matrix = np.zeros((points_quantity, points_quantity), dtype=int)

    for i, point1 in enumerate(points):
        distances = {j: calculate_distance(point1, point2) for j, point2 in enumerate(points) if i != j}
        nearest_points = sorted(distances, key=distances.get)[:connect_quantity]

        for j in nearest_points:
            point2 = points[j]
            adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1

            vector = (point2[0] - point1[0], point2[1] - point1[1])
            vector_length = calculate_distance(point1, point2)
            unit_vector = tuple(coord * unit_vector_length / vector_length for coord in vector)

            np1 = (point1[0] + unit_vector[0], point1[1] + unit_vector[1])
            np2 = (point2[0] - unit_vector[0], point2[1] - unit_vector[1])
            draw.line([np1, np2], fill=BLACK, width=4)

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
         dots_distance_threshold=50, connect_quantity=2, unit_vector_length=8, 
         images_path="images", matrices_path="adjacency_matrices"):
    """Main function to generate points, adjacency matrix, and save the image."""
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(matrices_path, exist_ok=True)

    for i in range(num_images):
        while True:
            image = Image.new("RGB", (image_width, image_length), WHITE)
            draw = ImageDraw.Draw(image)

            points = generate_points(points_quantity, image_width, image_length, edge, point_radius, dots_distance_threshold)

            for point in points:
                draw_circle(draw, point, point_radius, BLUE, BLACK)

            adjacency_matrix = generate_adjacency_matrix(points, connect_quantity, unit_vector_length, draw)

            if is_connected(adjacency_matrix):
                break
            print(f"[Info] Generated graph is not connected. Retrying...")

        image.save(os.path.join(images_path, f"image_{i}.png"))
        np.save(os.path.join(matrices_path, f"adjacency_matrix_{i}.npy"), adjacency_matrix)

        print(f"[Info] Image and adjacency matrix {i} saved in {images_path} and {matrices_path}")

if __name__ == "__main__":
    main(num_images=10)
