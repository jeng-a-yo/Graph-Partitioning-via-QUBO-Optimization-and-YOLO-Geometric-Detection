import networkx as nx
import numpy as np

def is_isomorphic(pred_matrix, correct_matrix):
    """
    Check if two adjacency matrices correspond to isomorphic graphs.

    Parameters:
    pred_matrix (np.ndarray): Predicted adjacency matrix.
    correct_matrix (np.ndarray): Correct adjacency matrix.

    Returns:
    bool: True if the graphs are isomorphic, False otherwise.
    """
    # Create graphs from the adjacency matrices
    predicted_graph = nx.from_numpy_array(pred_matrix)
    correct_graph = nx.from_numpy_array(correct_matrix)
    
    # Check for isomorphism
    return nx.is_isomorphic(predicted_graph, correct_graph)


def load_adjacency_matrix(file_path):
    """
    Load an adjacency matrix from a .npy file.

    Parameters:
    file_path (str): Path to the .npy file.

    Returns:
    np.ndarray: Loaded adjacency matrix.
    """
    try:
        matrix = np.load(file_path)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix loaded from {file_path} is not square.")
        return matrix
    except Exception as e:
        raise ValueError(f"Error loading matrix from {file_path}: {e}")

# Example usage
if __name__ == "__main__":
    try:
        predicted_matrix = load_adjacency_matrix('predict_matrix.npy')
        correct_matrix = load_adjacency_matrix('adjacency_matrix_0.npy')
        
        is_correct = is_isomorphic(predicted_matrix, correct_matrix)
        print("The prediction is correct:", is_correct)
    except Exception as e:
        print(e)


# print(pred_matrix)
# print('============================================================')
# print(correct_matrix)