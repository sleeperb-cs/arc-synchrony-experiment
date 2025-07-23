import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

def extract_topological_signature(states):
    """Extract persistent topological features from reservoir states"""
    # Convert to trajectory in phase space
    if len(states.shape) == 1:
        states = states.reshape(-1, 1)
    
    # Compute distance matrix (this captures the "shape" of the dynamics)
    distances = squareform(pdist(states))
    
    # Extract signature through persistence
    signature = {
        'mean_radius': np.mean(distances),
        'variance': np.var(distances),
        'entropy': entropy(distances.flatten()),
        'dimension_estimate': estimate_intrinsic_dimension(distances)
    }
    
    return signature

def estimate_intrinsic_dimension(distances):
    """Estimate the intrinsic dimension of the attractor"""
    # Sort distances for each point
    sorted_dists = np.sort(distances, axis=1)
    
    # Use ratio of consecutive distances (MLE approach)
    k = min(10, len(distances) - 1)
    ratios = sorted_dists[:, k] / sorted_dists[:, 1:k].mean(axis=1)
    
    # Estimate dimension
    dimension = 1.0 / np.log(ratios).mean()
    return dimension

# geometric_metrics.py - The measurement instruments
def measure_information_curvature(trajectory):
    """Where symbols reveal themselves through geometry"""
    # Local tangent space variation
    gradients = np.gradient(trajectory, axis=0)
    curvature = np.sum(np.abs(np.gradient(gradients, axis=0)))
    return curvature / len(trajectory)

def detect_symbol_emergence(reservoir_states, threshold=2.0):
    """The moment of crystallization"""
    curvature = measure_information_curvature(reservoir_states)
    if curvature > threshold:
        return extract_topological_signature(reservoir_states)
    return None