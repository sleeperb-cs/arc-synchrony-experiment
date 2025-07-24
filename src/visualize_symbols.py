# Fixed visualization
import numpy as np
import matplotlib.pyplot as plt
from synchronized_reasoner import SynchronizedReasoner

reasoner = SynchronizedReasoner()

grid = np.zeros((30, 30))
grid[10:20, 10:20] = 1
grid[15:25, 15:25] = 2

high_coherence_states = []
low_coherence_states = []

for t in range(500):
    state, coherence = reasoner.resonate(grid)
    
    if coherence > 0.8:
        high_coherence_states.append(state)
    elif coherence < 0.1:
        low_coherence_states.append(state)

# Plot phase portraits using different neurons as axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

if high_coherence_states:
    high_states = np.array(high_coherence_states)
    # Plot neuron 0 vs neuron 1
    ax1.scatter(high_states[:, 0], high_states[:, 1], alpha=0.6)
    ax1.set_title('High Coherence States')
    ax1.set_xlabel('Neuron 0')
    ax1.set_ylabel('Neuron 1')

if low_coherence_states:
    low_states = np.array(low_coherence_states)
    ax2.scatter(low_states[:, 0], low_states[:, 1], alpha=0.6)
    ax2.set_title('Low Coherence States')
    ax2.set_xlabel('Neuron 0') 
    ax2.set_ylabel('Neuron 1')

plt.tight_layout()
plt.show()