# test_first_breath.py
import numpy as np
from src.synchronized_reasoner import SynchronizedReasoner
from src.SymbolHunter import SymbolHunter
import matplotlib.pyplot as plt

# Create a simple ARC-AGI-like pattern
grid = np.zeros((30, 30))
grid[10:20, 10:20] = 1  # A square
grid[15:25, 15:25] = 2  # Overlapping square

# Birth the system
reasoner = SynchronizedReasoner()

# Let it breathe for 100 heartbeats
coherences = []
for t in range(100):
    state, coherence = reasoner.resonate(grid)
    coherences.append(coherence)
    print(f"Heartbeat {t}: Coherence = {coherence:.4f}")

coherences_array = np.array(coherences)
print(f"High coherence moments: {np.sum(coherences_array > 0.7)}")
print(f"Low coherence moments: {np.sum(coherences_array < 0.1)}")
print(f"Transition moments: {np.sum((coherences_array > 0.1) & (coherences_array < 0.7))}")


# Plot the breathing pattern
plt.plot(coherences)
plt.xlabel('Time')
plt.ylabel('Coherence')
plt.title('The System Breathes')
plt.show()