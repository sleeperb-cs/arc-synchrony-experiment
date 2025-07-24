# test_symbol_hunting.py
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

# Begin the symbol hunt!
hunter = SymbolHunter(reasoner)
hunter.hunt(grid, duration=500)  # Actually hunt for symbols
hunter.create_phase_portraits()  # Then create the visualization