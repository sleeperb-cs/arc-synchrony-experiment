# examine_symbols.py
import numpy as np
import matplotlib.pyplot as plt
from src.synchronized_reasoner import SynchronizedReasoner
from src.SymbolHunter import SymbolHunter

# Run the hunt
reasoner = SynchronizedReasoner()
hunter = SymbolHunter(reasoner)

grid = np.zeros((30, 30))
grid[10:20, 10:20] = 1
grid[15:25, 15:25] = 2

print("Hunting for symbols...")
hunter.hunt(grid, duration=4000)

# Examine individual captured symbols
if hunter.captured_symbols:
    print(f"\nCaught {len(hunter.captured_symbols)} symbols!")
    
    # Plot first few symbols as individual portraits
    fig, axes = plt.subplots(6, 7, figsize=(24, 18))
    axes = axes.flatten()
    
    for i, symbol in enumerate(hunter.captured_symbols[:42]):
        state = symbol['reservoir_state']
        
        # Plot first 50 neurons as a pattern
        axes[i].plot(state[:10], 'b-', linewidth=2)
        axes[i].set_title(f"Symbol {i} (t={symbol['time']}, c={symbol['coherence']:.3f})")
        axes[i].set_ylim(-1, 1)
        
    plt.tight_layout()
    plt.show()
    
    # Compare topological signatures
    print("\nTopological signatures:")
    for i, symbol in enumerate(hunter.captured_symbols[:3]):
        print(f"Symbol {i}: {symbol['topology']}")