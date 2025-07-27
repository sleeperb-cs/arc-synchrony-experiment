# simple_animated_demo.py
import numpy as np
from src.synchronized_reasoner import SynchronizedReasoner
from src.SymbolHunter import SymbolHunter

def main():
    print("🎯 Simple Animated Symbol Hunter Demo")
    print("====================================")
    
    # Create a test pattern
    pattern = np.zeros((30, 30))
    pattern[10:20, 10:20] = 1  # Square
    pattern[15:25, 15:25] = 2  # Overlapping square
    
    print("Creating reasoner and hunter...")
    reasoner = SynchronizedReasoner()
    hunter = SymbolHunter(reasoner)
    
    print(f"Animation parameters:")
    print(f"  - Symbol lifetime: {hunter.symbol_lifetime} frames")
    print(f"  - Animation interval: {hunter.animation_interval}ms")
    print(f"  - Max trail length: {hunter.max_trail_length}")
    
    print("\nStarting animated symbol hunt...")
    print("This will show 3 subplots:")
    print("  1. Phase Space Journey (3D, recent points)")
    print("  2. Symbol Exploration Path (2D trail)")
    print("  3. Active Symbols (symbols that appear and disappear)")
    print("\nClose the plot window when done watching.")
    
    try:
        # Run the animated hunt
        animation = hunter.hunt_animated(
            pattern, 
            duration=100,      # 100 time steps
            symbol_lifetime=30  # Symbols visible for 30 steps
        )
        
        print(f"\n📊 Final Results:")
        print(f"Total symbols captured: {len(hunter.captured_symbols)}")
        print(f"Total states processed: {len(hunter.all_states)}")
        
        if hunter.captured_symbols:
            print("\n🎯 Captured symbols:")
            for i, symbol in enumerate(hunter.captured_symbols):
                print(f"  Symbol {i+1}: time={symbol['time']}, coherence={symbol['coherence']:.3f}")
        
        print("\n✅ Animation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during animation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()