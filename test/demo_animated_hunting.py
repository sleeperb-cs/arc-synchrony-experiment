# demo_animated_hunting.py
import numpy as np
from src.synchronized_reasoner import SynchronizedReasoner
from src.SymbolHunter import SymbolHunter
import matplotlib.pyplot as plt

def create_test_patterns():
    """Create various test patterns for demonstration"""
    patterns = {}
    
    # Pattern 1: Simple square
    pattern1 = np.zeros((30, 30))
    pattern1[10:20, 10:20] = 1
    patterns['square'] = pattern1
    
    # Pattern 2: Cross pattern
    pattern2 = np.zeros((30, 30))
    pattern2[15, :] = 1  # Horizontal line
    pattern2[:, 15] = 1  # Vertical line  
    patterns['cross'] = pattern2
    
    # Pattern 3: Diagonal stripes
    pattern3 = np.zeros((30, 30))
    for i in range(30):
        for j in range(30):
            if (i + j) % 4 == 0:
                pattern3[i, j] = 1
    patterns['stripes'] = pattern3
    
    # Pattern 4: Random noise with structure
    pattern4 = np.random.random((30, 30)) * 0.3
    pattern4[5:25, 5:25] += 0.7  # Add structured region
    patterns['structured_noise'] = pattern4
    
    return patterns

def demo_animated_symbol_hunting():
    """Demonstrate the animated symbol hunting functionality"""
    print("🎯 Animated Symbol Hunter Demo")
    print("=" * 50)
    
    # Create reasoner and hunter
    reasoner = SynchronizedReasoner()
    hunter = SymbolHunter(reasoner)
    
    # Get test patterns
    patterns = create_test_patterns()
    
    print(f"Available patterns: {list(patterns.keys())}")
    pattern_name = input("Choose a pattern (default: square): ").strip() or 'square'
    
    if pattern_name not in patterns:
        print(f"Pattern '{pattern_name}' not found, using 'square'")
        pattern_name = 'square'
    
    pattern = patterns[pattern_name]
    
    # Get parameters from user
    try:
        duration = int(input("Animation duration (default: 100): ") or "100")
        symbol_lifetime = int(input("Symbol lifetime (default: 30): ") or "30")
    except ValueError:
        duration, symbol_lifetime = 100, 30
        print("Using default values: duration=100, symbol_lifetime=30")
    
    print(f"\n🚀 Starting animated hunt with pattern '{pattern_name}'...")
    print(f"Duration: {duration} frames")
    print(f"Symbol lifetime: {symbol_lifetime} frames")
    print(f"Animation interval: {hunter.animation_interval}ms")
    print("\nClose the plot window when you're done watching the animation.")
    
    # Clear any previous data
    hunter.captured_symbols.clear()
    hunter.all_states.clear()
    hunter.all_coherences.clear()
    hunter.live_symbols.clear()
    hunter.current_time = 0
    
    # Run animated hunt
    try:
        animation = hunter.hunt_animated(pattern, duration=duration, symbol_lifetime=symbol_lifetime)
        
        # Print results after animation
        print(f"\n📊 Animation completed!")
        print(f"Total symbols captured: {len(hunter.captured_symbols)}")
        print(f"Total states processed: {len(hunter.all_states)}")
        
        if hunter.captured_symbols:
            print("\n🎯 Captured symbols:")
            for i, symbol in enumerate(hunter.captured_symbols):
                print(f"  Symbol {i+1}: t={symbol['time']}, coherence={symbol['coherence']:.3f}")
        
        return animation
        
    except KeyboardInterrupt:
        print("\n⏹️  Animation interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error during animation: {e}")
        return None

def demo_static_vs_animated_comparison():
    """Compare static and animated symbol hunting"""
    print("\n🔄 Static vs Animated Comparison")
    print("=" * 40)
    
    reasoner1 = SynchronizedReasoner()
    reasoner2 = SynchronizedReasoner()
    hunter_static = SymbolHunter(reasoner1)
    hunter_animated = SymbolHunter(reasoner2)
    
    # Use the same pattern for both
    pattern = np.zeros((30, 30))
    pattern[10:20, 10:20] = 1
    pattern[15:25, 15:25] = 2
    
    duration = 50
    
    print("Running static hunt...")
    hunter_static.hunt(pattern, duration=duration)
    
    print("Running animated hunt (no display)...")
    # Mock the animation to avoid display
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    try:
        hunter_animated.hunt_animated(pattern, duration=duration)
    except:
        # If animation fails, manually run the steps
        for t in range(duration):
            state, coherence = hunter_animated.reasoner.resonate(pattern)
            if hasattr(state, 'flatten'):
                state = state.flatten()
            elif len(state.shape) > 1:
                state = state.reshape(-1)
            
            hunter_animated.all_states.append(state)
            hunter_animated.all_coherences.append(coherence)
            
            if coherence > 0.8:
                from src.geometric_metrics import extract_topological_signature
                symbol = {
                    'time': t,
                    'coherence': coherence,
                    'reservoir_state': state.copy(),
                    'topology': extract_topological_signature(state)
                }
                hunter_animated.captured_symbols.append(symbol)
    
    # Compare results
    print(f"\n📈 Results Comparison:")
    print(f"Static hunt - Symbols: {len(hunter_static.captured_symbols)}, States: {len(hunter_static.all_states)}")
    print(f"Animated hunt - Symbols: {len(hunter_animated.captured_symbols)}, States: {len(hunter_animated.all_states)}")
    
    # Restore interactive backend
    matplotlib.use('TkAgg')

if __name__ == "__main__":
    print("🎮 Animated Symbol Hunter Demonstration")
    print("======================================")
    
    while True:
        print("\nOptions:")
        print("1. Run animated symbol hunting demo")
        print("2. Compare static vs animated hunting")
        print("3. Exit")
        
        choice = input("Choose an option (1-3): ").strip()
        
        if choice == '1':
            demo_animated_symbol_hunting()
        elif choice == '2':
            demo_static_vs_animated_comparison()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")
    
    print("\nDemo completed! 🎉")