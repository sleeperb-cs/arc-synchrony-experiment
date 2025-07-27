# temporal_symbol_variance.py
import numpy as np
import time
from datetime import datetime
from src.SymbolHunter import SymbolHunter

class TemporalSymbolAnalyzer:
    def __init__(self, reasoner):
        self.reasoner = reasoner
        
    def analyze_temporal_variance(self, grid, num_runs=10, duration=500):
        """Run symbol hunts at different times and compare"""
        results = []
        
        for run in range(num_runs):
            print(f"\nRun {run} at {datetime.now()}")
            hunter = SymbolHunter(self.reasoner)
            hunter.hunt(grid, duration)
            
            results.append({
                'timestamp': time.time(),
                'num_symbols': len(hunter.captured_symbols),
                'coherences': [s['coherence'] for s in hunter.captured_symbols],
                'symbol_times': [s['time'] for s in hunter.captured_symbols]
            })
            
            # Wait a bit between runs
            time.sleep(30)  # 30 seconds
        
        # Analyze variance
        symbol_counts = [r['num_symbols'] for r in results]
        print(f"\nSymbol count variance: {np.var(symbol_counts)}")
        print(f"Symbol counts: {symbol_counts}")
        
        return results