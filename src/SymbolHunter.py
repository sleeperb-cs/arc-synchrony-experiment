# symbol_hunter.py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.geometric_metrics import extract_topological_signature

class SymbolHunter:
    def __init__(self, reasoner):
        self.reasoner = reasoner
        self.captured_symbols = []
        self.all_states = []
        self.all_coherences = []
        self.pca = PCA(n_components=3)
    
    def hunt(self, input_pattern, duration=1000):
        for t in range(duration):
            state, coherence = self.reasoner.resonate(input_pattern)
            
            # Ensure state is 1D array
            if hasattr(state, 'flatten'):
                state = state.flatten()
            elif len(state.shape) > 1:
                state = state.reshape(-1)
                
            self.all_states.append(state)
            self.all_coherences.append(coherence)
            
            if coherence > 0.8:  # Symbol emergence!
                symbol = {
                    'time': t,
                    'coherence': coherence,
                    'reservoir_state': state.copy(),
                    'topology': extract_topological_signature(state)
                }
                self.captured_symbols.append(symbol)
                print(f"Symbol captured at t={t}, coherence={coherence:.3f}")
    
    def create_phase_portraits(self):
        """Create PCA-based phase portraits"""
        states = np.array(self.all_states)
        coherences = np.array(self.all_coherences)
        
        # Fit PCA on all states
        states_pca = self.pca.fit_transform(states)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))
        
        # 3D Phase portrait colored by coherence
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(states_pca[:, 0], states_pca[:, 1], states_pca[:, 2], 
                            c=coherences, cmap='viridis', alpha=0.6)
        ax1.set_title('Phase Space Journey')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        plt.colorbar(scatter, ax=ax1, label='Coherence')
        
        # 2D projection with trajectory
        ax2 = fig.add_subplot(132)
        ax2.plot(states_pca[:, 0], states_pca[:, 1], 'k-', alpha=0.3, linewidth=0.5)
        ax2.scatter(states_pca[:, 0], states_pca[:, 1], c=coherences, 
                   cmap='viridis', s=20, alpha=0.8)
        ax2.set_title('Symbol Exploration Path')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        
        # Symbol locations only
        ax3 = fig.add_subplot(133)
        if self.captured_symbols:
            symbol_indices = [s['time'] for s in self.captured_symbols]
            symbol_states = states_pca[symbol_indices]
            ax3.scatter(symbol_states[:, 0], symbol_states[:, 1], 
                       s=100, c='red', marker='*', edgecolors='black')
            ax3.set_title('Discovered Symbols')
            ax3.set_xlabel('PC1')
            ax3.set_ylabel('PC2')
            
            # Add symbol numbers
            for i, (x, y) in enumerate(symbol_states[:, :2]):
                ax3.annotate(f'S{i}', (x, y), xytext=(5, 5), 
                           textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
        
        # Print PCA explained variance
        print(f"PCA explained variance: {self.pca.explained_variance_ratio_}")
        print(f"Total symbols found: {len(self.captured_symbols)}")