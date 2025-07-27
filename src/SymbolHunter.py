# symbol_hunter.py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time
from src.geometric_metrics import extract_topological_signature

class SymbolHunter:
    def __init__(self, reasoner):
        self.reasoner = reasoner
        self.captured_symbols = []
        self.all_states = []
        self.all_coherences = []
        self.pca = PCA(n_components=3)
        
        # Animation-specific attributes
        self.live_symbols = deque()  # Active symbols with expiration times
        self.symbol_lifetime = 50    # How long symbols stay visible
        self.current_time = 0
        self.animation_interval = 1  # milliseconds between frames
        self.max_trail_length = 50   # Length of exploration path trail
    
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
    
    def _update_live_symbols(self, current_time):
        """Update the list of currently visible symbols based on their lifetime"""
        # Remove expired symbols
        while self.live_symbols and self.live_symbols[0]['expire_time'] <= current_time:
            self.live_symbols.popleft()
        
        # Add newly captured symbols that should be visible
        for symbol in self.captured_symbols:
            symbol_time = symbol['time']
            expire_time = symbol_time + self.symbol_lifetime
            
            # Check if this symbol should be active now
            if (symbol_time <= current_time <= expire_time and 
                not any(s['time'] == symbol_time for s in self.live_symbols)):
                live_symbol = symbol.copy()
                live_symbol['expire_time'] = expire_time
                self.live_symbols.append(live_symbol)
    
    def hunt_animated(self, input_pattern, duration=1000, symbol_lifetime=50):
        """Hunt for symbols with real-time animated visualization"""
        self.symbol_lifetime = symbol_lifetime
        self.current_time = 0
        
        # Setup the figure and subplots
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)  
        ax3 = fig.add_subplot(133)
        
        # Initialize empty plots
        self.trail_points = deque(maxlen=self.max_trail_length)
        self.phase_scatter = ax1.scatter([], [], [], c=[], cmap='viridis')
        self.trail_line, = ax2.plot([], [], 'k-', alpha=0.3, linewidth=1)
        self.trail_scatter = ax2.scatter([], [], c=[], cmap='viridis', s=20, alpha=0.6)
        self.symbol_scatter = ax3.scatter([], [], s=200, c='red', marker='*', alpha=0.8, edgecolors='black')
        
        # Set up axes
        ax1.set_title('Phase Space Journey')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2') 
        ax1.set_zlabel('PC3')
        
        ax2.set_title('Symbol Exploration Path')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        
        ax3.set_title('Active Symbols')
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        
        def animate_frame(frame):
            if frame >= duration:
                return []
                
            # Process one step
            state, coherence = self.reasoner.resonate(input_pattern)
            
            # Ensure state is 1D array
            if hasattr(state, 'flatten'):
                state = state.flatten()
            elif len(state.shape) > 1:
                state = state.reshape(-1)
                
            self.all_states.append(state)
            self.all_coherences.append(coherence)
            self.current_time = frame
            
            # Check for symbol emergence
            if coherence > 0.8:  # Symbol emergence!
                symbol = {
                    'time': frame,
                    'coherence': coherence,
                    'reservoir_state': state.copy(),
                    'topology': extract_topological_signature(state)
                }
                self.captured_symbols.append(symbol)
                print(f"Symbol captured at t={frame}, coherence={coherence:.3f}")
            
            # Update live symbols
            self._update_live_symbols(frame)
            
            # Only proceed with visualization if we have enough data
            if len(self.all_states) < 2:
                return []
            
            # Fit PCA on available states
            states = np.array(self.all_states)
            coherences = np.array(self.all_coherences)
            
            if states.shape[0] >= 3:  # Need at least 3 points for PCA
                states_pca = self.pca.fit_transform(states)
                
                # Update trail
                if len(states_pca) > 0:
                    current_point = {
                        'pca': states_pca[-1],
                        'coherence': coherences[-1],
                        'time': frame
                    }
                    self.trail_points.append(current_point)
                
                # Update 3D phase space (show recent points)
                recent_window = min(50, len(states_pca))  
                recent_states = states_pca[-recent_window:]
                recent_coherences = coherences[-recent_window:]
                
                ax1.clear()
                ax1.scatter(recent_states[:, 0], recent_states[:, 1], recent_states[:, 2],
                           c=recent_coherences, cmap='viridis', alpha=0.6, s=30)
                ax1.set_title(f'Phase Space Journey (t={frame})')
                ax1.set_xlabel('PC1')
                ax1.set_ylabel('PC2')
                ax1.set_zlabel('PC3')
                
                # Update 2D exploration path  
                ax2.clear()
                if len(self.trail_points) > 1:
                    trail_pca = np.array([p['pca'][:2] for p in self.trail_points])
                    trail_coherences = np.array([p['coherence'] for p in self.trail_points])
                    
                    ax2.plot(trail_pca[:, 0], trail_pca[:, 1], 'k-', alpha=0.3, linewidth=1)
                    ax2.scatter(trail_pca[:, 0], trail_pca[:, 1], 
                               c=trail_coherences, cmap='viridis', s=20, alpha=0.6)
                
                ax2.set_title(f'Symbol Exploration Path (t={frame})')
                ax2.set_xlabel('PC1')
                ax2.set_ylabel('PC2')
                
                # Update active symbols
                ax3.clear()  
                if self.live_symbols:
                    symbol_indices = [s['time'] for s in self.live_symbols]
                    if all(i < len(states_pca) for i in symbol_indices):
                        symbol_states = states_pca[symbol_indices]
                        symbol_coherences = [s['coherence'] for s in self.live_symbols]
                        
                        scatter = ax3.scatter(symbol_states[:, 0], symbol_states[:, 1], 
                                            s=200, c=symbol_coherences, cmap='plasma', 
                                            marker='*', alpha=0.8, edgecolors='black',
                                            vmin=0.8, vmax=1.0)
                        
                        # Add time-to-expiry information
                        for i, (symbol, pos) in enumerate(zip(self.live_symbols, symbol_states)):
                            time_left = symbol['expire_time'] - frame
                            ax3.annotate(f'{time_left}', pos[:2], xytext=(5, 5), 
                                       textcoords='offset points', fontsize=8, alpha=0.7)
                
                ax3.set_title(f'Active Symbols (t={frame}, count={len(self.live_symbols)})')
                ax3.set_xlabel('PC1')
                ax3.set_ylabel('PC2')
            
            return []
        
        # Create and run animation
        anim = animation.FuncAnimation(fig, animate_frame, frames=duration, 
                                     interval=self.animation_interval, blit=False, repeat=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
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
        #scatter = ax1.scatter(states_pca[:, 0], states_pca[:, 1], states_pca[:, 2], s=10000,
                            #c=coherences, cmap='viridis', alpha=0.03, edgecolors="black")
        scatter = ax1.scatter(states_pca[:, 0], states_pca[:, 1], states_pca[:, 2],
                    c=coherences, cmap='viridis', alpha=0.1)
        ax1.set_title('Phase Space Journey')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        plt.colorbar(scatter, ax=ax1, label='Coherence')
        
        # 2D projection with trajectory
        ax2 = fig.add_subplot(132)
        ax2.plot(states_pca[:, 0], states_pca[:, 1], 'k-', alpha=0.1, linewidth=0.5)
        # colors
        ax2.scatter(states_pca[:, 0], states_pca[:, 1], c=coherences, 
                   cmap='viridis', s=10, alpha=.2)
        # no markers, lines only
        # ax2.scatter(states_pca[:, 0], states_pca[:, 1], 
                # facecolors='none', edgecolors='none')
        ax2.set_title('Symbol Exploration Path')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        
        # Symbol locations only
        ax3 = fig.add_subplot(133)
        if self.captured_symbols:
            symbol_indices = [s['time'] for s in self.captured_symbols]
            symbol_states = states_pca[symbol_indices]
            ax3.scatter(symbol_states[:, 0], symbol_states[:, 1], 
                       s=200, c='red', marker='*',alpha=.2, edgecolors='black')
            ax3.set_title('Discovered Symbols')
            ax3.set_xlabel('PC1')
            ax3.set_ylabel('PC2')
            
            # Add symbol numbers
            # for i, (x, y) in enumerate(symbol_states[:, :2]):
                # ax3.annotate(f'S{i}', (x, y), xytext=(5, 5), 
                           # textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
        
        # Print PCA explained variance
        print(f"PCA explained variance: {self.pca.explained_variance_ratio_}")
        print(f"Total symbols found: {len(self.captured_symbols)}")