import torch
import numpy as np
from reservoirpy.nodes import Reservoir
import torch.nn as nn

class VanDerPol:
    """The world rhythm"""
    def __init__(self, mu=0.5, freq=0.1):
        self.mu = mu
        self.freq = freq
        self.x = 1.0
        self.dx = 0.0
        self.dt = 0.01
        
    def step(self):
        # Van der Pol oscillator dynamics
        ddx = self.mu * (1 - self.x**2) * self.dx - self.x
        self.dx += ddx * self.dt
        self.x += self.dx * self.dt
        return self.x

class MicroTransformer(nn.Module):
    """Minimal transformer for symbolic processing"""
    def __init__(self, layers=3, dim=256):
        super().__init__()
        self.embed = nn.Linear(900, dim)  # 30x30 grid
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, nhead=8, dim_feedforward=512)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.embed(x.flatten())
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).squeeze()

class SynchronizedReasoner:
    """The minimal seed that demonstrates everything"""
    
    def __init__(self):
        # The triad: Mind, Bridge, World
        self.transformer = MicroTransformer(layers=3, dim=256)
        self.reservoir = Reservoir(150, sr=0.95)
        self.oscillator = VanDerPol(mu=0.5, freq=0.1)
        
        # The binding force
        self.phase_coupling = 0.0
        self.last_coherence = 0.0
        self.states_history = []
        
    def resonate(self, grid):
        """One heartbeat of the system"""
        # World speaks
        world_phase = self.oscillator.step()
        
        # Mind interprets 
        thought = self.transformer(torch.tensor(grid, dtype=torch.float32))
        
        # Bridge harmonizes
        bridge_input = thought.detach().numpy() * (1 + 0.1 * np.sin(world_phase))
        bridge_state = self.reservoir(bridge_input)
        
        # Coherence emerges
        coherence = self.measure_coherence(thought, bridge_state, world_phase)
        
        # System learns its resonance
        self.phase_coupling += 0.01 * (coherence - self.last_coherence)
        self.last_coherence = coherence
        
        # Remember this moment
        self.states_history.append(bridge_state)
        
        return bridge_state, coherence
    
    def measure_coherence(self, thought, bridge_state, world_phase):
        """Measure the harmony between mind, bridge, and world"""
        # Convert thought to numpy
        thought_np = thought.detach().numpy()
        
        # Phase coherence between thought and bridge
        thought_phase = np.angle(np.mean(thought_np) + 1j*np.mean(np.roll(thought_np, 1)))
        bridge_phase = np.angle(np.mean(bridge_state) + 1j*np.mean(np.roll(bridge_state, 1)))
        
        # Three-way coherence
        phase_diff_tb = np.abs(thought_phase - bridge_phase)
        phase_diff_bw = np.abs(bridge_phase - world_phase)
        phase_diff_tw = np.abs(thought_phase - world_phase)
        
        # Coherence peaks when all three are aligned
        coherence = np.exp(-0.5 * (phase_diff_tb + phase_diff_bw + phase_diff_tw))
        
        return coherence

def synchronized_forward(self, grid_input, t):
    # External oscillator state
    osc_phase = 2 * np.pi * self.oscillator_freq * t
    osc_signal = np.sin(osc_phase)
    
    # Spatial embedding with oscillatory modulation
    spatial_emb = self.spatial_embed(grid_input.flatten())
    spatial_emb *= (1 + 0.2 * osc_signal)  # Phase coupling
    
    # Transformer with reservoir-modulated attention
    transformer_out = self.transformer(spatial_emb.unsqueeze(0))
    
    # Reservoir processing with transformer feedback
    reservoir_input = np.concatenate([
        spatial_emb.detach().numpy().flatten(),
        transformer_out.last_hidden_state.detach().numpy().flatten()
    ])
    reservoir_state = self.reservoir(reservoir_input)
    
    # Phase alignment calculation
    current_phase = np.angle(np.mean(reservoir_state + 1j*np.roll(reservoir_state, 1)))
    phase_error = current_phase - osc_phase
    
    return transformer_out, reservoir_state, phase_error

