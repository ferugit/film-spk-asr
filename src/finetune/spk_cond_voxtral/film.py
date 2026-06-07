"""
FiLM (Feature-wise Linear Modulation) modules for conditioning
the Voxtral encoder with speaker embeddings (x-vectors).

Architecture
============
Given x-vector z ∈ R^{d_spk} (from SiAMResNet34, d_spk=256):

    FiLM(h) = γ ⊙ h + β

where  γ, β = split(MLP(z))  ∈ R^{d_model}  (d_model=1280 for Voxtral)

For normative speech (HC / Unknown), a learned gate can smoothly
interpolate between identity (γ=1, β=0) and full modulation.

References
----------
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
"""

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) from a speaker embedding.

    Parameters
    ----------
    spk_dim : int
        Dimension of the input speaker embedding (256 for SiAMResNet34).
    hidden_dim : int
        Hidden dimension of the conditioning MLP.
    out_dim : int
        Dimension of each modulation vector (= encoder d_model, 1280 for Voxtral).
    use_gate : bool
        If True, also produces a scalar gate α ∈ [0,1] that interpolates
        between identity (no modulation) and full FiLM modulation.
        This allows the model to *learn* when conditioning is useful.
    """
    def __init__(self, spk_dim: int = 256, hidden_dim: int = 512,
                 out_dim: int = 1280, use_gate: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.use_gate = use_gate

        # MLP:  z → hidden → (γ, β)
        self.mlp = nn.Sequential(
            nn.Linear(spk_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * out_dim),
        )

        # Optional gating network:  z → α  ∈ [0, 1]
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(spk_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize so that FiLM starts close to identity (γ≈1, β≈0).

        Only the *last* linear layer has near-zero weights.  The first
        layer keeps default (Kaiming) init so that gradients can still
        flow back through the MLP to the x-vector extractor.
        """
        last_linear = self.mlp[-1]
        # Small but non-zero weights so gradients flow, but output is
        # dominated by the bias (identity).
        nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            # bias[:out_dim] → γ, initialized to 1 (identity scale)
            last_linear.bias[:self.out_dim].fill_(1.0)
            # bias[out_dim:] → β, initialized to 0 (no shift)
            last_linear.bias[self.out_dim:].fill_(0.0)

        if self.use_gate:
            # Initialize gate bias so that sigmoid starts near 0 (weak modulation)
            # sigmoid(-2) ≈ 0.12 → conservative start
            gate_last = self.gate[-2]  # Linear before Sigmoid
            nn.init.zeros_(gate_last.weight)
            nn.init.constant_(gate_last.bias, -2.0)

    def forward(self, z: torch.Tensor):
        """
        Parameters
        ----------
        z : torch.Tensor, shape (B, spk_dim)
            Speaker embedding (x-vector).

        Returns
        -------
        gamma : torch.Tensor, shape (B, out_dim)
        beta  : torch.Tensor, shape (B, out_dim)
        alpha : torch.Tensor, shape (B, 1) or None
            Gating scalar (only if use_gate=True).
        """
        params = self.mlp(z)  # (B, 2 * out_dim)
        gamma, beta = params.split(self.out_dim, dim=-1)  # each (B, out_dim)

        alpha = None
        if self.use_gate:
            alpha = self.gate(z)  # (B, 1)

        return gamma, beta, alpha


class FiLMLayer(nn.Module):
    """
    Applies FiLM modulation to hidden states.

    h_out = α * (γ ⊙ h + β)  +  (1 - α) * h

    When α=0 → identity (no modulation).
    When α=1 → full FiLM modulation.
    When use_gate=False in the generator, α is assumed 1 (always modulate).
    """
    def forward(self, hidden_states: torch.Tensor,
                gamma: torch.Tensor, beta: torch.Tensor,
                alpha: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (B, T, D)
        gamma : (B, D)
        beta  : (B, D)
        alpha : (B, 1) or None
        """
        # Expand to (B, 1, D) for broadcasting over time dimension
        g = gamma.unsqueeze(1)
        b = beta.unsqueeze(1)

        modulated = g * hidden_states + b

        if alpha is not None:
            a = alpha.unsqueeze(1)  # (B, 1, 1)
            return a * modulated + (1 - a) * hidden_states
        else:
            return modulated


class FiLMBank(nn.Module):
    """
    A bank of FiLM generators, one per encoder layer.

    Supports two modes:
        - "per_layer":  Independent FiLM generator for each layer.
        - "shared":     One FiLM generator shared across all layers (+ layer embedding).
    """
    def __init__(self, num_layers: int = 32, spk_dim: int = 256,
                 hidden_dim: int = 512, out_dim: int = 1280,
                 use_gate: bool = True, mode: str = "per_layer"):
        super().__init__()
        self.num_layers = num_layers
        self.mode = mode
        self.film_apply = FiLMLayer()

        if mode == "per_layer":
            self.generators = nn.ModuleList([
                FiLMGenerator(spk_dim, hidden_dim, out_dim, use_gate)
                for _ in range(num_layers)
            ])
        elif mode == "shared":
            # Shared generator with layer embedding concatenated to z
            self.layer_embed = nn.Embedding(num_layers, 32)
            self.generator = FiLMGenerator(
                spk_dim + 32, hidden_dim, out_dim, use_gate
            )
        else:
            raise ValueError(f"Unknown FiLM mode: {mode}. Use 'per_layer' or 'shared'.")

    def get_film_params(self, layer_idx: int, z: torch.Tensor):
        """Get FiLM parameters for a specific layer."""
        if self.mode == "per_layer":
            return self.generators[layer_idx](z)
        else:
            # Concat layer embedding to speaker embedding
            layer_emb = self.layer_embed.weight[layer_idx].unsqueeze(0).expand(z.size(0), -1)
            z_aug = torch.cat([z, layer_emb], dim=-1)
            return self.generator(z_aug)

    def apply(self, layer_idx: int, hidden_states: torch.Tensor,
              z: torch.Tensor) -> torch.Tensor:
        """
        Compute and apply FiLM modulation for a given layer.

        The FiLM generator weights may live on a different device than
        hidden_states (e.g., multi-GPU with device_map="auto").  We move
        z to the generator's device, compute gamma/beta/alpha there,
        then move results to the hidden_states device for the element-wise
        modulation.

        Parameters
        ----------
        layer_idx : int
        hidden_states : (B, T, D)
        z : (B, spk_dim)  — speaker embedding (already on hidden_states device)

        Returns
        -------
        modulated : (B, T, D)
        """
        # Figure out where the generator lives
        if self.mode == "per_layer":
            gen_device = next(self.generators[layer_idx].parameters()).device
        else:
            gen_device = next(self.generator.parameters()).device

        # Move z to the generator's device, compute FiLM params
        gamma, beta, alpha = self.get_film_params(layer_idx, z.to(gen_device))

        # Move FiLM params to hidden_states device
        hs_device = hidden_states.device
        gamma = gamma.to(hs_device)
        beta = beta.to(hs_device)
        if alpha is not None:
            alpha = alpha.to(hs_device)

        return self.film_apply(hidden_states, gamma, beta, alpha)
