import math
import torch
from torch import nn
import torch.nn.functional as F
from inrdiff.networks.layers import ReLULayer, SIRENLayer, WIRELayer


class ReLUMLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 **kwargs):
        super().__init__()

        a = [ReLULayer(in_size, hidden_size, **kwargs)]
        for i in range(num_layers - 1):
            a.append(ReLULayer(hidden_size, hidden_size, **kwargs))
        a.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(a)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class SIRENMLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 siren_factor: float = 30.,
                 **kwargs):
        super().__init__()

        a = [SIRENLayer(in_size, hidden_size, siren_factor=siren_factor, **kwargs)]
        for i in range(num_layers - 1):
            a.append(SIRENLayer(hidden_size, hidden_size, siren_factor=siren_factor, **kwargs))
        a.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(a)
        self.initialize_weights(siren_factor)

    def initialize_weights(self, omega: float):
        """ See SIREN paper supplement Sec. 1.5 for discussion """
        old_weights = self.layers[1].linear.weight.clone()
        with torch.no_grad():
            # First layer initialization
            num_input = self.layers[0].linear.weight.size(-1)
            self.layers[0].linear.weight.uniform_(-1 / num_input, 1 / num_input)
            # Subsequent layer initialization uses based on omega parameter
            for layer in self.layers[1:-1]:
                num_input = layer.linear.weight.size(-1)
                layer.linear.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
            # Final linear layer also uses initialization based on omega parameter
            num_input = self.layers[-1].weight.size(-1)
            self.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)

        # Verify that weights did indeed change
        new_weights = self.layers[1].linear.weight
        assert (old_weights - new_weights).abs().sum() > 0.0

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class WIRENMLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 wire_omega: float = 10.0,
                 wire_input_skip: bool = False,
                 **kwargs):
        super().__init__()
        self.wire_input_skip = wire_input_skip

        a = [WIRELayer(in_size, hidden_size, wire_omega=wire_omega, **kwargs)]
        for i in range(num_layers - 1):
            a.append(WIRELayer(hidden_size, hidden_size, wire_omega=wire_omega, **kwargs))
        a.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(a)
        if self.wire_input_skip:
            self.input_skip = nn.Linear(in_size, out_size)
            nn.init.zeros_(self.input_skip.weight)
            nn.init.zeros_(self.input_skip.bias)
        self.initialize_weights(wire_omega)

    def initialize_weights(self, omega: float):
        old_weights = self.layers[1].freqs.weight.clone()
        with torch.no_grad():
            # First layer initialization
            num_input = self.layers[0].freqs.weight.size(-1)
            self.layers[0].freqs.weight.uniform_(-1 / num_input, 1 / num_input)
            self.layers[0].scale.weight.uniform_(-1 / num_input, 1 / num_input)
            # Subsequent layer initialization based on omega parameter
            for layer in self.layers[1:-1]:
                num_input = layer.freqs.weight.size(-1)
                layer.freqs.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
                layer.scale.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
            # Final linear layer also uses initialization based on omega parameter
            num_input = self.layers[-1].weight.size(-1)
            self.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
            self.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)

        # Verify that weights did indeed change
        new_weights = self.layers[1].freqs.weight
        assert (old_weights - new_weights).abs().sum() > 0.0

    def forward(self, x: torch.Tensor):
        x_in = x
        for layer in self.layers:
            x = layer(x)
        if self.wire_input_skip:
            x = x + self.input_skip(x_in)
        return x


class ResidualGatedBlock(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 2):
        super().__init__()
        inner = hidden_size * expansion
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return x + g * h


class ResidualGatedMLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 **kwargs):
        super().__init__()
        self.input_layer = nn.Linear(in_size, hidden_size)
        self.blocks = nn.ModuleList([
            ResidualGatedBlock(hidden_size=hidden_size, expansion=2)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, out_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        x = F.silu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class INCodeSineLayer(nn.Module):
    """INCODE sinusoidal layer: a * sin(b * omega * linear(x) + c) + d."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        first_layer_init: str = "siren_raw",
    ):
        super().__init__()
        self.in_size = in_size
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_size, out_size)
        self._init_weights(is_first=is_first, first_layer_init=first_layer_init)

    def _init_weights(self, is_first: bool, first_layer_init: str = "siren_raw"):
        with torch.no_grad():
            if is_first and first_layer_init == "siren_raw":
                # Canonical SIREN: for raw coordinate inputs in [-1, 1].
                bound = 1.0 / max(self.in_size, 1)
            else:
                # SIREN hidden-layer rule; also the correct rule for already-
                # positional-encoded inputs whose per-channel magnitude is O(1).
                bound = math.sqrt(6.0 / max(self.in_size, 1)) / max(self.omega_0, 1e-12)
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        a_param: torch.Tensor,
        b_param: torch.Tensor,
        c_param: torch.Tensor,
        d_param: torch.Tensor,
    ) -> torch.Tensor:
        z = self.linear(x)
        return a_param * torch.sin(b_param * self.omega_0 * z + c_param) + d_param


class INCODEMLP(nn.Module):
    """INCODE-style INR with a learnable prior embedding and harmonizer network."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        incode_first_omega_0: float = 30.0,
        incode_hidden_omega_0: float = 30.0,
        incode_latent_dim: int = 64,
        incode_prior_mode: str = "learned",
        incode_context_dim: int = 8,
        incode_layerwise_modulation: bool = False,
        incode_harmonizer_hidden: tuple = (64, 32),
        incode_harmonizer_bias: float = 0.31,
        incode_harmonizer_weight_std: float = 0.001,
        incode_activation_mode: str = "official_exp",
        incode_enforce_positive: bool = True,
        incode_input_skip: bool = False,
        incode_direct_modulation: bool = False,
        incode_first_layer_init: str = "siren_raw",
        incode_residual_skip: bool = False,
        incode_shore_factorized: bool = False,
        incode_shore_radial_order: int = 8,
        incode_shore_residual_init: float = 1.0,
        incode_output_refine_hidden: int = 0,
        # Idea 5: mono-exp residual factorization. The model's 95-dim output
        # becomes c_residual + project_to_shore(S0·exp(-b·D)), where S0 and D
        # are predicted from a 2-dim head off the trunk features. Final output
        # remains 95-dim so the locked evaluator's `out @ shore_M.T` path is
        # unchanged. Activates only when shore_pinv & bvals_train are passed.
        use_monoexp_head: bool = False,
        shore_pinv: torch.Tensor = None,
        bvals_train: torch.Tensor = None,
        monoexp_S0_init: float = 0.5,
        monoexp_D_init: float = 1.0e-3,
        **kwargs,
    ):
        super().__init__()
        self.incode_first_layer_init = incode_first_layer_init
        self.incode_residual_skip = incode_residual_skip
        self.incode_shore_factorized = bool(incode_shore_factorized)
        self.incode_shore_radial_order = int(incode_shore_radial_order)
        self.incode_shore_residual_init = float(incode_shore_residual_init)
        self.num_layers = num_layers
        self.incode_enforce_positive = incode_enforce_positive
        self.incode_prior_mode = incode_prior_mode
        self.incode_activation_mode = incode_activation_mode
        self.incode_layerwise_modulation = incode_layerwise_modulation
        self.incode_input_skip = incode_input_skip
        self.incode_direct_modulation = incode_direct_modulation
        self.num_mod_layers = num_layers + 1  # input layer + hidden sine layers

        # Prior embedding source.
        if self.incode_prior_mode == "b0_context":
            self.latent_code = None
            self.prior_context_encoder = nn.Sequential(
                nn.Linear(incode_context_dim, incode_latent_dim),
                nn.SiLU(),
                nn.Linear(incode_latent_dim, incode_latent_dim),
            )
            self.prior_fusion = None
            self.register_buffer("context_vec", torch.zeros(1, incode_context_dim))
        elif self.incode_prior_mode == "fusion":
            self.latent_code = nn.Parameter(torch.zeros(1, incode_latent_dim))
            self.prior_context_encoder = nn.Sequential(
                nn.Linear(incode_context_dim, incode_latent_dim),
                nn.SiLU(),
                nn.Linear(incode_latent_dim, incode_latent_dim),
            )
            self.prior_fusion = nn.Sequential(
                nn.Linear(2 * incode_latent_dim, incode_latent_dim),
                nn.SiLU(),
                nn.Linear(incode_latent_dim, incode_latent_dim),
            )
            self.register_buffer("context_vec", torch.zeros(1, incode_context_dim))
        else:
            # Default INCODE mode: single-scene learned prior token.
            self.latent_code = nn.Parameter(torch.zeros(1, incode_latent_dim))
            self.prior_context_encoder = None
            self.prior_fusion = None
            self.register_buffer("context_vec", torch.empty(0), persistent=False)

        if self.incode_direct_modulation:
            if self.incode_layerwise_modulation:
                self.raw_modulation = nn.Parameter(
                    torch.full((self.num_mod_layers, 4), float(incode_harmonizer_bias))
                )
            else:
                self.raw_modulation = nn.Parameter(torch.full((4,), float(incode_harmonizer_bias)))
            if incode_harmonizer_weight_std > 0:
                with torch.no_grad():
                    self.raw_modulation.add_(
                        torch.randn_like(self.raw_modulation) * float(incode_harmonizer_weight_std)
                    )
            self.harmonizer = None
        else:
            self.raw_modulation = None
            self.harmonizer = self._build_harmonizer(
                incode_latent_dim,
                incode_harmonizer_hidden,
                incode_harmonizer_bias,
                incode_harmonizer_weight_std,
                out_dim=4 * self.num_mod_layers if self.incode_layerwise_modulation else 4,
            )

        # Composer network.
        self.input_layer = INCodeSineLayer(
            in_size=in_size,
            out_size=hidden_size,
            omega_0=incode_first_omega_0,
            is_first=True,
            first_layer_init=incode_first_layer_init,
        )
        self.hidden_layers = nn.ModuleList(
            [
                INCodeSineLayer(
                    in_size=hidden_size,
                    out_size=hidden_size,
                    omega_0=incode_hidden_omega_0,
                    is_first=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_size, out_size)
        with torch.no_grad():
            bound = math.sqrt(6.0 / max(hidden_size, 1)) / max(incode_hidden_omega_0, 1e-12)
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()

        if self.incode_shore_factorized:
            N = self.incode_shore_radial_order
            n_indices, lm_indices = [], []
            for n in range(0, N + 1, 2):
                n_idx = n // 2
                for l in range(0, n + 1, 2):
                    lm_base = sum(2 * lp + 1 for lp in range(0, l, 2))
                    for m in range(-l, l + 1):
                        n_indices.append(n_idx)
                        lm_indices.append(lm_base + m + l)
            assert len(n_indices) == out_size
            self.register_buffer('shore_n_indices', torch.tensor(n_indices, dtype=torch.long), persistent=False)
            self.register_buffer('shore_lm_indices', torch.tensor(lm_indices, dtype=torch.long), persistent=False)
            n_radial = (N // 2) + 1
            n_angular = sum(2 * l + 1 for l in range(0, N + 1, 2))
            self.radial_head = nn.Linear(hidden_size, n_radial)
            self.angular_head = nn.Linear(hidden_size, n_angular)
            with torch.no_grad():
                # k chosen so Var(radial·angular per output) ≈ Var(Linear output).
                # Per-output Linear variance = hidden_size · σ_x² · bound²/3.
                # Per-output product variance = (hidden_size · σ_x² · k²/3)².
                # Match: k = (3·bound² / (hidden_size · σ_x²))^(1/4), σ_x≈1.
                k = (3.0 * bound * bound / max(hidden_size, 1)) ** 0.25
                self.radial_head.weight.uniform_(-k, k)
                self.radial_head.bias.zero_()
                self.angular_head.weight.uniform_(-k, k)
                self.angular_head.bias.zero_()
                self.output_layer.weight.mul_(float(self.incode_shore_residual_init))
        if self.incode_input_skip:
            self.input_skip = nn.Linear(in_size, out_size)
            nn.init.zeros_(self.input_skip.weight)
            nn.init.zeros_(self.input_skip.bias)
        else:
            self.input_skip = None

        if incode_output_refine_hidden and incode_output_refine_hidden > 0:
            refine_hidden = int(incode_output_refine_hidden)
            self.output_refine = nn.Sequential(
                nn.Linear(hidden_size, refine_hidden),
                nn.GELU(),
                nn.Linear(refine_hidden, out_size),
            )
            nn.init.zeros_(self.output_refine[-1].weight)
            nn.init.zeros_(self.output_refine[-1].bias)
        else:
            self.output_refine = None

        # Idea 5: mono-exp residual factorization. Activates if a fitted
        # SHORE pseudoinverse and training b-values are supplied. The head
        # outputs (log_S0, log_D) per voxel; their contribution to the final
        # 95-dim SHORE output is gated by sigmoid(raw_gate), with raw_gate
        # initialized very negative so the gate starts at ≈ 0 and the model
        # is equivalent to baseline at step 0. The gate then grows during
        # training to softly turn on the mono-exp baseline.
        self.use_monoexp_head = bool(use_monoexp_head) and (
            shore_pinv is not None and bvals_train is not None
        )
        if self.use_monoexp_head:
            self.monoexp_head = nn.Linear(hidden_size, 2)
            with torch.no_grad():
                self.monoexp_head.weight.zero_()
                self.monoexp_head.bias.data[0] = math.log(float(monoexp_S0_init))
                self.monoexp_head.bias.data[1] = math.log(float(monoexp_D_init))
            # Soft turn-on gate. Initial sigmoid(-6) ≈ 0.0025 → mono
            # contribution starts ≈ 0; trainable via a single scalar.
            self.monoexp_gate = nn.Parameter(torch.tensor(-6.0))
            # shore_pinv is [n_shore, n_volumes] = [95, 31]
            # bvals_train is [n_volumes]
            self.register_buffer(
                "_monoexp_shore_pinv", shore_pinv.float().clone(), persistent=True,
            )
            self.register_buffer(
                "_monoexp_bvals_train", bvals_train.float().clone(), persistent=True,
            )
        else:
            self.monoexp_head = None
            self.monoexp_gate = None

    @staticmethod
    def _build_harmonizer(
        latent_dim: int,
        hidden_dims,
        bias_value: float,
        weight_std: float,
        out_dim: int = 4,
    ) -> nn.Sequential:
        dims = [latent_dim] + [int(d) for d in hidden_dims] + [int(out_dim)]
        layers = []
        last_idx = len(dims) - 2
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if i == last_idx:
                # Keep output modulation layer close to paper defaults.
                nn.init.trunc_normal_(linear.weight, std=weight_std)
                if linear.bias is not None:
                    nn.init.constant_(linear.bias, bias_value)
            else:
                # Hidden harmonizer layers should preserve signal from latent code,
                # not be dominated by a fixed positive bias.
                nn.init.kaiming_uniform_(linear.weight, a=0.0, nonlinearity="relu")
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < last_idx:
                layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def set_context_vector(self, context_vec: torch.Tensor):
        """Set contextual prior vector for b0_context mode."""
        if self.incode_prior_mode not in {"b0_context", "fusion"}:
            return
        if context_vec.ndim == 1:
            context_vec = context_vec.unsqueeze(0)
        self.context_vec.copy_(context_vec.to(self.context_vec.device, dtype=self.context_vec.dtype))

    def _prior_embedding(self):
        if self.incode_prior_mode == "b0_context":
            return self.prior_context_encoder(self.context_vec)
        if self.incode_prior_mode == "fusion":
            ctx = self.prior_context_encoder(self.context_vec)
            fused = torch.cat([self.latent_code, ctx], dim=-1)
            return self.prior_fusion(fused)
        return self.latent_code

    def _activation_parameters(self):
        if self.incode_direct_modulation:
            raw = self.raw_modulation
        else:
            raw = self.harmonizer(self._prior_embedding())[0]  # [4]
            if self.incode_layerwise_modulation:
                raw = raw.view(self.num_mod_layers, 4)
        if self.incode_activation_mode == "official_exp":
            # Official INCODE parameterization:
            # y = exp(a) * sin(exp(b) * omega * x + c) + d
            if self.incode_layerwise_modulation:
                return torch.exp(raw[:, 0]), torch.exp(raw[:, 1]), raw[:, 2], raw[:, 3]
            return torch.exp(raw[0]), torch.exp(raw[1]), raw[2], raw[3]
        if not self.incode_enforce_positive:
            if self.incode_layerwise_modulation:
                return raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
            return raw[0], raw[1], raw[2], raw[3]

        # INCODE constraints from paper: a>=1, b>=1, c>=0, d>=0.
        if self.incode_layerwise_modulation:
            a_param = 1.0 + F.softplus(raw[:, 0])
            b_param = 1.0 + F.softplus(raw[:, 1])
            c_param = F.softplus(raw[:, 2])
            d_param = F.softplus(raw[:, 3])
        else:
            a_param = 1.0 + F.softplus(raw[0])
            b_param = 1.0 + F.softplus(raw[1])
            c_param = F.softplus(raw[2])
            d_param = F.softplus(raw[3])
        return a_param, b_param, c_param, d_param

    def activation_parameters(self):
        """Return detached activation parameters for monitoring."""
        a_param, b_param, c_param, d_param = self._activation_parameters()
        if self.incode_layerwise_modulation:
            def _stats(t):
                t = t.detach().cpu()
                return {
                    "mean": float(t.mean()),
                    "min": float(t.min()),
                    "max": float(t.max()),
                }
            return {
                "a": _stats(a_param),
                "b": _stats(b_param),
                "c": _stats(c_param),
                "d": _stats(d_param),
            }
        return {
            "a": float(a_param.detach().cpu()),
            "b": float(b_param.detach().cpu()),
            "c": float(c_param.detach().cpu()),
            "d": float(d_param.detach().cpu()),
        }

    def forward(self, x: torch.Tensor):
        x_in = x
        a_param, b_param, c_param, d_param = self._activation_parameters()
        if self.incode_layerwise_modulation:
            out = self.input_layer(x, a_param[0], b_param[0], c_param[0], d_param[0])
            for i, layer in enumerate(self.hidden_layers, start=1):
                new_out = layer(out, a_param[i], b_param[i], c_param[i], d_param[i])
                out = new_out + out if self.incode_residual_skip else new_out
        else:
            out = self.input_layer(x, a_param, b_param, c_param, d_param)
            for layer in self.hidden_layers:
                new_out = layer(out, a_param, b_param, c_param, d_param)
                out = new_out + out if self.incode_residual_skip else new_out

        hidden_feats = out
        if self.incode_shore_factorized:
            radial = self.radial_head(hidden_feats)
            angular = self.angular_head(hidden_feats)
            factorized = radial[:, self.shore_n_indices] * angular[:, self.shore_lm_indices]
            residual = self.output_layer(hidden_feats)
            out = factorized + residual
        else:
            out = self.output_layer(hidden_feats)

        if self.output_refine is not None:
            out = out + self.output_refine(hidden_feats)

        if self.monoexp_head is not None:
            # Predict (log_S0, log_D) per voxel; convert to S0, D > 0.
            log_params = self.monoexp_head(hidden_feats)
            log_S0 = log_params[:, 0]
            log_D = log_params[:, 1]
            S0 = torch.exp(log_S0)  # [B]
            D = torch.exp(log_D)    # [B]
            # Mono-exp signal at training shells: S0 · exp(-b · D)
            # bvals: [n_volumes]; broadcast → [B, n_volumes]
            bvals = self._monoexp_bvals_train  # [n_volumes]
            signal_mono = S0[:, None] * torch.exp(-bvals[None, :] * D[:, None])
            # Project onto SHORE basis: signal_mono @ pinv.T → [B, n_shore]
            # shore_pinv is [n_shore, n_volumes].
            mono_coeffs = signal_mono @ self._monoexp_shore_pinv.T
            gate = torch.sigmoid(self.monoexp_gate)
            out = out + gate * mono_coeffs

        if self.input_skip is not None:
            out = out + self.input_skip(x_in)
        return out


# ---------------------------------------------------------------------------
# PIN: Prolate Spheroidal Wave Function-based INR (Jayasundara et al., ICLR 2025)
# https://openreview.net/forum?id=Eh1QM3OK51
#
# SIREN-style MLP whose activation is the order-0 PSWF (optimal joint
# space-frequency concentration), instantiated as a learnable
#   psi_hat(x) = T * psi(w * x) + b
# with T (amplitude), w (frequency), b (height) learnable per layer.
# psi is the order-0 PSWF built in the normalized-Legendre basis via the
# prolate operator's tridiagonal eigenvector (scipy.pro_ang1 is numerically
# unstable, so we construct it directly). psi is evaluated by the Legendre
# recurrence on clamped t in [-1,1] (bounded => no polynomial blow-up).
# ---------------------------------------------------------------------------
import numpy as _np


def _pswf0_legendre_coef(c: float, max_even_k: int = 40):
    """Order-0 PSWF as coefficients on the *unnormalized* Legendre basis P_k.
    Returns a length-(max_even_k+1) numpy vector (odd entries 0; order 0 is even)."""
    ks = _np.arange(0, max_even_k + 1, 2)
    n = len(ks)
    A = _np.zeros((n, n))
    for i, k in enumerate(ks):
        A[i, i] = k * (k + 1) + c ** 2 * (2 * k * (k + 1) - 1) / ((2 * k - 1) * (2 * k + 3))
        if i + 1 < n:
            off = c ** 2 * (k + 1) * (k + 2) / ((2 * k + 3) * _np.sqrt((2 * k + 1) * (2 * k + 5)))
            A[i, i + 1] = off
            A[i + 1, i] = off
    w, v = _np.linalg.eigh(A)
    beta = v[:, 0]                       # smallest eigenvalue => order 0
    full = _np.zeros(max_even_k + 1)
    for k, b in zip(ks, beta):
        full[k] = b * _np.sqrt((2 * k + 1) / 2.0)   # de-normalize to plain P_k
    # normalize so peak ~1 (amplitude is then carried by learnable T)
    xg = _np.linspace(-1, 1, 401)
    yg = _np.polynomial.legendre.legval(xg, full)
    full = full / _np.max(_np.abs(yg))
    return full.astype(_np.float32)


class PSWFActivation(nn.Module):
    """psi_hat(x) = T * psi(w*x) + b ; psi = order-0 PSWF.

    The PSWF is precomputed once on a fine grid (via its Legendre expansion) and
    stored as a lookup table; at runtime the activation is clamp + index + linear
    interpolation — O(1) per element, vectorized, torch.compile-friendly (no
    per-call recurrence to unroll). As cheap as a sine activation."""
    def __init__(self, omega: float, c: float = 8.0, max_even_k: int = 20,
                 n_lut: int = 1025):
        super().__init__()
        full = _pswf0_legendre_coef(c, max_even_k)
        grid = _np.linspace(-1.0, 1.0, n_lut).astype(_np.float32)
        lut = _np.polynomial.legendre.legval(grid, full).astype(_np.float32)
        self.register_buffer("lut", torch.from_numpy(lut))
        self.n_lut = int(n_lut)
        self.T = nn.Parameter(torch.tensor(1.0))
        self.w = nn.Parameter(torch.tensor(float(omega)))
        self.b = nn.Parameter(torch.tensor(0.0))

    def _psi(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, -1.0, 1.0)
        pos = (t + 1.0) * 0.5 * (self.n_lut - 1)
        i0 = torch.floor(pos).long().clamp_(0, self.n_lut - 2)
        frac = pos - i0.to(pos.dtype)                  # grad flows through frac
        a = self.lut[i0]
        b = self.lut[i0 + 1]
        return a + (b - a) * frac

    def forward(self, x: torch.Tensor, mod=None) -> torch.Tensor:
        T, w, b = self.T, self.w, self.b
        if mod is not None:                 # additive B0-context modulation (mod~0 at init); beats exp form
            T = T + mod[0]; w = w + mod[1]; b = b + mod[2]
        return T * self._psi(w * x) + b


class PINLayer(nn.Module):
    def __init__(self, in_size, out_size, omega, c=8.0, max_even_k=20):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.act = PSWFActivation(omega, c=c, max_even_k=max_even_k)

    def forward(self, x, mod=None):
        return self.act(self.linear(x), mod)


class PINMLP(nn.Module):
    """Prolate Spheroidal Wave Function INR. SIREN-style init keeps w*(Wz)~unit
    so the clamped PSWF stays in its support."""
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 pin_first_omega: float = 24.0,
                 pin_hidden_omega: float = 17.0,
                 pin_c: float = 8.0,
                 pin_max_even_k: int = 40,
                 pin_b0_context: bool = False,
                 pin_context_dim: int = 10,
                 pin_harmonizer_hidden: int = 64,
                 pin_residual_skip: bool = False,
                 **kwargs):
        super().__init__()
        layers = [PINLayer(in_size, hidden_size, pin_first_omega, pin_c, pin_max_even_k)]
        for _ in range(num_layers - 1):
            layers.append(PINLayer(hidden_size, hidden_size, pin_hidden_omega, pin_c, pin_max_even_k))
        layers.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(layers)
        self.num_pin_layers = num_layers          # PINLayers (final Linear excluded)
        self.pin_residual_skip = bool(pin_residual_skip)
        self._init_weights(pin_first_omega, pin_hidden_omega)

        # Optional B0-context modulation (INCODE-style prior): a harmonizer maps a
        # subject-level B0-statistics vector to per-PIN-layer (dT, dw, db) offsets.
        # Initialized ~0 so it starts as the plain (working) PIN and learns to modulate.
        self.pin_b0_context = bool(pin_b0_context)
        if self.pin_b0_context:
            self.register_buffer("context_vec", torch.zeros(1, pin_context_dim))
            self.harmonizer = nn.Sequential(
                nn.Linear(pin_context_dim, pin_harmonizer_hidden), nn.ReLU(),
                nn.Linear(pin_harmonizer_hidden, self.num_pin_layers * 3),
            )
            nn.init.zeros_(self.harmonizer[-1].weight)
            nn.init.zeros_(self.harmonizer[-1].bias)

    def set_context_vector(self, context_vec: torch.Tensor):
        if not self.pin_b0_context:
            return
        if context_vec.ndim == 1:
            context_vec = context_vec.unsqueeze(0)
        self.context_vec.copy_(context_vec.to(self.context_vec.device, dtype=self.context_vec.dtype))

    def _init_weights(self, first_omega, hidden_omega):
        with torch.no_grad():
            n0 = self.layers[0].linear.weight.size(-1)
            self.layers[0].linear.weight.uniform_(-1.0 / n0, 1.0 / n0)
            nn.init.zeros_(self.layers[0].linear.bias)
            for layer in self.layers[1:-1]:
                ni = layer.linear.weight.size(-1)
                bound = math.sqrt(6.0 / ni) / hidden_omega
                layer.linear.weight.uniform_(-bound, bound)
                nn.init.zeros_(layer.linear.bias)
            nf = self.layers[-1].weight.size(-1)
            self.layers[-1].weight.uniform_(-math.sqrt(6.0 / nf) / hidden_omega,
                                            math.sqrt(6.0 / nf) / hidden_omega)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor):
        mod = None
        if self.pin_b0_context:
            mod = self.harmonizer(self.context_vec)[0].view(self.num_pin_layers, 3)
        for i, layer in enumerate(self.layers):
            if i < self.num_pin_layers:       # PIN layers get modulation; final Linear doesn't
                out = layer(x, None if mod is None else mod[i])
                # residual skip on hidden PIN layers (matching dims, i.e. not the first)
                x = out + x if (self.pin_residual_skip and i > 0 and out.shape == x.shape) else out
            else:
                x = layer(x)
        return x
