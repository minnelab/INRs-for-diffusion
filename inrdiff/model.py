import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from inrdiff.logging_utils import compute_image_metrics, normalize_within_mask, log_metrics_and_images, log_multirow_bvalue_visualization
import nibabel as nib
import os


class INRLightningModule(pl.LightningModule):
    def __init__(
        self,
        network,
        train_loader,
        val_loaders_dict,
        pos_encoder=None,
        lr=0.001,
        weight_decay=0.0,
        name="",
        val_slice_func=None,
        log_coords_raw=False,
        log_coords_encoded=False,
        eval_gradients=None,
        max_bval=None,
        steps_per_viz=2000,
        export_dwi=False,
        epochs=1000,
        use_cos_lrsheduler=False,
        train_log_interval=1,
        use_dropout=False,
        b0_image=None,
        bval_smooth_weight=0.0,
        bval_smooth_delta=0.05,
        sh_output=False,
        y_matrix=None,
        shore_output=False,
        shore_matrix=None,
        shore_b0_loss=False,
        shore_aniso_weight=0.0,
        shore_shell_norm=False,
        shore_coeff_weight=0.0,
        shore_range_weight=0.0,
        shore_range_max=1.0,
        b0_anchor_weight=0.0,
        q_smooth_weight=0.0,
        q_mono_weight=0.0,
        q_st_weight=0.0,
        q_convex_weight=0.0,
        q_monoexp_weight=0.0,
        b_inf_anchor_weight=0.0,
        b0_input_volume=None,
        b0_output_scale_volume=None,
        b0_output_scale_anchor=1.0,
        # Ball-and-multi-stick forward-model mode. When `model_output_type`
        # is 'ball_stick', the network outputs `4·n_sticks+3` raw parameters
        # per voxel that decode to (f_ball, f_sticks, D_iso, D_par, v_i),
        # and the predicted training signal is computed analytically via
        # `ball_stick.signal_at` instead of `outputs @ shore_matrix.T`.
        model_output_type="shore",
        ball_stick_n_sticks=2,
        ball_stick_use_fodf=False,
        ball_stick_compile=False,
        ball_stick_fodf_l_max=6,
        ball_stick_use_spot=False,
        ball_stick_spot_K=8,
        ball_stick_spot_voxel_size_norm=None,
        ball_stick_spot_smooth_weight=0.0,
        ball_stick_fodf_nonneg_weight=0.0,
        ball_stick_fodf_laplace_weight=0.0,
        ball_stick_fodf_unpin_c00=False,
        ball_stick_fodf_b0_anchor_weight=0.0,
        ball_stick_fodf_Y_dense=None,
        ball_stick_fodf_Y=None,
        ball_stick_fodf_b_idx_per_vol=None,
        ball_stick_fodf_unique_bvals=None,
        ball_stick_fodf_Pl_w=None,
        ball_stick_fodf_t=None,
        ball_stick_fodf_l_idx_per_coef=None,
        bvals_train_full=None,
        bvecs_train_full=None,
    ):
        super().__init__()
        self.sh_output = sh_output
        self.shore_output = shore_output
        self.shore_b0_loss = shore_b0_loss
        self.shore_aniso_weight = shore_aniso_weight
        self.shore_shell_norm = shore_shell_norm
        self.shore_coeff_weight = shore_coeff_weight
        self.shore_range_weight = shore_range_weight
        self.shore_range_max = shore_range_max
        self.b0_anchor_weight = b0_anchor_weight
        self.q_smooth_weight = q_smooth_weight
        self.q_mono_weight = q_mono_weight
        self.q_st_weight = q_st_weight
        self.q_convex_weight = q_convex_weight
        self.q_monoexp_weight = q_monoexp_weight
        self.b_inf_anchor_weight = b_inf_anchor_weight
        self._b0_batches = None  # set externally from dataset
        self._shell_indices = None  # list of index tensors per shell, set externally
        self._shell_weights = None  # optional shell weights aligned with _shell_indices
        self._shore_pinv = None  # pseudoinverse for coefficient-space loss, set externally
        self._shore_q0_row = None  # column-normalized q=0 SHORE row, set externally
        self._q_smooth_M_minus = None  # SHORE rows at q_centers - delta
        self._q_smooth_M_center = None  # SHORE rows at q_centers
        self._q_smooth_M_plus = None  # SHORE rows at q_centers + delta
        self._q_mono_M_low = None  # SHORE rows at b_low
        self._q_mono_M_high = None  # SHORE rows at b_low + delta_b
        self._b_inf_M = None  # SHORE rows at high b (signal should ≈ 0)
        self._q_monoexp_S0_batches = None  # per-voxel mono-exp targets, batched
        self._q_monoexp_D_batches = None
        self._q_smooth_b_centers = None  # b-values at q_smooth sample points
        # Per-voxel B0 conditioning: optional 3D volume looked up inside
        # forward() via grid_sample. Shape stored as [1, 1, Z, Y, X] so
        # grid_sample's (x→W, y→H, z→D) convention matches our (x, y, z)
        # normalized coords. None → input feature concatenation is skipped.
        if b0_input_volume is not None:
            # b0_input_volume: [X, Y, Z], already normalized like y targets.
            # Permute to [Z, Y, X] then add (N, C) dims.
            b0_5d = b0_input_volume.float().permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
            self.register_buffer('_b0_volume_input', b0_5d)
        else:
            self._b0_volume_input = None
        # Per-voxel S0 output scaling (Option 2 — fix held-out magnitude
        # truncation): multiply the model's SHORE coefficients by
        # `S0[voxel] / S0_anchor`. Lets high-S0 voxels (CSF, ventricles)
        # naturally exceed the training-shell signal range without the
        # model having to learn that magnitude from training data.
        # Stored in raw signal units; the divide by anchor normalizes.
        if b0_output_scale_volume is not None:
            b0_scale_5d = b0_output_scale_volume.float().permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
            self.register_buffer('_b0_volume_scale', b0_scale_5d)
            self.b0_output_scale_anchor = float(b0_output_scale_anchor)
        else:
            self._b0_volume_scale = None
            self.b0_output_scale_anchor = float(b0_output_scale_anchor)

        self.model_output_type = str(model_output_type)
        self.ball_stick_n_sticks = int(ball_stick_n_sticks)
        self.ball_stick_use_fodf = bool(ball_stick_use_fodf)
        self.ball_stick_compile = bool(ball_stick_compile)
        # Lazily-built (optionally torch.compiled) ball-fODF forward. The
        # analytic forward is launch-bound (~6-8 small kernels); compiling fuses
        # them (~2x) — see EXPERIMENT_LOG. Built on first training_step use.
        self._bs_fodf_callable = None
        self.ball_stick_fodf_l_max = int(ball_stick_fodf_l_max)
        self.ball_stick_use_spot = bool(ball_stick_use_spot)
        self.ball_stick_spot_K = int(ball_stick_spot_K)
        if ball_stick_spot_voxel_size_norm is not None:
            self.register_buffer(
                "_spot_voxel_size_norm",
                ball_stick_spot_voxel_size_norm.float(),
            )
        else:
            self._spot_voxel_size_norm = None
        self.ball_stick_fodf_nonneg_weight = float(ball_stick_fodf_nonneg_weight)
        self.ball_stick_fodf_laplace_weight = float(ball_stick_fodf_laplace_weight)
        self.ball_stick_fodf_unpin_c00 = bool(ball_stick_fodf_unpin_c00)
        self.ball_stick_fodf_b0_anchor_weight = float(ball_stick_fodf_b0_anchor_weight)
        if self.model_output_type == "ball_stick":
            if bvals_train_full is None or bvecs_train_full is None:
                raise ValueError(
                    "ball_stick mode requires bvals_train_full + bvecs_train_full"
                )
            self.register_buffer("_bs_bvals", bvals_train_full.float())
            self.register_buffer("_bs_bvecs", bvecs_train_full.float())
            if self.ball_stick_use_fodf:
                if ball_stick_fodf_Y is None:
                    raise ValueError(
                        "ball-fODF mode requires Y/quadrature buffers from main.py"
                    )
                self.register_buffer("_bs_fodf_Y", ball_stick_fodf_Y.float())
                self.register_buffer("_bs_fodf_b_idx_per_vol",
                                     ball_stick_fodf_b_idx_per_vol.long())
                self.register_buffer("_bs_fodf_unique_bvals",
                                     ball_stick_fodf_unique_bvals.float())
                self.register_buffer("_bs_fodf_Pl_w", ball_stick_fodf_Pl_w.float())
                self.register_buffer("_bs_fodf_t", ball_stick_fodf_t.float())
                self.register_buffer("_bs_fodf_l_idx_per_coef",
                                     ball_stick_fodf_l_idx_per_coef.long())
                if ball_stick_fodf_Y_dense is not None:
                    self.register_buffer("_bs_fodf_Y_dense",
                                         ball_stick_fodf_Y_dense.float())
                else:
                    self._bs_fodf_Y_dense = None
        if y_matrix is not None:
            self.register_buffer('y_matrix', y_matrix)
        if shore_matrix is not None:
            self.register_buffer('shore_matrix', shore_matrix)
        self.b0_image = b0_image
        self.bval_smooth_weight = bval_smooth_weight
        self.bval_smooth_delta = bval_smooth_delta
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cos_lrsheduler=use_cos_lrsheduler
        self.epochs=epochs
        self.network = network
        self.pos_encoder = self.setup_pos_encoder(pos_encoder)
        self.name = name
        self.val_slice_func = val_slice_func
        self.log_coords_raw = log_coords_raw
        self.log_coords_encoded = log_coords_encoded
        self.export_dwi=export_dwi
        self.use_dropout = use_dropout
        # attach dataloaders here - required to run multible
        self._train_loader = train_loader
        if val_loaders_dict is not None:
            self._val_loader_dict = val_loaders_dict
            self._val_loader_names = list(val_loaders_dict.keys())

        # Add gradient indices to evaluate - default to [0] if none provided
        self.eval_gradients = eval_gradients if eval_gradients is not None else [0]
        self.max_bval = max_bval
        self.steps_per_viz = steps_per_viz
        self.epsilon = 1e-8
        
        # Save hyperparameters for checkpoint loading, but exclude functions
        self.save_hyperparameters(ignore=['val_slice_func', 'train_slice_func', 'val_loaders_dict', 'train_loader', 'b0_image', 'y_matrix', 'shore_matrix'])

    def log(self, name, value, *args, **kwargs):
        """
        wrapper to log only in an interval during training
        """        
        kwargs.setdefault("on_step", False)
        kwargs.setdefault("on_epoch", True)
        # if self.training:
        #     if self.global_step % self.train_log_interval != 0:
        #         return  # skip logging this step
        super().log(name, value, *args, **kwargs)

    @staticmethod
    def setup_pos_encoder(pos_encoder):
        if pos_encoder is not None:
            if isinstance(pos_encoder, torch.nn.Module):
                pe = torch.nn.ModuleList([pos_encoder])
            else:
                pe = torch.nn.ModuleList(pos_encoder)
        else:
            pe = None

        return pe

    def configure_optimizers(self):
        # Optional param group split: output linear gets output_lr_mult * self.lr,
        # backbone gets self.lr. Scheduler still anchors on self.lr, so the
        # output-group LR scales proportionally during cos anneal.
        #
        # NOTE: the optimizer must cover parameters OUTSIDE self.network too. The
        # positional encoder is a registered submodule, so Lightning's summary lists
        # its parameters as trainable -- but building Adam over
        # self.network.parameters() silently excludes them, and they never update.
        # That made `spatial_freq_learnable: True` a no-op: the learnable Gaussian
        # frequency matrix B (3 x freq_num) was created, reported as 768 trainable
        # params, and then frozen in practice, reproducing the frozen-B run
        # BIT-IDENTICALLY. For every config where the encoder holds no parameters
        # (B registered as a buffer, i.e. all prior runs) this is exactly equivalent
        # to the old behaviour, so existing results are unaffected.
        net_ids = {id(p) for p in self.network.parameters()}
        extra_params = [p for p in self.parameters()
                        if id(p) not in net_ids and p.requires_grad]
        if extra_params:
            print(f"[optim] including {sum(p.numel() for p in extra_params)} non-network "
                  f"trainable params ({len(extra_params)} tensors, e.g. a learnable "
                  f"positional encoder)")
        output_lr_mult = getattr(self, '_output_lr_mult', None)
        if output_lr_mult is not None and output_lr_mult != 1.0 and hasattr(self.network, 'output_layer'):
            output_params = list(self.network.output_layer.parameters())
            output_ids = {id(p) for p in output_params}
            backbone_params = [p for p in self.network.parameters() if id(p) not in output_ids]
            optimizer = torch.optim.Adam(
                [
                    {'params': backbone_params + extra_params, 'lr': self.lr},
                    {'params': output_params, 'lr': self.lr * float(output_lr_mult)},
                ],
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                list(self.network.parameters()) + extra_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )

        if self.use_cos_lrsheduler:
            eta_min_ratio = getattr(self, 'cos_eta_min_ratio', 0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=self.lr * float(eta_min_ratio),
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # or 'step' if you want per-step updates
                    "frequency": 1,
                    "monitor": None
                }
            }

        else:
            return optimizer

    
    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        # return list here such that we can acess different dataloaders
        return list(self._val_loader_dict.values())

    def forward(self, coords):
        # Apply dropout to bvalues during training
        if self.training and self.use_dropout:
            # Get the original shape to restore later
            orig_shape = coords.shape
            
            # Extract bvalues which are at index 3
            bvals = coords[:, 3:4]

            # add a 20 % perturbation to the bvalues
            #bvals = bvals + (torch.rand_like(bvals)-0.5) * 0.2 * bvals
            
            # Apply dropout to bvalues with probability 0.2
            dropout_mask = torch.rand_like(bvals) > 0.5
            coords = coords.clone()  # Create a copy to avoid modifying the original
            coords[:, 3:4] = bvals * dropout_mask  # Zero out some bvalues

        # Save xyz for downstream lookups (b0 input feature, S0 output scale).
        xyz_norm_saved = coords[:, :3]

        # Per-voxel B0 lookup: do this BEFORE the encoder eats the
        # spatial coords. Normalized xyz in [-1, 1] → grid_sample on the
        # stored 3D B0 volume → scalar feature per sample.
        b0_feature = None
        if self._b0_volume_input is not None:
            grid = xyz_norm_saved.view(1, -1, 1, 1, 3).to(self._b0_volume_input.dtype)
            b0_lookup = F.grid_sample(
                self._b0_volume_input, grid,
                mode='bilinear', padding_mode='border', align_corners=False,
            )  # [1, 1, B, 1, 1]
            b0_feature = b0_lookup.view(-1, 1)  # [B, 1]

        if self.pos_encoder is not None:
            for encoder in self.pos_encoder:
                coords = encoder(coords)

        if b0_feature is not None:
            coords = torch.cat([coords, b0_feature], dim=-1)

        if self.log_coords_encoded:
            for i in range(coords.shape[-1]):
                self.logger.experiment.add_histogram(
                    f"encoded_coords_{i}", coords[:, i], self.global_step)

        out = self.network(coords)

        # Per-voxel S0 output scaling: multiply SHORE coefficients by
        # (S0[voxel] / S0_anchor). At eval, signal(b) = coeffs @ shore_M
        # is then also scaled by this ratio, letting CSF voxels (high S0)
        # naturally reach higher signal magnitudes than the training-shell
        # range allowed. SKIPPED for ball_stick: the model's output is
        # interpretable PARAMETERS (log-D, fractions, stick dirs), not
        # signal-space coefficients — scaling all entries uniformly would
        # corrupt them. Ball-stick handles S0 via the per-voxel scaling
        # inside `training_step`.
        if self._b0_volume_scale is not None and self.model_output_type != "ball_stick":
            grid = xyz_norm_saved.view(1, -1, 1, 1, 3).to(self._b0_volume_scale.dtype)
            s0_lookup = F.grid_sample(
                self._b0_volume_scale, grid,
                mode='bilinear', padding_mode='border', align_corners=False,
            ).view(-1, 1)  # [B, 1]
            scale_ratio = s0_lookup / max(self.b0_output_scale_anchor, 1e-6)
            out = out * scale_ratio

        return out

    def training_step(self, batch, batch_idx):

        coords, values, _ = batch
        coords = coords.view(-1, coords.shape[-1])
        values = values.view(-1, values.shape[-1])
        n_data = coords.shape[0]

        # B-value smoothness: append perturbed coords to the batch
        # (not applicable for SHORE output — no bval in input)
        if self.bval_smooth_weight > 0 and not self.shore_output:
            n_smooth = max(1, n_data // 5)
            idx = torch.randperm(n_data, device=coords.device)[:n_smooth]
            coords_sub = coords[idx]

            delta = self.bval_smooth_delta
            coords_plus = coords_sub.clone()
            coords_minus = coords_sub.clone()
            coords_plus[:, 3] = (coords_sub[:, 3] + delta).clamp(0, 1)
            coords_minus[:, 3] = (coords_sub[:, 3] - delta).clamp(0, 1)

            # Single forward pass: [data | plus | minus]
            all_coords = torch.cat([coords, coords_plus, coords_minus], dim=0)
            all_outputs = self.forward(all_coords)

            outputs = all_outputs[:n_data]
            out_plus = all_outputs[n_data:n_data + n_smooth]
            out_minus = all_outputs[n_data + n_smooth:]
            out_center = outputs[idx]

            second_deriv = (out_plus - 2 * out_center + out_minus) / (delta ** 2)
            smooth_loss = torch.mean(second_deriv ** 2)
        elif (self.model_output_type == "ball_stick"
              and self.ball_stick_use_spot
              and self._spot_voxel_size_norm is not None):
            # SPOT-style sub-voxel sampling: per voxel, sample K sub-positions
            # uniformly in the voxel cube, forward each, then average signals.
            K = max(1, int(self.ball_stick_spot_K))
            B = coords.shape[0]
            vs = self._spot_voxel_size_norm.to(coords.device)  # [3]
            offsets = (torch.rand(B, K, 3, device=coords.device) - 0.5) * vs
            sub_coords = coords.unsqueeze(1) + offsets       # [B, K, 3]
            outputs_sub = self.forward(sub_coords.view(B * K, 3))
            outputs = outputs_sub                            # [B*K, n_raw]
        else:
            outputs = self.forward(coords)

        # Compute loss: convert outputs to signal first.
        if self.model_output_type == "ball_stick":
            # outputs are raw ball-stick params per voxel; forward via
            # the analytical signal model. Output is normalized attenuation
            # in units of S0. Per-voxel S0 scaling has already been applied
            # to outputs via the b0_output_scale path if active. The
            # ball-stick model itself outputs attenuation in [0, 1].
            from inrdiff.ball_stick import (signal_at as _bs_signal_at,
                                    signal_at_fodf as _bs_signal_at_fodf)
            # NOTE: when use_b0_output_scale is on, `outputs` has already
            # been multiplied by S0(voxel)/anchor. For ball_stick we do NOT
            # want the parameter-space output scaled; revert that by using
            # the same scale_ratio in the inverse — but the cleanest path
            # is to disable use_b0_output_scale in ball_stick configs.
            if self.ball_stick_use_fodf:
                quad_data = {"t": self._bs_fodf_t, "Pl_w": self._bs_fodf_Pl_w}
                if self._bs_fodf_callable is None:
                    if self.ball_stick_compile:
                        try:
                            self._bs_fodf_callable = torch.compile(_bs_signal_at_fodf)
                            print("[torch.compile] ball-fODF forward compiled")
                        except Exception as exc:
                            print(f"[torch.compile] ball-fODF compile failed, eager: {exc}")
                            self._bs_fodf_callable = _bs_signal_at_fodf
                    else:
                        self._bs_fodf_callable = _bs_signal_at_fodf
                signal_E = self._bs_fodf_callable(
                    outputs, self._bs_bvals, self._bs_bvecs,
                    l_max=self.ball_stick_fodf_l_max,
                    Y_at_bvecs=self._bs_fodf_Y,
                    b_idx_per_vol=self._bs_fodf_b_idx_per_vol,
                    unique_bvals=self._bs_fodf_unique_bvals,
                    quad_data=quad_data,
                    l_idx_per_coef=self._bs_fodf_l_idx_per_coef,
                    unpin_c00=self.ball_stick_fodf_unpin_c00,
                )
            else:
                signal_E = _bs_signal_at(
                    outputs, self._bs_bvals, self._bs_bvecs,
                    n_sticks=self.ball_stick_n_sticks,
                )  # [B(*K), n_volumes] in [0, 1]
                if self.ball_stick_use_spot:
                    K = max(1, int(self.ball_stick_spot_K))
                    B = signal_E.shape[0] // K
                    signal_E = signal_E.view(B, K, -1).mean(dim=1)
            # Multiply by per-voxel S0 (looked up via the same grid_sample
            # as b0_output_scale) to recover signal in the training-target
            # normalized scale. The training target `values` is already
            # normalized via (y - y_min) / y_range, so we have to convert
            # `signal_E * S0_voxel` from raw-signal units back into the
            # normalized target space.
            if self._b0_volume_scale is None or self._b0_batches is None:
                raise RuntimeError(
                    "ball_stick mode requires b0_output_scale_volume to "
                    "provide per-voxel S0 and b0_batches for normalization."
                )
            S0_voxel = self._b0_batches[batch_idx]  # [B] in normalized space
            predicted = signal_E * S0_voxel.unsqueeze(-1)  # [B, n_volumes]
            # Per-shell normalized MSE if shell_indices available; else plain.
            if self.shore_shell_norm and self._shell_indices is not None:
                shell_losses = []
                for idx_tensor in self._shell_indices:
                    p_shell = predicted[:, idx_tensor]
                    v_shell = values[:, idx_tensor]
                    shell_losses.append(F.mse_loss(p_shell, v_shell))
                shell_losses = torch.stack(shell_losses)
                if self._shell_weights is not None and self._shell_weights.numel() == shell_losses.numel():
                    weights = self._shell_weights.to(shell_losses.device)
                    weights = weights / weights.sum().clamp_min(1e-8)
                    loss = (shell_losses * weights).sum()
                else:
                    loss = shell_losses.mean()
            else:
                loss = F.mse_loss(predicted, values)
            # CSD-style non-negativity penalty on the fODF (sampled densely).
            if (self.ball_stick_use_fodf
                    and self.ball_stick_fodf_nonneg_weight > 0
                    and self._bs_fodf_Y_dense is not None):
                from inrdiff.ball_stick import decode_params_fodf as _bs_decode_fodf
                p_fodf = _bs_decode_fodf(outputs, self.ball_stick_fodf_l_max,
                                         unpin_c00=self.ball_stick_fodf_unpin_c00)
                fODF_at_grid = p_fodf["fODF_lm"] @ self._bs_fodf_Y_dense.T
                # Penalize negative values (CSD soft constraint).
                neg_part = F.relu(-fODF_at_grid)
                nonneg_penalty = (neg_part ** 2).mean()
                loss = loss + self.ball_stick_fodf_nonneg_weight * nonneg_penalty
                self.log(f"{self.name}/fodf_nonneg_penalty", nonneg_penalty,
                         on_step=False, on_epoch=True)
            # Laplace-Beltrami SH smoother: Δ_S² Y_lm = -l(l+1)·Y_lm, so the
            # spherical-Laplacian energy of the fODF is Σ l(l+1)·|c_lm|².
            # Penalizing it shrinks high-band coefs (l>0) and biases the fODF
            # toward smooth angular profiles — physically motivated and the
            # standard CSD smoothness prior at fixed L_max.
            if (self.ball_stick_use_fodf
                    and self.ball_stick_fodf_laplace_weight > 0
                    and self._bs_fodf_l_idx_per_coef is not None):
                from inrdiff.ball_stick import decode_params_fodf as _bs_decode_fodf
                p_fodf = _bs_decode_fodf(outputs, self.ball_stick_fodf_l_max,
                                         unpin_c00=self.ball_stick_fodf_unpin_c00)
                fODF_free = p_fodf["fODF_lm"][:, 1:]  # drop l=0
                l_idx = self._bs_fodf_l_idx_per_coef[1:].to(fODF_free.device)
                weight_l = (l_idx * (l_idx + 1)).float()
                laplace_penalty = (weight_l * fODF_free.square()).mean()
                loss = loss + self.ball_stick_fodf_laplace_weight * laplace_penalty
                self.log(f"{self.name}/fodf_laplace_penalty", laplace_penalty,
                         on_step=False, on_epoch=True)
            # b=0 anchor: with c_00 unpinned the model's E(b=0) is no longer
            # ≡ 1; without this penalty the gauge between (f_ball, c_00) is
            # free and the model can split iso+aniso mass arbitrarily. Pull
            # the analytic E(b=0) toward 1 so AFD becomes physically meaningful.
            if (self.ball_stick_use_fodf
                    and self.ball_stick_fodf_unpin_c00
                    and self.ball_stick_fodf_b0_anchor_weight > 0):
                from inrdiff.ball_stick import e_at_b0_fodf as _bs_e_b0
                e_b0 = _bs_e_b0(outputs, self.ball_stick_fodf_l_max,
                                unpin_c00=True)
                b0_anchor = (e_b0 - 1.0).square().mean()
                loss = loss + self.ball_stick_fodf_b0_anchor_weight * b0_anchor
                self.log(f"{self.name}/fodf_b0_anchor",
                         b0_anchor, on_step=False, on_epoch=True)
            self.log(f"{self.name}/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("lr", current_lr)
            with torch.no_grad():
                relative_error = torch.mean(
                    torch.abs(predicted - values) / (torch.abs(values) + 1e-3)
                ) * 100
                self.log(f"{self.name}/train_relative_error_%", relative_error, on_step=False, prog_bar=True)
            return loss
        elif self.shore_output:
            shore_signal = outputs @ self.shore_matrix.T  # [B, n_measurements]
            if self.shore_b0_loss and self._b0_batches is not None:
                b0_batch = self._b0_batches[batch_idx].unsqueeze(-1)
                predicted = shore_signal * b0_batch
            else:
                predicted = shore_signal

            # Per-shell normalized MSE: equalize b-shell contributions
            if self.shore_shell_norm and self._shell_indices is not None:
                shell_losses = []
                for idx_tensor in self._shell_indices:
                    p_shell = predicted[:, idx_tensor]
                    v_shell = values[:, idx_tensor]
                    shell_losses.append(F.mse_loss(p_shell, v_shell))
                shell_losses = torch.stack(shell_losses)
                if self._shell_weights is not None and self._shell_weights.numel() == shell_losses.numel():
                    weights = self._shell_weights.to(shell_losses.device)
                    weights = weights / weights.sum().clamp_min(1e-8)
                    loss = (shell_losses * weights).sum()
                else:
                    loss = shell_losses.mean()
            else:
                loss = F.mse_loss(predicted, values)

            # Angular contrast loss: penalize under-fitting directional variation
            if self.shore_aniso_weight > 0:
                pred_mean = predicted.mean(dim=1, keepdim=True)
                val_mean = values.mean(dim=1, keepdim=True)
                aniso_loss = F.mse_loss(predicted - pred_mean, values - val_mean)
                self.log(f"{self.name}/aniso_loss", aniso_loss)
                loss = loss + self.shore_aniso_weight * aniso_loss

            # Keep predicted SHORE-reconstructed signal in the physical/normalized range.
            if self.shore_range_weight > 0:
                low_violation = torch.relu(-predicted)
                high_violation = torch.relu(predicted - self.shore_range_max)
                range_loss = (low_violation.square() + high_violation.square()).mean()
                self.log(f"{self.name}/range_loss", range_loss)
                loss = loss + self.shore_range_weight * range_loss

            # Coefficient-space loss: equal weight per angular order
            if self.shore_coeff_weight > 0 and self._shore_pinv is not None:
                gt_coeffs = values @ self._shore_pinv.T  # [B, n_shore]
                coeff_loss = F.mse_loss(outputs, gt_coeffs)
                self.log(f"{self.name}/coeff_loss", coeff_loss)
                loss = loss + self.shore_coeff_weight * coeff_loss

            # q=0 (B0) anchor loss: pull predicted SHORE coefficients toward
            # the per-voxel B0 signal. The held-out basis ceiling rises from
            # 3.51 PSNR_g (unanchored) to 14.62 PSNR_g (anchored, train-scale)
            # under this constraint — see program.md §9.NB and the
            # eval_shore_holdout_ceiling.py probe.
            if (
                self.b0_anchor_weight > 0
                and self._shore_q0_row is not None
                and self._b0_batches is not None
            ):
                pred_b0 = outputs @ self._shore_q0_row  # [B]
                b0_target = self._b0_batches[batch_idx].to(pred_b0.dtype)  # [B]
                anchor_loss = F.mse_loss(pred_b0, b0_target)
                self.log(f"{self.name}/b0_anchor_loss", anchor_loss)
                loss = loss + self.b0_anchor_weight * anchor_loss

            # Q-space smoothness regularizer: penalize 2nd-order finite
            # difference of predicted signal w.r.t. b along random q-points.
            # Attacks the high-frequency q-space oscillation that the per-voxel
            # min-norm SHORE pinv produces in the under-determined regime.
            if (
                self.q_smooth_weight > 0
                and self._q_smooth_M_center is not None
            ):
                s_minus = outputs @ self._q_smooth_M_minus.T  # [B, n_smooth]
                s_center = outputs @ self._q_smooth_M_center.T
                s_plus = outputs @ self._q_smooth_M_plus.T
                second_diff = s_plus - 2.0 * s_center + s_minus
                smooth_loss = second_diff.square().mean()
                self.log(f"{self.name}/q_smooth_loss", smooth_loss)
                loss = loss + self.q_smooth_weight * smooth_loss

            # High-b decay anchor: at b far above the training range, the
            # signal should asymptote to 0 (Stejskal-Tanner: S ∝ exp(-b·D)
            # → 0 as b → ∞). Penalizes predicted signal magnitude at a
            # fixed grid of high-b q-points.
            if (
                self.b_inf_anchor_weight > 0
                and self._b_inf_M is not None
            ):
                s_inf = outputs @ self._b_inf_M.T  # [B, n_inf]
                inf_loss = s_inf.square().mean()
                self.log(f"{self.name}/b_inf_anchor_loss", inf_loss)
                loss = loss + self.b_inf_anchor_weight * inf_loss

            # Per-voxel mono-exp consistency: at sampled q-points, the
            # predicted signal should approach S0[v]·exp(-b·D[v]) where
            # (S0, D) are per-voxel estimates fitted from training shells
            # via 2-shell log-linear regression. Stronger than b_inf_anchor
            # (single asymptotic point) because it constrains the entire
            # S(b) curve to mono-exp shape per voxel.
            if (
                self.q_monoexp_weight > 0
                and self._q_smooth_M_center is not None
                and self._q_monoexp_S0_batches is not None
                and self._q_smooth_b_centers is not None
            ):
                S0_b = self._q_monoexp_S0_batches[batch_idx]  # [B]
                D_b = self._q_monoexp_D_batches[batch_idx]    # [B]
                bvals_c = self._q_smooth_b_centers              # [n_smooth]
                target = S0_b[:, None] * torch.exp(-bvals_c[None, :] * D_b[:, None])
                pred = outputs @ self._q_smooth_M_center.T
                monoexp_loss = (pred - target).square().mean()
                self.log(f"{self.name}/q_monoexp_loss", monoexp_loss)
                loss = loss + self.q_monoexp_weight * monoexp_loss

            # Stejskal-Tanner log-space smoothness: under mono-exp decay
            # S(b) = S0·exp(-b·D), log(S) is LINEAR in b — i.e., its 2nd
            # derivative w.r.t. b is exactly zero. q_smooth penalizes 2nd
            # derivative of S itself (which is non-zero for exponentials);
            # this term penalizes 2nd derivative of log(S) instead, which
            # is the physically correct smoothness prior.
            # Reuses q_smooth's M_minus/center/plus sample points.
            if (
                self.q_st_weight > 0
                and self._q_smooth_M_center is not None
            ):
                s_minus = outputs @ self._q_smooth_M_minus.T
                s_center = outputs @ self._q_smooth_M_center.T
                s_plus = outputs @ self._q_smooth_M_plus.T
                # softplus + epsilon keeps log() finite even if a predicted
                # signal goes briefly negative (rare; smooth NaN-safe).
                eps = 1e-3
                log_s_minus = torch.log(F.softplus(s_minus) + eps)
                log_s_center = torch.log(F.softplus(s_center) + eps)
                log_s_plus = torch.log(F.softplus(s_plus) + eps)
                log_second_diff = log_s_plus - 2.0 * log_s_center + log_s_minus
                st_loss = log_second_diff.square().mean()
                self.log(f"{self.name}/q_st_loss", st_loss)
                loss = loss + self.q_st_weight * st_loss

            # Q-space convexity constraint: under mono-exp decay
            # S(b) = S0·exp(-b·D), the 2nd derivative ∂²S/∂b² = S·D² is
            # STRICTLY POSITIVE. q_smooth penalizes deviations of any sign
            # from zero curvature, but the truly physical constraint is
            # only that NEGATIVE curvature (concave dips) is unphysical.
            # Penalize relu(-(s_plus - 2·s_center + s_minus))² to allow
            # natural convexity while suppressing concave excursions.
            # Reuses q_smooth's M_minus/center/plus sample points.
            if (
                self.q_convex_weight > 0
                and self._q_smooth_M_center is not None
            ):
                s_minus = outputs @ self._q_smooth_M_minus.T
                s_center = outputs @ self._q_smooth_M_center.T
                s_plus = outputs @ self._q_smooth_M_plus.T
                second_diff = s_plus - 2.0 * s_center + s_minus
                # Convex (positive 2nd-diff) is OK; penalize only negative.
                concave_penalty = F.relu(-second_diff).square().mean()
                self.log(f"{self.name}/q_convex_loss", concave_penalty)
                loss = loss + self.q_convex_weight * concave_penalty

            # Q-space monotonic-decay constraint: signal must not increase
            # with b along any direction (Stejskal-Tanner: S(b) ∝ exp(-b·D)).
            # Sample random (b_low, b_high) pairs and penalize relu(s_high -
            # s_low)². Pinv cannot enforce this — it's non-linear in c.
            if (
                self.q_mono_weight > 0
                and self._q_mono_M_low is not None
            ):
                s_low = outputs @ self._q_mono_M_low.T   # [B, n_mono]
                s_high = outputs @ self._q_mono_M_high.T
                mono_loss = F.relu(s_high - s_low).square().mean()
                self.log(f"{self.name}/q_mono_loss", mono_loss)
                loss = loss + self.q_mono_weight * mono_loss
        elif self.sh_output:
            predicted = outputs @ self.y_matrix.T  # [B, n_dirs]
            loss = F.mse_loss(predicted, values)
        else:
            predicted = outputs
            loss = F.mse_loss(outputs, values)

        if self.bval_smooth_weight > 0 and not self.shore_output:
            loss = loss + self.bval_smooth_weight * smooth_loss
            self.log(f"{self.name}/smooth_loss", smooth_loss)

        # logging
        self.log(f"{self.name}/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr)

        # Calculate and log relative error (MAPE) with increased epsilon
        with torch.no_grad():
            relative_error = torch.mean(
                torch.abs(predicted - values) / (torch.abs(values) + 1e-3)
            ) * 100

            self.log(f"{self.name}/train_relative_error_%", relative_error, on_step=False, prog_bar=True)

            # Compute per-bvalue errors if b-values present
            if coords.shape[-1] >= 4:
                bvals = coords[:, 3]
                unique_bvals = torch.unique(bvals)

                for bval in unique_bvals:
                    mask = torch.isclose(bvals, bval, rtol=1e-5)
                    if mask.any():
                        values_masked_safe = torch.clamp(torch.abs(values[mask]), min=self.epsilon)
                        bval_error = torch.mean(
                            torch.abs(predicted[mask] - values[mask]) / values_masked_safe
                        ) * 100

                        if self.max_bval is not None:
                            actual_bval = float(bval.item()) * float(self.max_bval)
                            rounded_bval = round(actual_bval / 100) * 100
                        else:
                            rounded_bval = round(float(bval.item()) / 100) * 100

                        self.log(f"{self.name}/relative_error_b{rounded_bval:.0f}", bval_error)

        return loss
    
    @staticmethod
    def normalize_for_tensorboard(img):
        """Normalize tensor to [0,1] for TensorBoard logging"""
        img_min = img.min()
        img_max = img.max()

        return (img - img_min) / (img_max - img_min + 1e-8)  # Avoid division by zero

    def get_bval_from_idx(self, b_idx, val_type):

        loader = self._val_loader_dict[val_type]
    
        # Get the b-value identifier for logging
        b_value = loader.dataset.b_indices[b_idx]
        if hasattr(loader.dataset.reference_dataset, 'bvals'):
            actual_bval = loader.dataset.reference_dataset.bvals[b_value].item()
            rounded_bval = round(actual_bval / 100) * 100
        else:
            rounded_bval = b_value

        return rounded_bval

    def validate_slice(self, x_list, y_list):
        # Initialize metrics for averaging across b-values (better: do this for volume)
        total_loss = 0
        total_rel_error = 0
         
         # Process each b-value separately
        for b_idx, (x, y) in enumerate(zip(x_list, y_list)):

            # SLICE METRICS

            # Forward pass for this b-value
            
            y_hat = self.forward(x)

            y_slice = self.normalize_for_tensorboard(self.val_slice_func(x, y, denormalize=True, b_idx_pos=b_idx))
            y_hat_slice = self.normalize_for_tensorboard(self.val_slice_func(x, y_hat.squeeze(), denormalize=True, b_idx_pos=b_idx))
            se_slice = (y_slice - y_hat_slice) ** 2
                    
            # Compute image quality metrics
            psnr_val, ssim_val, lpips_val = compute_image_metrics(
                y_hat_slice, y_slice, device=self.device
            )
                    
            rounded_bval = self.get_bval_from_idx(b_idx=b_idx, val_type='slice')

            # Log metrics for this b-value
            self.log(f"{self.name}/val_psnr_b{rounded_bval}", psnr_val)
            self.log(f"{self.name}/val_ssim_b{rounded_bval}", ssim_val)
            self.log(f"{self.name}/val_lpips_b{rounded_bval}", lpips_val)
                    
            # Log images for this b-value
            self.logger.experiment.add_image(
                f"{self.name}/val_target_b{rounded_bval}",
                y_slice.unsqueeze(0),
                self.global_step
            )
            self.logger.experiment.add_image(
                f"{self.name}/val_prediction_b{rounded_bval}",
                y_hat_slice.unsqueeze(0),
                self.global_step
            )
            self.logger.experiment.add_image(
                f"{self.name}/val_squared_error_b{rounded_bval}",
                se_slice.unsqueeze(0),
                self.global_step
            )

            # GENERAL METRICS, evaluated on slice (for now)
            # Calculate loss
            loss = F.mse_loss(y_hat, y)
            total_loss += loss
                
            # Calculate relative error with increased epsilon and clamping
            epsilon = 1e-8
            y_safe = torch.clamp(torch.abs(y), min=epsilon)
            rel_error = torch.mean(torch.abs(y_hat - y) / y_safe) * 100
            rel_error = torch.clamp(rel_error, max=1000.0)
            total_rel_error += rel_error
                
            # Log individual b-value metrics
            self.log(f"{self.name}/val_loss_b{rounded_bval}", loss)
            self.log(f"{self.name}/val_relative_error_b{rounded_bval}_%", rel_error)
                
           
        # Calculate and log average metrics across b-values
        num_b_values = len(x_list)
        avg_loss = total_loss / num_b_values
        avg_rel_error = total_rel_error / num_b_values
            
        self.log(f"{self.name}/val_loss", avg_loss)
        self.log(f"{self.name}/val_relative_error_%", avg_rel_error)

        return avg_loss
 


    def validate_volume(self, x_, y_):

        if self.export_dwi:
            # Process all b-values together (that are included in this volume)
            y_hat = self.forward(x_)

            dset = self._val_loader_dict['volume'].dataset             

            # fold ...
            dwi = dset._fold_xyz(x_, y_, skip_checks=True, denormalize=True, force_int_coords=True)
            dwi_hat = dset._fold_xyz(x_, y_hat, skip_checks=True, denormalize=True, force_int_coords=True) # bit hacky with the int coords

            # dump to disk
            nib.save(nib.Nifti1Image(dwi.cpu().numpy().astype(float), affine=np.eye(4)), 
                     os.path.join(self.logger.log_dir, 'dwi.nii.gz'))
            nib.save(nib.Nifti1Image(dwi_hat.cpu().numpy().astype(float), affine=np.eye(4)), 
                     os.path.join(self.logger.log_dir, 'dwi_hat.nii.gz'))

            # also save bvecs and bvals here bcause why not
            bvecs = dset.bvecs.T 
            np.savetxt(os.path.join(self.logger.log_dir, "bvecs"), 
                       bvecs, fmt="%.5f", delimiter=" ")
            bvals = dset.bvals
            np.savetxt(os.path.join(self.logger.log_dir, "bvals"), 
                       bvals[np.newaxis, :], fmt="%d", delimiter=" ")

        else:
            return # Do nothing for now


    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # fetch type of dataloader 
        val_type = self._val_loader_names[dataloader_idx]
        
        # Unpack the batch - now x and y are lists of tensors, one per b-value
        x_list, y_list, idx = batch
        
        # case 1: just logging via slice dataloader
        # Generate and log slice visualizations for this b-value
        if self.val_slice_func is not None and val_type=='slice':
            self.validate_slice(x_list, y_list)

            return

        # case 2: evaluate on a volume - no logging of img to tb!
        elif val_type=='volume':
            avg_loss = self.validate_volume(x_list, y_list)

            return avg_loss

        else:
            raise ValueError('unrecognized validation dataloader type, pick <slice, volume>')
