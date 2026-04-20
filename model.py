"""PyTorch implementation of the PNAS splicing model and related helpers."""

# Torch Analog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def lanczos_kernel(x, order):
    """Compute Lanczos kernel weights for the given offsets.

    Args:
        x: Scalar or array-like offsets from the interpolation target.
        order: Size of the Lanczos window.

    Returns:
        A scalar or NumPy array of Lanczos kernel weights with the same shape
        as ``x``.
    """
    return np.sinc(x) * np.sinc(x/order) * ((x > -order) * (x < order))

def lanczos_interpolate(arr, positions, order=3):
    """Interpolate a 1D array at arbitrary positions with a Lanczos kernel.

    Args:
        arr: One-dimensional NumPy array to sample from.
        positions: Array-like floating-point sample locations.
        order: Size of the Lanczos window. Defaults to ``3``.

    Returns:
        A NumPy array containing interpolated values for each input position.
    """
    result = np.zeros_like(positions)
    # Written this way to support non-scalar x.
    for i, x in enumerate(positions):
        i_min, i_max = int(np.floor(x) - order +1), int(np.floor(x) + order + 1)
        i_min, i_max = max(i_min, 0), min(i_max, len(arr))
        window = np.arange(i_min, i_max)
        result[i] = np.sum(arr[window] * lanczos_kernel(x - window, order))

    return result

def lanczos_resampling(arr, new_len, order=3):
    """Resample a 1D array to a new length with Lanczos interpolation.

    Args:
        arr: One-dimensional NumPy array to resample.
        new_len: Number of output samples.
        order: Size of the Lanczos window. Defaults to ``3``.

    Returns:
        A NumPy array of length ``new_len``.
    """
    return lanczos_interpolate(arr, np.linspace(0, len(arr)-1, num=new_len), order)

class SumDiff(nn.Module):
    """Aggregate inclusion and skipping activations into a scalar energy."""

    def __init__(self):
        super(SumDiff, self).__init__()
        self.w = nn.Parameter(torch.randn(1))  # Learnable weight
        self.b = nn.Parameter(torch.zeros(1))   # Learnable bias

    def forward(self, x):
        """Compute the weighted sum-difference score.

        Args:
            x: Tensor of shape ``(batch_size, 2, num_filters, seq_length)``.
                Index ``0`` is treated as inclusion and index ``1`` as skipping.

        Returns:
            Tensor of shape ``(batch_size,)`` containing the scalar energy per
            example.
        """
        # x shape: (batch_size, 2, num_filters, seq_length)
        diff = x[:, 0].sum(dim=(1, 2)) - x[:, 1].sum(dim=(1, 2))
        return self.w * diff + self.b

class ResidualTuner(nn.Module):
    """Residual calibration head used after the energy score.

    This module mirrors the original Keras implementation:

    ``Dense(hidden) -> ReLU -> BatchNorm -> Dense(hidden) -> ReLU
    -> BatchNorm -> Dense(1) -> residual add``.

    The input is expected to have a trailing dimension of size ``1`` so the
    residual addition can be applied directly.
    """
    def __init__(self, hidden_units: int = 100, eps: float = 1e-3, momentum: float = 0.99):
        """Initialize the tuner network.

        Args:
            hidden_units: Width of the two hidden linear layers.
            eps: Batch normalization epsilon.
            momentum: Keras-style batch normalization momentum. Internally
                converted to the PyTorch convention.
        """
        super().__init__()
        self.hidden_units = hidden_units

        self.fc1 = nn.Linear(1, hidden_units)          # in_features fixed to 1 to match Dense(?, hidden)
        self.bn1 = nn.BatchNorm1d(hidden_units, eps=eps, momentum=1 - momentum)

        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units, eps=eps, momentum=1 - momentum)

        self.fc3 = nn.Linear(hidden_units, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Run the residual calibration network.

        Args:
            inp: Tensor with shape ``(..., 1)``.

        Returns:
            Tensor with the same shape as ``inp``.

        Raises:
            ValueError: If the last dimension of ``inp`` is not ``1``.
        """
        if inp.shape[-1] != 1:
            raise ValueError(f"ResidualTuner expects last dim == 1, got {inp.shape[-1]}")

        # Flatten to (N, C) for BatchNorm1d, then restore shape
        orig_shape = inp.shape
        x = inp.reshape(-1, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.fc3(x)

        x = x.reshape(orig_shape)
        return x + inp

    @torch.no_grad()
    def load_weights_from_dict(self, weight_dict):
        """Load weights exported from the TensorFlow/Keras tuner.

        Args:
            weight_dict: Mapping containing dense and batch-normalization
                parameters. Expected keys are ``fc1_w``, ``fc1_b``,
                ``bn1_gamma``, ``bn1_beta``, ``bn1_mean``, ``bn1_var``,
                ``fc2_w``, ``fc2_b``, ``bn2_gamma``, ``bn2_beta``,
                ``bn2_mean``, ``bn2_var``, ``fc3_w``, and ``fc3_b``.

        Returns:
            The current module instance.
        """
    
        def _copy(dst, src, transpose=False):
            if transpose:
                src = src.t()
            dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
    
        # ---- Dense 1 ----
        _copy(self.fc1.weight, weight_dict["fc1_w"], transpose=True)
        _copy(self.fc1.bias,   weight_dict["fc1_b"])
    
        # ---- BN 1 ----
        _copy(self.bn1.weight,       weight_dict["bn1_gamma"])  # gamma
        _copy(self.bn1.bias,         weight_dict["bn1_beta"])   # beta
        _copy(self.bn1.running_mean, weight_dict["bn1_mean"])
        _copy(self.bn1.running_var,  weight_dict["bn1_var"])
    
        # ---- Dense 2 ----
        _copy(self.fc2.weight, weight_dict["fc2_w"], transpose=True)
        _copy(self.fc2.bias,   weight_dict["fc2_b"])
    
        # ---- BN 2 ----
        _copy(self.bn2.weight,       weight_dict["bn2_gamma"])
        _copy(self.bn2.bias,         weight_dict["bn2_beta"])
        _copy(self.bn2.running_mean, weight_dict["bn2_mean"])
        _copy(self.bn2.running_var,  weight_dict["bn2_var"])
    
        # ---- Dense 3 ----
        _copy(self.fc3.weight, weight_dict["fc3_w"], transpose=True)
        _copy(self.fc3.bias,   weight_dict["fc3_b"])
    
        return self

class PNASModel(nn.Module):
    """Inference model for exon inclusion prediction from sequence and structure."""

    def __init__(self, input_length=90):
        """Initialize the model architecture.

        Args:
            input_length: Total length of the input window, including flanking
                context. Defaults to ``90``.
        """
        super(PNASModel, self).__init__()
        self.input_length = input_length
        self.seq_kernel_size = 6
        self.struct_kernel_size = 30
        
        ### Sequence layers ###
        # (valid padding) #
        self.conv_skip = nn.Conv1d(in_channels=4, out_channels=20, kernel_size=self.seq_kernel_size, padding=0)
        self.conv_incl = nn.Conv1d(in_channels=4, out_channels=20, kernel_size=self.seq_kernel_size, padding=0)

        # Position bias layers
        conv_out_shape = input_length - self.seq_kernel_size + 1
        self.position_bias_skip = nn.Parameter(torch.zeros(20, conv_out_shape))
        self.position_bias_incl = nn.Parameter(torch.zeros(20, conv_out_shape))

        ### Structure layers ###
        # (same padding) #
        self.conv_struct_skip = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=self.struct_kernel_size, padding='same')
        self.conv_struct_incl = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=self.struct_kernel_size, padding='same')
        self.position_bias_skip_struct = nn.Parameter(torch.zeros(8, input_length))
        self.position_bias_incl_struct = nn.Parameter(torch.zeros(8, input_length))

        ### Aggregation ###
        self.energy_seq_struct = SumDiff()

        ### Activation ###
        self.energy_activation_incl = nn.Softplus()
        self.energy_activation_skip = nn.Softplus()

        ### Tuner ###
        self.tuner = ResidualTuner(hidden_units=4)
        self.output_activation = nn.Sigmoid()

    @torch.no_grad()
    def load_weights_from_dict(self, parameter_dict):
        """Load a parameter dictionary exported outside of PyTorch.

        Args:
            parameter_dict: Mapping containing convolution, bias, aggregation,
                and tuner parameters. The nested ``"tuner"`` key is forwarded to
                :meth:`ResidualTuner.load_weights_from_dict`.

        Returns:
            The current model instance.
        """
        def _to_like(t, ref):
            return t.to(dtype=ref.dtype, device=ref.device)
    
        def _copy_param(dst, src):
            dst.copy_(_to_like(src, dst))
    
        def _load_conv1d(conv: nn.Conv1d, w_key: str, b_key: str):
            w = parameter_dict[w_key]
            b = parameter_dict[b_key]
            _copy_param(conv.weight, w)
            _copy_param(conv.bias, b)
    
        def _load_linear(fc: nn.Linear, w_key: str, b_key: str, tf_kernel: bool = True):
            """
            If tf_kernel=True, assumes src kernel is TF layout (in, out) and transposes to (out, in).
            """
            w = parameter_dict[w_key]
            b = parameter_dict[b_key]
            if tf_kernel:
                w = w.t()
            _copy_param(fc.weight, w)
            _copy_param(fc.bias, b)
    
        def _load_bn(bn: nn.BatchNorm1d, gamma_key: str, beta_key: str, mean_key: str, var_key: str):
            gamma = parameter_dict[gamma_key]
            beta  = parameter_dict[beta_key]
            mean  = parameter_dict[mean_key]
            var   = parameter_dict[var_key]
            bn.weight.copy_(_to_like(gamma, bn.weight))         # gamma
            bn.bias.copy_(_to_like(beta, bn.bias))              # beta
            bn.running_mean.copy_(_to_like(mean, bn.running_mean))
            bn.running_var.copy_(_to_like(var, bn.running_var))
    
        # -------------------------
        # Sequence conv + pos bias
        # -------------------------
        _load_conv1d(self.conv_incl, "conv_incl_w", "conv_incl_b")
        _load_conv1d(self.conv_skip, "conv_skip_w", "conv_skip_b")
        _copy_param(self.position_bias_incl, parameter_dict["position_bias_incl"])
        _copy_param(self.position_bias_skip, parameter_dict["position_bias_skip"])
    
        # -------------------------
        # Structure conv + pos bias
        # -------------------------
        _load_conv1d(self.conv_struct_incl, "conv_struct_incl_w", "conv_struct_incl_b")
        _load_conv1d(self.conv_struct_skip, "conv_struct_skip_w", "conv_struct_skip_b")
        _copy_param(self.position_bias_incl_struct, parameter_dict["position_bias_incl_struct"])
        _copy_param(self.position_bias_skip_struct, parameter_dict["position_bias_skip_struct"])
    
        # -------------------------
        # SumDiff (energy_seq_struct)
        # -------------------------
        _copy_param(self.energy_seq_struct.w, parameter_dict["energy_seq_struct_w"])
        _copy_param(self.energy_seq_struct.b, parameter_dict["energy_seq_struct_b"])

        tuner_params = parameter_dict['tuner']
        self.tuner.load_weights_from_dict(tuner_params)
    
        return self

    def forward(self, x_seq, x_struct, x_wobble):
        """Compute exon inclusion probabilities.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            x_struct: Structure tensor of shape ``(batch_size, 3, input_length)``.
            x_wobble: Wobble tensor of shape ``(batch_size, 1, input_length)``.

        Returns:
            Tensor containing sigmoid-transformed predictions for each example.
            A batch of size one will be returned as a scalar because of the
            final ``squeeze()``.
        """
        # Compute sequence activations - each is (batch_size, num_filters, 85)
        conv_skip_out = self.conv_skip(x_seq) + self.position_bias_skip.unsqueeze(0)  # Add position bias
        conv_incl_out = self.conv_incl(x_seq) + self.position_bias_incl.unsqueeze(0)
        
        # Compute structure activations - each is (batch_size, num_structure_filters, 90)
        struct_input = torch.cat([x_seq, x_struct, x_wobble], dim=1)  # Concatenate along channel dimension
        conv_struct_skip_out = self.conv_struct_skip(struct_input) + self.position_bias_skip_struct.unsqueeze(0)
        conv_struct_incl_out = self.conv_struct_incl(struct_input) + self.position_bias_incl_struct.unsqueeze(0)

        # Crop to match sequence activations
        conv_struct_skip_out = conv_struct_skip_out[:, :, 2:-3]
        conv_struct_incl_out = conv_struct_incl_out[:, :, 2:-3]

        # Concatenated activations
        activations_skip = self.energy_activation_skip(torch.cat([conv_skip_out, conv_struct_skip_out], dim=1))  # (batch_size, 28, 85)
        activations_incl = self.energy_activation_incl(torch.cat([conv_incl_out, conv_struct_incl_out], dim=1))  # (batch_size, 28, 85)

        # Apply sum-difference
        energy_in = torch.stack([activations_incl, activations_skip], dim=1)  # (batch_size, 2, 28, 85)
        energy_out = self.energy_seq_struct(torch.stack([activations_incl, activations_skip], dim=1)).unsqueeze(1)  # (batch_size, 1)

        # Apply tuner
        tuner_out = self.tuner(energy_out)  # (batch_size, 1)
        out = self.output_activation(tuner_out).squeeze()  # (batch_size, 1)

        return out

    def compute_sequence_activations(self, x_seq, agg='mean'):
        """Summarize sequence filter activations for inclusion and skipping.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            agg: Aggregation to apply over the sequence axis. Supported values
                are ``"mean"`` and ``"sum"``.

        Returns:
            A tuple ``(a_incl, a_skip)`` where each tensor has shape
            ``(batch_size, 20)`` after aggregation.

        Raises:
            ValueError: If ``agg`` is not supported.
        """
        conv_skip_out = self.conv_skip(x_seq) + self.position_bias_skip.unsqueeze(0)  # Add position bias
        conv_incl_out = self.conv_incl(x_seq) + self.position_bias_incl.unsqueeze(0)
        a_skip, a_incl = F.softplus(conv_skip_out), F.softplus(conv_incl_out)


        if agg == 'mean':
            a_incl = torch.mean(a_incl, dim=2)
            a_skip = torch.mean(a_skip, dim=2)
        elif agg == 'sum':
            a_incl = torch.sum(a_incl, dim=2)
            a_skip = torch.sum(a_skip, dim=2)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

        return a_incl, a_skip

    def compute_sr_balance(self, x_seq, agg='mean'):
        """Compute the net inclusion-minus-skipping sequence score.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            agg: Aggregation mode passed to
                :meth:`compute_sequence_activations`.

        Returns:
            Tensor of shape ``(batch_size,)`` containing the summed balance per
            example.
        """
        a_incl, a_skip = self.compute_sequence_activations(x_seq, agg)
        return a_incl.sum(dim=1) - a_skip.sum(dim=1)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a PyTorch state dict, resampling position biases when needed.

        This override allows checkpoints trained with a different input length
        to be adapted by Lanczos-resampling the position-bias tensors.

        Args:
            state_dict: Standard PyTorch state dictionary.
            strict: Passed through to ``nn.Module.load_state_dict``.

        Returns:
            The return value of ``nn.Module.load_state_dict``.
        """
        sd = dict(state_dict)  # shallow copy

        F = 10  # fixed flank length
        margin = 5
        
        pad_seq = min(F + margin, (self.input_length - self.seq_kernel_size + 1)//2)  # -> 15
        pad_struct = min(F + (self.struct_kernel_size - 1)//2 + margin, self.input_length//2)  # -> 29
    
        # --- sequence pos bias: shape (20, input_length - seq_kernel + 1)
        if "position_bias_skip" in sd:
            sd["position_bias_skip"] = self._resample_position_bias(
                sd["position_bias_skip"],
                out_len=self.input_length - self.seq_kernel_size + 1,
                padding=pad_seq,
            )
        if "position_bias_incl" in sd:
            sd["position_bias_incl"] = self._resample_position_bias(
                sd["position_bias_incl"],
                out_len=self.input_length - self.seq_kernel_size + 1,
                padding=pad_seq,
            )
    
        # --- structure pos bias: shape (8, input_length)  (NO kernel_size term)
        if "position_bias_skip_struct" in sd:
            sd["position_bias_skip_struct"] = self._resample_position_bias(
                sd["position_bias_skip_struct"],
                out_len=self.input_length,
                padding=pad_struct,
            )
        if "position_bias_incl_struct" in sd:
            sd["position_bias_incl_struct"] = self._resample_position_bias(
                sd["position_bias_incl_struct"],
                out_len=self.input_length,
                padding=pad_struct,
            )
    
        return super().load_state_dict(sd, strict=strict)

    def _resample_position_bias(self, orig_weight: torch.Tensor, out_len: int, padding: int):
        """Resample a position-bias tensor while preserving edge padding.

        Args:
            orig_weight: Tensor of shape ``(channels, old_length)``.
            out_len: Target output length.
            padding: Number of values to preserve at both ends before
                resampling the middle segment.

        Returns:
            Tensor of shape ``(channels, out_len)`` on the same device and with
            the same dtype as ``orig_weight``.
        """
        # Ensure CPU numpy conversion is safe
        w = orig_weight.detach().cpu().numpy()  # (C, L_old)
    
        def resample_one_channel(x):
            # x: (L_old,)
            left = x[:padding]
            mid  = x[padding:-padding]
            right = x[-padding:]
    
            # lanczos_resampling(mid, new_mid_len) must exist and return 1D array
            new_mid_len = out_len - 2 * padding
            new_mid = lanczos_resampling(mid, new_mid_len)
    
            return np.concatenate([left, new_mid, right], axis=0)
    
        new_w = np.apply_along_axis(resample_one_channel, 1, w)  # (C, out_len)
        return torch.from_numpy(new_w).to(dtype=orig_weight.dtype, device=orig_weight.device)
