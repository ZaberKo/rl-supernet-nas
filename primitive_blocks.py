from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



def group_norm_group_count(num_channels: int, channels_per_group: int) -> int:
    if channels_per_group <= 0:
        raise ValueError("channels_per_group must be positive.")
    if num_channels % channels_per_group != 0:
        raise ValueError("num_channels must be divisible by channels_per_group.")
    return num_channels // channels_per_group


class ElasticLayerNorm(nn.LayerNorm):
    def __init__(self, *, super_hidden_size, **kwargs):
        super().__init__(super_hidden_size, **kwargs)
        self.super_hidden_size = super_hidden_size
        self.sample_hidden_size = super_hidden_size

    def set_sample_config(self, *, sample_hidden_size):
        assert sample_hidden_size <= self.super_hidden_size, (
            "sample_hidden_size cannot be larger than super_hidden_size"
        )
        self.sample_hidden_size = sample_hidden_size

    def forward(self, x):
        if self.elementwise_affine:
            weight = self.weight[: self.sample_hidden_size]
            bias = self.bias[: self.sample_hidden_size]
        else:
            weight = bias = None

        return F.layer_norm(
            x,
            (self.sample_hidden_size,),
            weight=weight,
            bias=bias,
            eps=self.eps,
        )

    def get_active_subnet(self):
        if self.elementwise_affine:
            slice_weight = self.weight[: self.sample_hidden_size]
            slice_bias = self.bias[: self.sample_hidden_size]
            device, dtype = slice_weight.device, slice_weight.dtype
        else:
            device, dtype = torch.device("cpu"), torch.float32

        size = self.sample_hidden_size
        sub_layer = nn.LayerNorm(
            size,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
            bias=self.bias is not None,
            device=device,
            dtype=dtype,
        )

        if self.elementwise_affine:
            with torch.no_grad():
                sub_layer.weight.copy_(slice_weight)
                sub_layer.bias.copy_(slice_bias)

        return sub_layer

    @property
    def elastic_num_params(self):
        if self.elementwise_affine:
            return 2 * self.sample_hidden_size
        else:
            return 0


class ElasticLinear(nn.Linear):
    def __init__(self, *, super_in_dim, super_out_dim, **kwargs):
        super().__init__(super_in_dim, super_out_dim, **kwargs)
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.sample_in_dim = super_in_dim
        self.sample_out_dim = super_out_dim

    def set_sample_config(self, *, sample_in_dim, sample_out_dim):
        assert sample_in_dim <= self.super_in_dim, (
            "sample_in_dim cannot be larger than super_in_dim"
        )
        assert sample_out_dim <= self.super_out_dim, (
            "sample_out_dim cannot be larger than super_out_dim"
        )
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

    def forward(self, x):
        weight = self.weight[: self.sample_out_dim, : self.sample_in_dim]
        bias = self.bias[: self.sample_out_dim] if self.bias is not None else None
        return F.linear(x, weight, bias)

    def get_active_subnet(self):
        sub_layer = nn.Linear(
            self.sample_in_dim,
            self.sample_out_dim,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

        with torch.no_grad():
            sub_layer.weight.copy_(
                self.weight[: self.sample_out_dim, : self.sample_in_dim]
            )
            if self.bias is not None:
                sub_layer.bias.copy_(self.bias[: self.sample_out_dim])

        return sub_layer

    @property
    def elastic_num_params(self):
        bias_params = self.sample_out_dim if self.bias is not None else 0
        return self.sample_in_dim * self.sample_out_dim + bias_params


class ElasticConv2d(nn.Conv2d):
    """Elastic Conv2d supporting dynamic input/output channels, groups, and kernel size."""

    def __init__(
        self,
        *,
        super_in_channels: int,
        super_out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        candidate_kernel_sizes: tuple[int, ...] | None = None,
        use_kernel_transform: bool = True,
    ):
        if isinstance(kernel_size, tuple):
            if kernel_size[0] != kernel_size[1]:
                raise ValueError("ElasticConv2d only supports square kernels.")
            kernel_size = kernel_size[0]

        if candidate_kernel_sizes is None:
            candidate_kernel_sizes = (kernel_size,)
        candidate_kernel_sizes = tuple(
            sorted(set(int(k) for k in candidate_kernel_sizes))
        )
        if not candidate_kernel_sizes:
            raise ValueError("candidate_kernel_sizes must not be empty.")

        max_kernel_size = max(candidate_kernel_sizes)
        super().__init__(
            super_in_channels,
            super_out_channels,
            max_kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.super_in_channels = super_in_channels
        self.super_out_channels = super_out_channels
        self.super_groups = groups
        self.candidate_kernel_sizes = candidate_kernel_sizes
        self.max_kernel_size = max_kernel_size
        self.base_padding = self.padding

        self.sample_in_channels = super_in_channels
        self.sample_out_channels = super_out_channels
        self.sample_groups = groups
        self.sample_kernel_size = max_kernel_size
        self.use_kernel_transform = use_kernel_transform

        self._ks_set = list(candidate_kernel_sizes)
        if self.use_kernel_transform and len(self._ks_set) > 1:
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = f"{ks_larger}to{ks_small}_matrix"
                self.register_parameter(param_name, Parameter(torch.eye(ks_small**2)))

        self.active_padding = None
        self.active_groups = None

        self.set_sample_config()

    def set_sample_config(
        self,
        *,
        sample_in_channels: int | None = None,
        sample_out_channels: int | None = None,
        sample_groups: int | None = None,
        sample_kernel_size: int | None = None,
    ):
        if sample_in_channels is not None:
            self.sample_in_channels = sample_in_channels
        if sample_out_channels is not None:
            self.sample_out_channels = sample_out_channels
        if sample_groups is not None:
            self.sample_groups = sample_groups
        if sample_kernel_size is not None:
            if sample_kernel_size not in self.candidate_kernel_sizes:
                raise ValueError(f"Unsupported kernel size: {sample_kernel_size}")
            self.sample_kernel_size = sample_kernel_size

        if self.base_padding == (self.max_kernel_size // 2, self.max_kernel_size // 2):
            self.active_padding = (
                self.sample_kernel_size // 2,
                self.sample_kernel_size // 2,
            )
        else:
            self.active_padding = self.base_padding

        self.active_groups = self.sample_groups
        if self.super_groups == self.super_in_channels:
            self.active_groups = self.sample_in_channels

    def _get_active_weights(self):
        if self.super_groups == self.super_in_channels:
            active_in_per_group = 1
        else:
            active_in_per_group = self.sample_in_channels // self.active_groups

        filters = self.weight[: self.sample_out_channels, :active_in_per_group, :, :]
        if self.sample_kernel_size == self.max_kernel_size:
            active_weight = filters
        else:
            if not self.use_kernel_transform:
                center = self.max_kernel_size // 2
                dev = self.sample_kernel_size // 2
                start, end = center - dev, center + dev + 1
                active_weight = filters[:, :, start:end, start:end]
            else:
                start_filter = filters
                for i in range(len(self._ks_set) - 1, 0, -1):
                    src_ks = self._ks_set[i]
                    if src_ks <= self.sample_kernel_size:
                        break
                    target_ks = self._ks_set[i - 1]
                    center = src_ks // 2
                    dev = target_ks // 2
                    start, end = center - dev, center + dev + 1
                    cropped = start_filter[:, :, start:end, start:end].contiguous()
                    cropped = cropped.view(cropped.size(0), cropped.size(1), -1)
                    cropped = cropped.view(-1, cropped.size(2))
                    cropped = F.linear(
                        cropped, getattr(self, f"{src_ks}to{target_ks}_matrix")
                    )
                    cropped = cropped.view(
                        self.sample_out_channels,
                        active_in_per_group,
                        target_ks,
                        target_ks,
                    )
                    start_filter = cropped
                active_weight = start_filter

        active_bias = (
            self.bias[: self.sample_out_channels] if self.bias is not None else None
        )
        return active_weight, active_bias

    def forward(self, x):
        weight, bias = self._get_active_weights()
        return F.conv2d(
            x,
            weight.contiguous(),
            bias,
            self.stride,
            self.active_padding,
            self.dilation,
            self.active_groups,
        )

    def get_active_subnet(self):
        weight, bias = self._get_active_weights()
        sub = nn.Conv2d(
            self.sample_in_channels,
            self.sample_out_channels,
            self.sample_kernel_size,
            self.stride,
            self.active_padding,
            self.dilation,
            self.active_groups,
            self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

        with torch.no_grad():
            sub.weight.copy_(weight)
            if self.bias is not None:
                sub.bias.copy_(bias)
        return sub

    @property
    def elastic_num_params(self):
        w_params = (
            self.sample_out_channels
            * (self.sample_in_channels // self.active_groups)
            * self.sample_kernel_size
            * self.sample_kernel_size
        )
        b_params = self.sample_out_channels if self.bias is not None else 0
        return w_params + b_params


class ElasticBatchNorm2d(nn.BatchNorm2d):
    """Elastic BatchNorm2d supporting dynamic number of features."""

    def __init__(
        self,
        *,
        super_num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__(super_num_features, eps, momentum, affine, track_running_stats)
        self.super_num_features = super_num_features
        self.sample_num_features = super_num_features

    def set_sample_config(self, *, sample_num_features: int):
        self.sample_num_features = sample_num_features

    def forward(self, x):
        feature_dim = x.size(1)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        return F.batch_norm(
            x,
            self.running_mean[:feature_dim] if self.running_mean is not None else None,
            self.running_var[:feature_dim] if self.running_var is not None else None,
            self.weight[:feature_dim] if self.affine else None,
            self.bias[:feature_dim] if self.affine else None,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

    def get_active_subnet(self):
        sub = nn.BatchNorm2d(
            self.sample_num_features,
            self.eps,
            self.momentum,
            self.affine,
            self.track_running_stats,
        )
        with torch.no_grad():
            sub.weight.copy_(self.weight[: self.sample_num_features])
            sub.bias.copy_(self.bias[: self.sample_num_features])
            sub.running_mean.copy_(self.running_mean[: self.sample_num_features])
            sub.running_var.copy_(self.running_var[: self.sample_num_features])
        return sub

    @property
    def elastic_num_params(self):
        return 2 * self.sample_num_features if self.affine else 0


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel axis for NCHW feature maps."""

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.norm = nn.LayerNorm(
            num_channels,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("LayerNorm2d expects an NCHW tensor.")
        if x.size(1) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, but got {x.size(1)}."
            )
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ElasticLayerNorm2d(nn.Module):
    """Elastic LayerNorm2d supporting dynamic channel width for NCHW tensors."""

    def __init__(
        self,
        *,
        super_num_channels: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.super_num_channels = super_num_channels
        self.sample_num_channels = super_num_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(super_num_channels))
            if bias:
                self.bias = nn.Parameter(torch.zeros(super_num_channels))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def set_sample_config(self, *, sample_num_channels: int):
        if sample_num_channels > self.super_num_channels:
            raise ValueError("sample_num_channels cannot exceed super_num_channels.")
        self.sample_num_channels = sample_num_channels

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("ElasticLayerNorm2d expects an NCHW tensor.")
        feature_dim = x.size(1)
        if feature_dim > self.super_num_channels:
            raise ValueError("Input channels cannot exceed super_num_channels.")

        if self.elementwise_affine:
            weight = self.weight[:feature_dim]
            bias = self.bias[:feature_dim] if self.bias is not None else None
        else:
            weight = bias = None

        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (feature_dim,), weight=weight, bias=bias, eps=self.eps)
        return x.permute(0, 3, 1, 2).contiguous()

    def get_active_subnet(self):
        if self.elementwise_affine:
            slice_weight = self.weight[: self.sample_num_channels]
            slice_bias = (
                self.bias[: self.sample_num_channels]
                if self.bias is not None
                else None
            )
            device, dtype = slice_weight.device, slice_weight.dtype
        else:
            slice_weight = slice_bias = None
            device, dtype = torch.device("cpu"), torch.float32

        sub = LayerNorm2d(
            self.sample_num_channels,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
            bias=self.bias is not None,
            device=device,
            dtype=dtype,
        )

        if self.elementwise_affine:
            with torch.no_grad():
                sub.norm.weight.copy_(slice_weight)
                if slice_bias is not None:
                    sub.norm.bias.copy_(slice_bias)

        return sub

    @property
    def elastic_num_params(self):
        if not self.elementwise_affine:
            return 0
        bias_params = self.sample_num_channels if self.bias is not None else 0
        return self.sample_num_channels + bias_params


class GroupNorm2d(nn.Module):
    """GroupNorm for NCHW feature maps with a LayerNorm2d-compatible API."""

    def __init__(
        self,
        num_channels: int,
        channels_per_group: int = 16,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.channels_per_group = channels_per_group
        self.num_groups = group_norm_group_count(num_channels, channels_per_group)
        self.norm = nn.GroupNorm(
            self.num_groups,
            num_channels,
            eps=eps,
            affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )
        if elementwise_affine and not bias:
            self.norm.bias = None

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("GroupNorm2d expects an NCHW tensor.")
        if x.size(1) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, but got {x.size(1)}."
            )
        return self.norm(x)


class ElasticGroupNorm2d(nn.Module):
    """Elastic GroupNorm2d supporting dynamic channel width for NCHW tensors."""

    def __init__(
        self,
        *,
        super_num_channels: int,
        channels_per_group: int = 16,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.super_num_channels = super_num_channels
        self.sample_num_channels = super_num_channels
        self.channels_per_group = channels_per_group
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        group_norm_group_count(super_num_channels, channels_per_group)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(super_num_channels))
            if bias:
                self.bias = nn.Parameter(torch.zeros(super_num_channels))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def set_sample_config(self, *, sample_num_channels: int):
        if sample_num_channels > self.super_num_channels:
            raise ValueError("sample_num_channels cannot exceed super_num_channels.")
        group_norm_group_count(sample_num_channels, self.channels_per_group)
        self.sample_num_channels = sample_num_channels

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("ElasticGroupNorm2d expects an NCHW tensor.")
        feature_dim = x.size(1)
        if feature_dim > self.super_num_channels:
            raise ValueError("Input channels cannot exceed super_num_channels.")
        num_groups = group_norm_group_count(feature_dim, self.channels_per_group)

        if self.elementwise_affine:
            weight = self.weight[:feature_dim]
            bias = self.bias[:feature_dim] if self.bias is not None else None
        else:
            weight = bias = None

        return F.group_norm(x, num_groups, weight=weight, bias=bias, eps=self.eps)

    def get_active_subnet(self):
        if self.elementwise_affine:
            slice_weight = self.weight[: self.sample_num_channels]
            slice_bias = (
                self.bias[: self.sample_num_channels]
                if self.bias is not None
                else None
            )
            device, dtype = slice_weight.device, slice_weight.dtype
        else:
            slice_weight = slice_bias = None
            device, dtype = torch.device("cpu"), torch.float32

        sub = GroupNorm2d(
            self.sample_num_channels,
            channels_per_group=self.channels_per_group,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
            bias=self.bias is not None,
            device=device,
            dtype=dtype,
        )

        if self.elementwise_affine:
            with torch.no_grad():
                sub.norm.weight.copy_(slice_weight)
                if slice_bias is not None:
                    sub.norm.bias.copy_(slice_bias)

        return sub

    @property
    def elastic_num_params(self):
        if not self.elementwise_affine:
            return 0
        bias_params = self.sample_num_channels if self.bias is not None else 0
        return self.sample_num_channels + bias_params


class ElasticConv1d(nn.Conv1d):
    """Elastic Conv1d supporting dynamic input/output channels and groups."""

    def __init__(
        self,
        *,
        super_in_channels: int,
        super_out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        super_groups: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            super_in_channels,
            super_out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            super_groups,
            bias,
        )
        self.super_in_channels = super_in_channels
        self.super_out_channels = super_out_channels
        self.super_groups = super_groups

        self.active_in_channels = super_in_channels
        self.active_out_channels = super_out_channels
        self.active_groups = super_groups

    def set_sample_config(
        self,
        *,
        sample_in_channels: int | None = None,
        sample_out_channels: int | None = None,
        sample_groups: int | None = None,
    ):
        if sample_in_channels is not None:
            self.active_in_channels = sample_in_channels
        if sample_out_channels is not None:
            self.active_out_channels = sample_out_channels
        if sample_groups is not None:
            self.active_groups = sample_groups

    def _get_active_weight(self) -> torch.Tensor:
        if self.super_groups == self.super_in_channels:
            return self.weight[: self.active_out_channels, :1, :]

        active_in_per_group = self.active_in_channels // self.active_groups
        return self.weight[: self.active_out_channels, :active_in_per_group, :]

    def forward(self, x):
        groups = self.active_in_channels if self.super_groups == self.super_in_channels else self.active_groups
        weight = self._get_active_weight()
        bias = self.bias[: self.active_out_channels] if self.bias is not None else None
        return F.conv1d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            groups,
        )

    def get_active_subnet(self):
        groups = self.active_in_channels if self.super_groups == self.super_in_channels else self.active_groups
        sub = nn.Conv1d(
            self.active_in_channels,
            self.active_out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            groups,
            self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

        with torch.no_grad():
            sub.weight.copy_(self._get_active_weight())
            if self.bias is not None:
                sub.bias.copy_(self.bias[: self.active_out_channels])

        return sub

    @property
    def elastic_num_params(self):
        groups = self.active_in_channels if self.super_groups == self.super_in_channels else self.active_groups
        weight_params = (
            self.active_out_channels
            * (self.active_in_channels // groups)
            * self.weight.shape[2]
        )
        bias_params = self.active_out_channels if self.bias is not None else 0
        return weight_params + bias_params


class ElasticEmbedding(nn.Embedding):
    """Elastic Embedding layer that supports dynamic embedding dimension."""

    def __init__(self, *, num_embeddings: int, super_embedding_dim: int, **kwargs):
        """Initializes the ElasticEmbedding.

        Args:
            num_embeddings: The number of embeddings.
            super_embedding_dim: The maximum embedding dimension.
            **kwargs: Additional arguments for nn.Embedding.
        """
        super().__init__(num_embeddings, super_embedding_dim, **kwargs)
        self.super_embedding_dim = super_embedding_dim
        self.sample_embedding_dim = super_embedding_dim

    def set_sample_config(self, *, sample_embedding_dim: int):
        """Sets the sampling configuration.

        Args:
            sample_embedding_dim: The sampled embedding dimension.
        """
        assert sample_embedding_dim <= self.super_embedding_dim, (
            "sample_embedding_dim cannot be larger than super_embedding_dim"
        )
        self.sample_embedding_dim = sample_embedding_dim

    def forward(self, x):
        weight = self.weight[:, : self.sample_embedding_dim]
        return F.embedding(
            x,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def get_active_subnet(self):
        """Gets the active subnet.

        Returns:
            nn.Embedding: A standard Embedding layer with the sampled configuration and weights.
        """
        sub_layer = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.sample_embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

        with torch.no_grad():
            sub_layer.weight.copy_(self.weight[:, : self.sample_embedding_dim])

        return sub_layer

    @property
    def elastic_num_params(self):
        """Returns the number of active parameters."""
        return self.num_embeddings * self.sample_embedding_dim


class QKVProjector(nn.Module):
    """A standard QKV Projector containing separated q, k, v linear projections."""

    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)


class ElasticQKVProjector(nn.Module):
    """An elastic QKV projector that encapsulates separate Q, K, and V elastic linear projections
    to ensure proper tensor slicing behavior while providing a unified API.
    """

    def __init__(self, *, super_in_dim: int, super_out_dim: int, bias: bool = True):
        super().__init__()
        self.q_proj = ElasticLinear(super_in_dim=super_in_dim, super_out_dim=super_out_dim, bias=bias)
        self.k_proj = ElasticLinear(super_in_dim=super_in_dim, super_out_dim=super_out_dim, bias=bias)
        self.v_proj = ElasticLinear(super_in_dim=super_in_dim, super_out_dim=super_out_dim, bias=bias)

    def set_sample_config(self, *, sample_in_dim: int, sample_out_dim: int):
        self.q_proj.set_sample_config(sample_in_dim=sample_in_dim, sample_out_dim=sample_out_dim)
        self.k_proj.set_sample_config(sample_in_dim=sample_in_dim, sample_out_dim=sample_out_dim)
        self.v_proj.set_sample_config(sample_in_dim=sample_in_dim, sample_out_dim=sample_out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def get_active_subnet(self) -> nn.Module:
        return QKVProjector(
            self.q_proj.get_active_subnet(),
            self.k_proj.get_active_subnet(),
            self.v_proj.get_active_subnet(),
        )

    @property
    def elastic_num_params(self):
        return (
            self.q_proj.elastic_num_params
            + self.k_proj.elastic_num_params
            + self.v_proj.elastic_num_params
        )


class MHSAQKVProjector(nn.Module):
    """A standard MHSA QKV Projector that automatically splits the channel dimension into heads
    and transposes the sequence and head dimensions.
    """

    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, num_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(-2, -3)
        k = self.k_proj(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(-2, -3)
        v = self.v_proj(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(-2, -3)
        return q, k, v


class ElasticMHSAQKVProjector(nn.Module):
    """An elastic MHSA QKV projector that encapsulates separate Q, K, and V elastic linear projections
    and automatically splits the channel dimension into heads and transposes them during forward pass.
    """

    def __init__(self, *, super_in_dim: int, super_out_dim: int, head_dim: int, bias: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.sample_num_heads = super_out_dim // head_dim
        self.q_proj = ElasticLinear(super_in_dim=super_in_dim, super_out_dim=super_out_dim, bias=bias)
        self.k_proj = ElasticLinear(super_in_dim=super_in_dim, super_out_dim=super_out_dim, bias=bias)
        self.v_proj = ElasticLinear(super_in_dim=super_in_dim, super_out_dim=super_out_dim, bias=bias)

    def set_sample_config(self, *, sample_in_dim: int, sample_out_dim: int):
        self.sample_num_heads = sample_out_dim // self.head_dim
        self.q_proj.set_sample_config(sample_in_dim=sample_in_dim, sample_out_dim=sample_out_dim)
        self.k_proj.set_sample_config(sample_in_dim=sample_in_dim, sample_out_dim=sample_out_dim)
        self.v_proj.set_sample_config(sample_in_dim=sample_in_dim, sample_out_dim=sample_out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x).unflatten(-1, (self.sample_num_heads, self.head_dim)).transpose(-2, -3)
        k = self.k_proj(x).unflatten(-1, (self.sample_num_heads, self.head_dim)).transpose(-2, -3)
        v = self.v_proj(x).unflatten(-1, (self.sample_num_heads, self.head_dim)).transpose(-2, -3)
        return q, k, v

    def get_active_subnet(self) -> nn.Module:
        return MHSAQKVProjector(
            self.q_proj.get_active_subnet(),
            self.k_proj.get_active_subnet(),
            self.v_proj.get_active_subnet(),
            self.sample_num_heads,
            self.head_dim,
        )

    @property
    def elastic_num_params(self):
        return (
            self.q_proj.elastic_num_params
            + self.k_proj.elastic_num_params
            + self.v_proj.elastic_num_params
        )
