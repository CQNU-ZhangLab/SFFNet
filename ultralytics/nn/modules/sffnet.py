"""SFFNet modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .block import C2f
from .conv import Conv

__all__ = (
    "LDConv",
    "MDDC",
    "WPM",
)



class LDConv(nn.Module):
    """Linear deformable convolution from the official LDConv implementation."""

    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU(),
        )
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # Clip p.
        p = torch.cat(
            [torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1
        )

        # Bilinear kernel (b, h, w, N).
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # Resample the features using the four neighboring coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        return self.conv(x_offset)

    def _get_p_n(self, N):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(torch.arange(0, row_number), torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number),
            )
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        return torch.cat([p_n_x, p_n_y], 0).view(1, 2 * N, 1, 1)

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        return torch.cat([p_0_x, p_0_y], 1).type(dtype)

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_0 = self._get_p_0(h, w, N, dtype)
        return p_0 + self.p_n + offset

    @staticmethod
    def _get_x_q(x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        return x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        return rearrange(x_offset, "b c h w n -> b c (h n) w")


class WPMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        spatial_extent = 31
        spatial_padding = spatial_extent // 2
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)

        depthwise_specs = (
            ("dw_1x31", (1, spatial_extent), (0, spatial_padding)),
            ("dw_31x1", (spatial_extent, 1), (spatial_padding, 0)),
            ("dw_31x31", spatial_extent, spatial_padding),
            ("dw_1x1", 1, 0),
        )
        for name, kernel_size, padding in depthwise_specs:
            setattr(self, name, nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim))
        self.act = nn.ReLU()

    def forward(self, x):
        spatial_features = self.in_conv(x)
        spatial_features = (
            x
            + self.dw_1x31(spatial_features)
            + self.dw_31x1(spatial_features)
            + self.dw_31x31(spatial_features)
            + self.dw_1x1(spatial_features)
        )
        return self.out_conv(self.act(spatial_features))


class WPM(nn.Module):
    """Wide-area perception module."""

    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, k=1)
        self.cv2 = Conv(dim, dim, k=1)
        self.m = WPMBlock(dim=int(dim * e))

    def forward(self, x):
        projected_features = self.cv1(x)
        channel_partition = [int(x.size(1) * self.e), int(x.size(1) * (1 - self.e))]
        perception_input, bypass = torch.split(projected_features, channel_partition, dim=1)
        return self.cv2(torch.cat((self.m(perception_input), bypass), dim=1))


class DEIE(nn.Module):
    """Dual-domain edge information extraction module."""

    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.weight_x = nn.Parameter(torch.ones(1))
        self.weight_edge = nn.Parameter(torch.ones(1))
        self.weight_enhanced_edge = nn.Parameter(torch.zeros(1))

    @staticmethod
    def fft2d(x):
        return torch.fft.fft2(x)

    @staticmethod
    def ifft2d(x):
        return torch.fft.ifft2(x)

    @staticmethod
    def high_pass_filter(x_fft, threshold=0.1):
        amplitude = torch.abs(x_fft)
        amplitude[amplitude < threshold] = 0
        return amplitude * torch.exp(1j * torch.angle(x_fft))

    @staticmethod
    def adaptive_frequency_enhancement(x_fft, edge_strength_map, factor=1.5):
        amplitude = torch.abs(x_fft)
        modulated_amplitude = amplitude * (1 + factor * edge_strength_map)
        return modulated_amplitude * torch.exp(1j * torch.angle(x_fft))

    @staticmethod
    def frequency_sharpening(x_fft, sharpness_factor=1.2):
        amplitude = torch.abs(x_fft)
        scaled_amplitude = amplitude * sharpness_factor
        return scaled_amplitude * torch.exp(1j * torch.angle(x_fft))

    @staticmethod
    def local_contrast(x, kernel_size=3):
        neighborhood_mean = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)
        return torch.abs(x - neighborhood_mean)

    def forward(self, x):
        neighborhood_mean = self.pool(x)
        spatial_edges = self.out_conv(x - neighborhood_mean)

        input_spectrum = self.fft2d(x)
        filtered_spectrum = self.high_pass_filter(input_spectrum, threshold=0.1)
        contrast_map = self.local_contrast(spatial_edges, kernel_size=3)
        modulated_spectrum = self.adaptive_frequency_enhancement(filtered_spectrum, contrast_map, factor=1.5)
        sharpened_spectrum = self.frequency_sharpening(modulated_spectrum, sharpness_factor=1.2)
        frequency_edges = self.ifft2d(sharpened_spectrum)

        input_term = self.weight_x * x
        spatial_term = self.weight_edge * spatial_edges
        frequency_term = self.weight_enhanced_edge * frequency_edges.real
        return input_term + spatial_term + frequency_term


class MDDCBlock(nn.Module):
    """Multi-scale dynamic dual-domain coupling block."""

    def __init__(self, inc, bins):
        super().__init__()
        channels_per_scale = inc // len(bins)

        self.features = nn.ModuleList(
            self._make_scale_branch(inc, channels_per_scale, pool_size) for pool_size in bins
        )
        self.scale_weights = nn.ParameterList(
            nn.Parameter(torch.full((1, channels_per_scale, 1, 1), float(pool_size in {3, 6})))
            for pool_size in bins
        )
        self.ees = nn.ModuleList(DEIE(in_dim=channels_per_scale) for _ in bins)
        self.local_conv = Conv(inc, inc, k=3)
        self.final_conv = Conv(inc * 2, inc)

    @staticmethod
    def _make_scale_branch(in_channels, branch_channels, pool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            Conv(in_channels, branch_channels, k=1),
            Conv(branch_channels, branch_channels, k=3, g=branch_channels),
        )

    def forward(self, x):
        branch_outputs = [self.local_conv(x)]
        for scale_branch, edge_extractor, scale_weight in zip(self.features, self.ees, self.scale_weights):
            resized_features = F.interpolate(scale_branch(x), size=x.shape[2:], mode="bilinear", align_corners=True)
            branch_outputs.append(edge_extractor(resized_features) * scale_weight)
        return self.final_conv(torch.cat(branch_outputs, dim=1))


class MDDC(C2f):
    """C2f block with multi-scale dynamic dual-domain coupling."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):

        super().__init__(c1, c2, n, shortcut, g, e)
        pooling_scales = [3, 6, 9, 12]
        self.m = nn.ModuleList(MDDCBlock(inc=self.c, bins=pooling_scales) for _ in range(n))
