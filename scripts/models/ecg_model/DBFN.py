import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ECABlock1D(nn.Module):
    def __init__(self, b: int = 1, gamma: int = 2):
        super().__init__()
        self.b = b
        self.gamma = gamma
        self.convs = nn.ModuleDict()

    @staticmethod
    def _ks(c: int, b: int, gamma: int) -> int:
        k = int(abs((math.log2(c) + b) / gamma))
        return k if k % 2 == 1 else k + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        k = self._ks(C, self.b, self.gamma)
        key = f"{C}_{k}"

        if key not in self.convs:
            conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
            conv = conv.to(device=x.device, dtype=x.dtype)
            self.convs[key] = conv
        conv = self.convs[key]

        if conv.weight.dtype != x.dtype or conv.weight.device != x.device:
            self.convs[key] = conv.to(device=x.device, dtype=x.dtype)
            conv = self.convs[key]

        gap = x.mean(dim=-1, keepdim=True)
        attn = torch.sigmoid(conv(gap.transpose(1, 2)))
        attn = attn.transpose(1, 2)
        return x * attn

class TemporalASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        assert out_ch % len(dilations) == 0
        per = out_ch // len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, per, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(per),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = torch.cat([b(x) for b in self.branches], dim=1)
        return self.proj(y)


class FiLM2d(nn.Module):
    """Condition feature channels on age and sex metadata."""
    def __init__(self, age_in: int, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(age_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * channels)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor, age: Tensor) -> Tensor:
        B, C, T = x.shape
        gb = self.net(age)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class Bottleneck1DPreAct(nn.Module):
    expansion = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        use_eca: bool = True,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer=nn.BatchNorm1d,
        drop_path: float = 0.0,
    ):
        super().__init__()
        width = int(planes * (width_per_group / 64.)) * groups

        self.bn1 = norm_layer(inplanes)
        self.conv1 = nn.Conv1d(inplanes, width, kernel_size=1, bias=False)

        self.bn2 = norm_layer(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, groups=groups, bias=False)

        self.bn3 = norm_layer(width)
        self.conv3 = nn.Conv1d(width, planes * self.expansion, kernel_size=1, bias=False)

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            )

        self.eca = ECABlock1D() if use_eca else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)

        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)

        out = F.relu(self.bn3(out), inplace=True)
        out = self.conv3(out)

        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = identity + self.drop_path(out)
        return out

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class XResNet1D101_HiRes(nn.Module):
    """High-resolution 1D backbone with metadata conditioning."""
    def __init__(
        self,
        c_out: int = 256,
        stem_channels: int = 32,
        base_planes: int = 32,
        block_layers: Sequence[int] = (3, 4, 23, 3),
        use_eca: bool = True,
        age_in: int = 2,
        film_hidden: int = 64,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        Block = Bottleneck1DPreAct
        self.inplanes = stem_channels

        self.stem = nn.Sequential(
            nn.Conv1d(12, stem_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        planes_list = [base_planes, base_planes*2, base_planes*4, base_planes*4]
        strides     = [2, 1, 1, 1]
        dilations   = [1, 2, 4, 8]

        total_blocks = sum(block_layers)
        dp_idx = 0

        self.stage1 = self._make_layer(Block, planes_list[0], block_layers[0],
                                       stride=strides[0], dilation=dilations[0],
                                       use_eca=use_eca, drop_path_rate=drop_path_rate,
                                       dp_idx_start=dp_idx, total_blocks=total_blocks)
        dp_idx += block_layers[0]

        self.stage2 = self._make_layer(Block, planes_list[1], block_layers[1],
                                       stride=strides[1], dilation=dilations[1],
                                       use_eca=use_eca, drop_path_rate=drop_path_rate,
                                       dp_idx_start=dp_idx, total_blocks=total_blocks)
        dp_idx += block_layers[1]

        self.stage3 = self._make_layer(Block, planes_list[2], block_layers[2],
                                       stride=strides[2], dilation=dilations[2],
                                       use_eca=use_eca, drop_path_rate=drop_path_rate,
                                       dp_idx_start=dp_idx, total_blocks=total_blocks)
        dp_idx += block_layers[2]

        self.stage4 = self._make_layer(Block, planes_list[3], block_layers[3],
                                       stride=strides[3], dilation=dilations[3],
                                       use_eca=use_eca, drop_path_rate=drop_path_rate,
                                       dp_idx_start=dp_idx, total_blocks=total_blocks)

        self.proj_out = nn.Sequential(
            nn.Conv1d(planes_list[3] * Block.expansion, c_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

        self.aspp = TemporalASPP(in_ch=planes_list[3] * Block.expansion, out_ch=planes_list[3] * Block.expansion,
                                 dilations=(1, 2, 4, 8))

        self.film1 = FiLM2d(age_in=age_in, channels=planes_list[0]*Block.expansion, hidden=film_hidden)
        self.film2 = FiLM2d(age_in=age_in, channels=planes_list[1]*Block.expansion, hidden=film_hidden)
        self.film3 = FiLM2d(age_in=age_in, channels=planes_list[2]*Block.expansion, hidden=film_hidden)
        self.film4 = FiLM2d(age_in=age_in, channels=planes_list[3]*Block.expansion, hidden=film_hidden)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int,
        dilation: int,
        use_eca: bool,
        drop_path_rate: float,
        dp_idx_start: int,
        total_blocks: int,
        norm_layer=nn.BatchNorm1d,
    ) -> nn.Sequential:
        layers = []
        inplanes = self.inplanes
        for i in range(blocks):
            s = stride if i == 0 else 1
            d = dilation
            dp = drop_path_rate * (dp_idx_start + i) / (total_blocks - 1) if drop_path_rate > 0 and total_blocks > 1 else 0.0
            layers.append(block(inplanes, planes, stride=s, dilation=d, use_eca=use_eca, norm_layer=norm_layer, drop_path=dp))
            inplanes = planes * block.expansion
            self.inplanes = inplanes
        return nn.Sequential(*layers)

    def forward(self, x12: Tensor, age: Tensor) -> Tensor:
        x = self.stem(x12)

        x = self.stage1(x)
        x = self.film1(x, age)

        x = self.stage2(x)
        x = self.film2(x, age)

        x = self.stage3(x)
        x = self.film3(x, age)

        x = self.stage4(x)
        x = self.film4(x, age)

        x = self.proj_out(x)
        return x


class BottleneckBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int = 32,
        bottleneck_ch: int = 64,
        dilation: int = 2,
        dropout: float = 0.0,
        use_eca: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, bottleneck_ch, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(bottleneck_ch)

        self.conv2 = nn.Conv1d(
            bottleneck_ch, bottleneck_ch,
            kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.bn2   = nn.BatchNorm1d(bottleneck_ch)

        self.conv3 = nn.Conv1d(bottleneck_ch, in_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm1d(in_ch)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.eca = ECABlock1D() if use_eca else nn.Identity()

        self.gamma = nn.Parameter(torch.ones(1, in_ch, 1) * 1.0)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)

        out = self.eca(out)
        out = identity + self.gamma * out
        return F.relu(out, inplace=True)

class FeatureNetworkHiResDeep(nn.Module):
    """Single-lead high-resolution feature branch."""
    def __init__(
        self,
        age_dim: int = 32,
        num_blocks: int = 8,
        dilations: Optional[Sequence[int]] = None,
        bottleneck_ch: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(16)
        self.pool  = nn.MaxPool1d(2, 2)

        self.block1 = BottleneckBlock1D(in_ch=16, bottleneck_ch=32, dilation=2)
        self.pool1  = nn.MaxPool1d(2, 2)

        self.proj32 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        if dilations is None:
            base = [2, 4, 6, 8, 10, 12, 14, 16]
            dilations = base[:num_blocks] if num_blocks <= len(base) else (base * ((num_blocks + len(base)-1)//len(base)))[:num_blocks]
        self.blocks = nn.ModuleList([
            BottleneckBlock1D(
                in_ch=32,
                bottleneck_ch=bottleneck_ch,
                dilation=d,
                dropout=dropout,
                use_eca=True,
            ) for d in dilations
        ])

        self.age_mlp = nn.Sequential(
            nn.Linear(2, age_dim),
            nn.ReLU(inplace=True)
        )
        self.age_dim = age_dim

    def forward(self, x1: Tensor, age: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x1)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.proj32(x)

        for blk in self.blocks:
            x = blk(x)

        B, C, T = x.shape
        age_feat = self.age_mlp(age).unsqueeze(-1).expand(B, self.age_dim, T)

        return torch.cat([x, age_feat], dim=1)


class ECGBackboneForXAttn(nn.Module):
    """Twelve-lead ECG backbone used by the query decoder."""
    def __init__(self, use_ecgNet_Diagnosis="", num_classes=2):
        super().__init__()
        self.branches = nn.ModuleList([FeatureNetworkHiResDeep(age_dim=32) for _ in range(12)])
        self.branch_gate = nn.Linear(64, 1, bias=False)

        self.full = XResNet1D101_HiRes(
                    c_out=256,
                    stem_channels=32,
                    base_planes=32,
                    block_layers=(3,4,23,3),
                    use_eca=True,
                    age_in=2,
                    film_hidden=64,
                    drop_path_rate=0.1
                )

        self.fuse = nn.Sequential(
            nn.Conv1d(768 + 256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.use_ecgNet_Diagnosis = use_ecgNet_Diagnosis
        if use_ecgNet_Diagnosis == 'ecgNet':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, num_classes)
            )

    def forward(self, x12: Tensor, age: Tensor) -> Dict[str, Tensor]:
        branch_feats = []
        for i in range(12):
            fi = self.branches[i](x12[:, i:i+1, :], age)
            gate = torch.sigmoid(self.branch_gate(fi.mean(dim=-1)))
            fi = fi * gate.unsqueeze(-1)
            branch_feats.append(fi)
        merge = torch.cat(branch_feats, dim=1)

        full = self.full(x12, age)

        fused = self.fuse(torch.cat([full, merge], dim=1))
        if self.use_ecgNet_Diagnosis == 'ecgNet':
            return self.head(fused)
        return fused
