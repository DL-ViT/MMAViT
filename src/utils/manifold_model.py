import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init

from src.utils.lds import grassmanian_point
from src.utils.riemmanian_model import cov_frobenius_norm
from .stochastic_depth import DropPath


class RiemGrassAtt(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, sequence_length=-1,
                 ln_attention=False):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // self.num_heads

        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.riem_scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.grassman_scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.qkv = Linear(dim, dim * 3, bias=True)

        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)
        self.sequence_len = sequence_length

        if ln_attention:
            self.conv_attn = nn.Sequential(
                nn.LayerNorm((2 * num_heads, sequence_length, sequence_length)),
                nn.Conv2d(in_channels=2 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1))
            )
        else:
            self.conv_attn = nn.Sequential(

                nn.Conv2d(in_channels=2 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1))
            )

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # old_shape = q.shape

        qgr, _ = grassmanian_point(q)
        kgr, _ = grassmanian_point(k)

        qgr = qgr  # .permute(0, 1, 3, 2)
        kgr = kgr.permute(0, 1, 3, 2)

        dots = torch.matmul(qgr, kgr).unsqueeze(2)

        attn_grassmman = torch.linalg.norm(dots, dim=2) ** 2. * self.grassman_scale

        attn_riemmanian = attn_grassmman  # cov_frobenius_norm(q, k) * self.riem_scalee

        attn_ = self.conv_attn(torch.cat((attn_riemmanian, attn_grassmman), dim=1)).softmax(
            dim=-1)

        out = torch.matmul(self.attn_drop(attn_), v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class EuclRiemGrassAtt(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attention_dropout=0.1, projection_dropout=0.1,
                 sequence_length=1,
                 ln_attention=True, return_map=False):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.riem_scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.grassman_scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)
        self.sequence_len = sequence_length
        #        self.attn_matrix = nn.Parameter(torch.randn(sequence_length, sequence_length))
        if ln_attention:
            self.conv_attn = nn.Sequential(
                nn.LayerNorm((3 * num_heads, sequence_length, sequence_length)),
                nn.Conv2d(in_channels=3 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1)),

            )
        else:
            self.conv_attn = nn.Sequential(

                nn.Conv2d(in_channels=3 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1))
            )
        self.return_map = return_map

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # old_shape = q.shape

        qgr, _ = grassmanian_point(q)
        kgr, _ = grassmanian_point(k)


        kgr = kgr.permute(0, 1, 3, 2)

        dots = torch.matmul(qgr, kgr).unsqueeze(2)

        attn_grassmman = torch.linalg.norm(dots, dim=2) ** 2. * self.grassman_scale
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_riemmanian = self.attn_drop(cov_frobenius_norm(q, k) * self.riem_scale)
        # print(attn_riemmanian.shape)
        attn_ = self.conv_attn(torch.cat((attn, attn_riemmanian, attn_grassmman), dim=1)).softmax(
            dim=-1)

        out = torch.matmul(self.attn_drop(attn_), v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)

        return self.proj_drop(self.proj(out)), attn_


class EuclGrassAtt(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, sequence_length=-1,
                 ln_attention=True):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        # self.riem_scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.grassman_scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.qkv = Linear(dim, dim * 3, bias=True)

        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)
        self.sequence_len = sequence_length
        #self.attn_matrix = nn.Parameter(torch.randn(sequence_length, sequence_length))
        if ln_attention:
            self.conv_attn = nn.Sequential(
                nn.LayerNorm((2 * num_heads, sequence_length, sequence_length)),
                nn.Conv2d(in_channels=2 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1))
            )
        else:
            self.conv_attn = nn.Sequential(

                nn.Conv2d(in_channels=2 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1))
            )

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # old_shape = q.shape

        qgr, _ = grassmanian_point(q)
        kgr, _ = grassmanian_point(k)
        # print(qgr.shape)
        # # rieem_attn = log_dist(q, k, use_covariance=True, use_log=False)
        #
        qgr = qgr  # .permute(0, 1, 3, 2)
        kgr = kgr.permute(0, 1, 3, 2)

        dots = torch.matmul(qgr, kgr).unsqueeze(2)

        attn_grassmman = torch.linalg.norm(dots, dim=2) ** 2. * self.grassman_scale
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn_ = self.conv_attn(torch.cat((attn, attn_grassmman), dim=1)).softmax(
            dim=-1)

        out = torch.matmul(self.attn_drop(attn_), v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj_drop(self.proj(out)), attn_


class EuclideanRiemmanianAtt(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attention_dropout=0.1, projection_dropout=0.1,
                 sequence_length=-1, ln_attention=False):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.riem_scale = nn.Parameter(torch.tensor(head_dim ** -0.5))

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)
        self.sequence_len = sequence_length
        if ln_attention:
            self.conv_attn = nn.Sequential(
                nn.LayerNorm((2 * num_heads, sequence_length, sequence_length)),
                # nn.InstanceNorm2d(2*self.num_heads),
                nn.Conv2d(in_channels=2 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1))
            )
        else:
            self.conv_attn = nn.Sequential(

                nn.Conv2d(in_channels=2 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1))
            )

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ((q @ k.transpose(-2, -1)) * self.scale)
        attn_riemmanian = cov_frobenius_norm(q, k) * self.riem_scale

        # print(attn_riemmanian.std(),attn.std(),attn.mean(),attn_riemmanian.mean())
        attn_ = torch.cat((attn, attn_riemmanian), dim=1)
        # [N, C, H , W] --> [N, H, W, C]
        # 0 ,
        # attn_ = self.norm(attn_.permute(0,2,3,1)).permute(0,3,1,2)
        attn_ = self.conv_attn(attn_).softmax(dim=-1)

        out = torch.matmul(self.attn_drop(attn_), v)

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj_drop(self.proj(out)), attn_


class ManifoldEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    Attenion on riemmanian and grassmanian manifolds


    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1, attention_type='all', sequence_length=-1,
                 ln_attention=False, return_map=False, **kwargs):
        super(ManifoldEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        if attention_type == 'spd':
            self.self_attn = EuclideanRiemmanianAtt(dim=d_model, num_heads=nhead,
                                                    attention_dropout=attention_dropout, projection_dropout=dropout,
                                                    sequence_length=sequence_length, ln_attention=ln_attention)
        elif attention_type == 'all':
            self.self_attn = EuclRiemGrassAtt(dim=d_model, num_heads=nhead,
                                              attention_dropout=attention_dropout, projection_dropout=dropout,
                                              sequence_length=sequence_length, ln_attention=ln_attention,
                                              return_map=return_map)
        elif attention_type == 'gm':
            self.self_attn = EuclGrassAtt(dim=d_model, num_heads=nhead,
                                          attention_dropout=attention_dropout, projection_dropout=dropout,
                                          sequence_length=sequence_length, ln_attention=ln_attention)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x, w = self.self_attn(self.pre_norm(src))
        src = src + self.drop_path(x)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src, w


class ManifoldformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 attention_type='riem',
                 sequence_length=None,
                 ln_attention=True,
                 return_map=False):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.attention_type = attention_type
        self.return_map = return_map

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            ManifoldEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                 dim_feedforward=dim_feedforward, dropout=dropout,
                                 attention_dropout=attention_dropout, drop_path_rate=dpr[i],
                                 attention_type=attention_type,
                                 sequence_length=sequence_length, ln_attention=ln_attention, return_map=self.return_map)
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x, return_attention=False):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x, attn_weights = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)

        else:
            x = x[:, 0]

        x = self.fc(x)
        if return_attention:
            return x, attn_weights
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
