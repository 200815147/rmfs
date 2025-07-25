import pdb

import torch
from torch import nn
import torch.nn.functional as F

def backward_hook(module, grad_input, grad_output):
    # for g in list(grad_output):
    #     if g is None:
    #         continue
    #     module.grads.append(g.abs().mean().item())
    if grad_input[0] is not None:
        # print('input')
        # print(grad_input[0], grad_input[0].shape, grad_input[0].sum())
        if grad_input[0].sum() < 1e-6:
            print(f'input {module}')
    if grad_output[0] is not None:
        # print('output')
        # print(grad_output[0], grad_output[0].shape, grad_output[0].sum())
        if grad_output[0].sum() < 1e-6:
            print(f'output {module}')
    # print(module)
    # pdb.set_trace()
    # if any(g.abs().min() < 1e-12 for g in list(grad_input) + list(grad_output) if g is not None):
    #     print(f"module {module} has nan grad")
    #     import pdb
    #     pdb.set_trace()
        
    # if any(torch.isnan(g).any() for g in list(grad_input) + list(grad_output) if g is not None):
    #     print(f"module {module} has nan grad")
    #     import pdb
    #     pdb.set_trace()
        # raise RuntimeError(f'NaN values detected in the gradient w.r.t. the input of {module}')


class VacancyMLP(nn.Module):
    def __init__(self, input_dim, spatial_dim, embed_dim, n_shelves):
        super().__init__()
        self.n_shelves = n_shelves
        self.spatial_dim = spatial_dim
        self.vacancy_mlp = nn.Sequential(
            nn.Linear(spatial_dim, embed_dim), nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LeakyReLU()
        )
        self.shelf_mlp = nn.Sequential(
            nn.Linear(spatial_dim + input_dim, embed_dim), nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LeakyReLU()
        )
        self.embed_dim = embed_dim

    def forward(self, state, x):
        # state [B, Nv, 1]
        state = state.squeeze(-1)
        B = state.shape[0]
        Nv = state.shape[1]
        v_batch_idx, v_idx = torch.where(state == self.n_shelves)
        v_feat = x[v_batch_idx, v_idx, :self.spatial_dim]
        v_out = self.vacancy_mlp(v_feat)
        s_batch_idx, s_idx = torch.where(state != self.n_shelves)
        s_feat = x[s_batch_idx, s_idx]
        s_out = self.shelf_mlp(s_feat)
        out = torch.zeros((B, Nv, self.embed_dim), device=x.device).float()
        out[v_batch_idx, v_idx] = v_out
        out[s_batch_idx, s_idx] = s_out
        return out
    
class RobotMLP(nn.Module):
    def __init__(self, input_dim, spatial_dim, embed_dim, n_shelves):
        super().__init__()
        self.n_shelves = n_shelves
        self.spatial_dim = spatial_dim
        self.robot_mlp = nn.Sequential(
            nn.Linear(spatial_dim, embed_dim), nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LeakyReLU()
        )
        self.shelf_mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LeakyReLU()
        )
        self.placeholder = nn.Parameter(torch.randn(embed_dim))
        self.embed_dim = embed_dim

    def forward(self, shelf, coord, inventory):
        # shelf [B, 1] coord [B, 2] inventory [B, ns, ntype]
        # pdb.set_trace()
        robot_out = self.robot_mlp(coord).unsqueeze(1)
        B = shelf.shape[0]
        no_shelf_idx = torch.where(shelf == self.n_shelves)[0]
        carry_shelf_idx = torch.where(shelf != self.n_shelves)[0]
        s_feat = inventory[carry_shelf_idx, shelf[carry_shelf_idx]]
        s_out = self.shelf_mlp(s_feat)
        shelf_out = torch.zeros((B, self.embed_dim), device=shelf.device).float()
        shelf_out[no_shelf_idx] = self.placeholder
        shelf_out[carry_shelf_idx] = s_out
        shelf_out = shelf_out.unsqueeze(1)
        return robot_out + shelf_out
    


class DistanceAwareMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, num_rbf=16, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        # RBF parameters
        self.num_rbf = num_rbf
        self.centers = nn.Parameter(torch.linspace(0, 10, num_rbf))
        self.gamma   = nn.Parameter(torch.ones(num_rbf) * 0.5)
        # project RBF → scalar bias
        self.dist_proj = nn.Linear(num_rbf, 1, bias=False)

    def _rbf(self, dists):
        # dists: (B, L, L)
        # output: (B, L, L, num_rbf)
        return torch.exp(-self.gamma * (dists.unsqueeze(-1) - self.centers)**2)

    def forward(self,
                query,                # (L, B, D)
                key,                  # (S, B, D)
                value,                # (S, B, D)
                coords=None,          # (B, L, 3) or (B, S, 3)
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None):
        """
        coords: 坐标张量，如果 query/key 长度相同，可只传一个；否则传 key 的 coords
        """
        # 1. 先算 standard Q,K,V
        #    PyTorch 会把 attn_mask/bias 统一加到 attn_output_weights 里
        #    但它不提供 bias 参数，我们要 override 整个打分过程
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        # 2. 计算 pairwise distance
        #    假设 coords_query == coords_key，如果不同可分开传
        # coords: (B, L, 3)
        assert coords is not None, "需要提供 coords 计算距离 bias"
        coords_q = coords
        coords_k = coords
        diff = coords_q.unsqueeze(2) - coords_k.unsqueeze(1)   # (B, L, 1, 3) - (B, 1, L, 3)
        dists = diff.norm(dim=-1)                              # (B, L, L)

        # 3. RBF 编码并投影成 scalar bias
        rbf_feats = self._rbf(dists)                           # (B, L, L, num_rbf)
        bias = self.dist_proj(rbf_feats).squeeze(-1)           # (B, L, L)

        # 4. 直接调用父类的多头注意力实现，但注入 bias
        #    我们把 bias 加到 attn_mask，如果你同时用 attn_mask，要把它们合并
        #    PyTorch 1.13+ support attn_bias kwarg; 否则可在 forward 源码里插入。
        # 这里我们假设新版支持 attn_bias：
        return super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            attn_bias=bias           # 把我们的距离 bias 注入进去
        )


# --------- 把它插入到 Transformer 中 ---------
# 你可以用 TransformerEncoderLayer 自定义：
class DistanceAwareTransformerLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, **kwargs):
        super().__init__(d_model, nhead, **kwargs)
        # 用我们自定义的 Attention 替换原来的 self_attn
        self.self_attn = DistanceAwareMultiheadAttention(d_model, nhead)

    def forward(self, src, coords, src_mask=None, src_key_padding_mask=None):
        """
        src: (L, B, D)
        coords: (B, L, 3)
        """
        # 1. Self‑Attention with distance bias
        sa_out, _ = self.self_attn(
            src, src, src,
            coords=coords,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(sa_out)
        src = self.norm1(src)
        # 2. FFN
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src