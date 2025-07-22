import pdb

import torch
from torch import nn

from common_args import env_attr


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
    def __init__(self, input_dim, spatial_dim, embed_dim):
        super().__init__()
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
        v_batch_idx, v_idx = torch.where(state == env_attr.n_shelves)
        v_feat = x[v_batch_idx, v_idx, :self.spatial_dim]
        v_out = self.vacancy_mlp(v_feat)
        s_batch_idx, s_idx = torch.where(state != env_attr.n_shelves)
        s_feat = x[s_batch_idx, s_idx]
        s_out = self.shelf_mlp(s_feat)
        out = torch.zeros((B, Nv, self.embed_dim), device=x.device).float()
        out[v_batch_idx, v_idx] = v_out
        out[s_batch_idx, s_idx] = s_out
        return out
    
class RobotMLP(nn.Module):
    def __init__(self, input_dim, spatial_dim, embed_dim):
        super().__init__()
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
        no_shelf_idx = torch.where(shelf == env_attr.n_shelves)[0]
        carry_shelf_idx = torch.where(shelf != env_attr.n_shelves)[0]
        s_feat = inventory[carry_shelf_idx, shelf[carry_shelf_idx]]
        s_out = self.shelf_mlp(s_feat)
        shelf_out = torch.zeros((B, self.embed_dim), device=shelf.device).float()
        shelf_out[no_shelf_idx] = self.placeholder
        shelf_out[carry_shelf_idx] = s_out
        shelf_out = shelf_out.unsqueeze(1)
        return robot_out + shelf_out