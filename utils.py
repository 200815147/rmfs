import pdb

import numpy as np
import torch

from common_args import RobotState, env_attr


def dict_to_batch_tensor(x, dtype=torch.float32, device=None):
    """
    Recursively convert a nested structure of dicts/lists/tuples containing
    numpy arrays, Python lists, ints/floats into torch.Tensors.
    
    :param x: the object to convert
    :param device: torch.device or device str, e.g. "cuda" or "cpu"
    :param dtype: default dtype for numeric types (can override per-array later)
    """
    if isinstance(x, dict):
        return {k: dict_to_batch_tensor(v, device=device, dtype=torch.int32 if k in ['state', 'shelf', 'next_robot', 'action_mask'] else dtype) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        raise ValueError
    else:
        # leaf node: numpy array, torch tensor, scalar, etc.
        if isinstance(x, torch.Tensor):
            raise ValueError
        else:
            t = torch.as_tensor(x, dtype=dtype).unsqueeze(0)
        if device is not None:
            t = t.to(device)
        return t

def encode_action(x, y):
    return x * env_attr.y_max + y

def decode_action(action):
    return action // env_attr.y_max, action % env_attr.y_max

def flatten_obs(obs_batch):
    if not isinstance(obs_batch, dict):
        raise ValueError("Expected batched obs to be a dict")
    if isinstance(obs_batch['map']['id'], torch.Tensor):
        return flatten_obs_tensor(obs_batch)
    else:
        return flatten_obs_numpy(obs_batch)

def flatten_obs_numpy(obs_batch):
    """
    Flatten a batch of nested obs dicts (e.g., from Dict space) to shape (B, flat_dim).
    :param obs_batch: dict[str, np.ndarray] where each value has shape (B, ...)
    :return: np.ndarray of shape (B, flat_dim)
    """
    batch_size = None
    flat_components = []

    for key in sorted(obs_batch):  # sorted for consistent ordering
        val = obs_batch[key]
        if isinstance(val, dict):
            sub_flat = flatten_obs_numpy(val)
            flat_components.append(sub_flat)
        else:
            val = np.asarray(val)
            if batch_size is None:
                batch_size = val.shape[0]
            flat_components.append(val.reshape(batch_size, -1))  # flatten per item

    return np.concatenate(flat_components, axis=1)  # final shape: (B, flat_dim)

def flatten_obs_tensor(obs_batch: dict) -> torch.Tensor:
    batch_size = None
    flat_components = []

    for key in sorted(obs_batch):  # sorted for consistent ordering
        val = obs_batch[key]
        if isinstance(val, dict):
            # 递归处理嵌套 dict
            sub_flat = flatten_obs_tensor(val)
            flat_components.append(sub_flat)
        else:
            if not isinstance(val, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(val)} for key {key}")
            # 第 0 维是 batch size
            if batch_size is None:
                batch_size = val.size(0)
            # 展平除第一维以外的所有维度 
            flat_components.append(val.reshape(batch_size, -1))

    # 将所有扁平块按最后一维拼接
    return torch.cat(flat_components, dim=1)