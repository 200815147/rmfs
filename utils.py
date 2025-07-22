import json
import pdb

import matplotlib.pyplot as plt
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



def create_grid_visualization(colors, cell_labels=None, title="网格图", save_path="grid_visualization.png", 
                             grid_color='black', grid_width=1.0, figsize=None, dpi=300, colormap=None):
    """
    创建自定义颜色的网格图并保存为图片
    
    参数:
        colors: 二维数组，形状为(n, m)，每个元素为颜色值
                颜色值可以是:
                - 颜色名称字符串（如'red', 'blue'）
                - RGB元组（如(1, 0, 0)表示红色）
                - 0-1之间的标量值（需配合colormap使用）
        cell_labels: 二维数组，形状为(n, m)，每个元素为单元格内的文本标签（可选）
        title: 图表标题（字符串）
        save_path: 保存图片的路径（字符串）
        grid_color: 网格线的颜色（字符串或RGB元组）
        grid_width: 网格线的宽度（浮点数）
        figsize: 图表大小（元组，如(10, 8)表示宽10英寸，高8英寸）
        dpi: 图片分辨率（每英寸点数）
        colormap: 颜色映射对象，当colors为标量值时使用（如plt.cm.viridis）
    """
    # 获取网格的行数和列数
    n, m = np.array(colors).shape
    
    # 创建图形和轴
    if figsize is None:
        # 自动计算合适的图形大小，每个单元格约1英寸
        figsize = (m, n)
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制网格
    for i in range(n + 1):
        ax.axhline(i, color=grid_color, linewidth=grid_width)
    for j in range(m + 1):
        ax.axvline(j, color=grid_color, linewidth=grid_width)
    
    # 填充每个单元格的颜色
    for i in range(n):
        for j in range(m):
            color = colors[i][j]
            # 绘制矩形填充颜色
            rect = plt.Rectangle((j, i), 1, 1, fill=True, color=color, edgecolor='none')
            ax.add_patch(rect)
            
            # 如果提供了单元格标签，则添加文本
            if cell_labels is not None and cell_labels[i][j] is not None:
                ax.text(j + 0.5, i + 0.5, str(cell_labels[i][j]), 
                        ha='center', va='center', color='black', 
                        fontsize=min(figsize[0] * 2, figsize[1] * 2))
    
    # 设置坐标轴范围和标签
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)  # 反转y轴，使原点位于左上角
    ax.set_aspect('equal')
    ax.set_title(title)
    
    # 隐藏坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 如果使用标量值和颜色映射，添加颜色条
    if colormap is not None and isinstance(colors[0][0], (int, float)):
        cmap = plt.cm.get_cmap(colormap)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"网格图已保存至: {save_path}")

# 示例使用：定义颜色矩阵并创建网格图
if __name__ == "__main__":
    # 示例1：使用颜色名称
    file = 'small'
    with open(f'{file}.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    x_max = json_data['x_max']
    y_max = json_data['y_max']
    colors = [['white'] * x_max for _ in range(y_max)]
    for workstation in json_data['workstations']:
        x, y = workstation
        colors[x_max - 1 - y][x] = 'black'
    for shelf in json_data['shelves']:
        x, y = shelf['coord']
        colors[x_max - 1 - y][x] = 'blue'
    create_grid_visualization(colors, title=f"{file}_layout", save_path=f"output/{file}.png")
    # exit(0)