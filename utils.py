import json
import pdb
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from common_args import RobotState, env_attr


def dict_to_batch_tensor(x, dtype, device=None):
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

def read_xmap():
    with open("dataset/static/map_2_1715935647135.xmap", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nodes = data['exportMapDto']['exportMapNodeDtoList']
    # 1. 抽取所有 dict 的 key 并构造一个扁平的 key 序列
    all_keys = []
    shelves = []
    queues = []
    turns = []
    blockeds = []
    stations = []
    charger_pis = []
    chargers = []
    x_indexes, y_indexes = [], []
    for d in nodes:
        all_keys.append(d['cellType'])
        x_indexes.append(int(d['indexX']))
        y_indexes.append(int(d['indexY']))
        if d['cellType'] == 'SHELF_CELL':
            shelves.append(d)
        elif d['cellType'] == 'QUEUE_CELL':
            queues.append(d)
        elif d['cellType'] == 'TURN_CELL':
            turns.append(d)
        elif d['cellType'] == 'BLOCKED_CELL':
            blockeds.append(d)
        elif d['cellType'] == 'STATION_CELL':
            stations.append(d)
        elif d['cellType'] == 'CHARGER_PI_CELL':
            charger_pis.append(d)
        elif d['cellType'] == 'CHARGER_CELL':
            chargers.append(d)
        

    # 2. 用 Counter 统计出现次数
    counter = Counter(all_keys)
    json_dict = {}
    print(counter)
    print(min(x_indexes), max(x_indexes))
    print(min(y_indexes), max(y_indexes))
    min_x = min(x_indexes) - 1
    min_y = min(y_indexes) - 1
    x_max = max(x_indexes) - min_x + 1 + 1
    y_max = max(y_indexes) - min_y + 1 + 1
    json_dict['x_max'] = x_max
    json_dict['y_max'] = y_max
    json_dict['shelves'] = []
    json_dict['workstations'] = []
    json_dict['robots'] = []
    warehouse_map = np.zeros((x_max, y_max), dtype=np.int16)
    for shelf in shelves:
        json_dict['shelves'].append({
            'coord': [shelf['indexX'] - min_x, shelf['indexY'] - min_y],
            'inventory': []
        })
        warehouse_map[shelf['indexX'] - min_x, shelf['indexY'] - min_y] = 1
    for station in stations:
        json_dict['workstations'].append([station['indexX'] - min_x, station['indexY'] - min_y])
        warehouse_map[station['indexX'] - min_x, station['indexY'] - min_y] = 2

    with open('geekplus.json', 'w') as f:
        json.dump(json_dict, f)  
    for i in range(max(y_indexes) - min(y_indexes), -1, -1):
        for j in range(max(x_indexes) - min(x_indexes) + 1):
            if warehouse_map[j][i] == 0:
                print(' ', end='')
            elif warehouse_map[j][i] == 1:
                print('*', end='')
            else:
                print('#', end='')
        print('')
    # pdb.set_trace()

def gen_mid_layout():
    json_dict = {}
    x_max = 19
    y_max = 11
    json_dict['x_max'] = x_max
    json_dict['y_max'] = y_max
    json_dict['shelves'] = []
    json_dict['workstations'] = []
    json_dict['robots'] = []
    json_dict['n_sku_types'] = 20
    json_dict['order_num_l'] = 20
    json_dict['order_num_r'] = 50
    json_dict['order_time_l'] = 0
    json_dict['order_time_r'] = 200
    for i in range(4, 14, 3):
        for x in [i, i + 1]:
            for y in [1, 2, 3, 4, 6, 7, 8, 9]:
                shelf = {
                    'coord': [x, y],
                    'inventory': []
                }
                for j in range(json_dict['n_sku_types']):
                    shelf['inventory'].append([j, np.random.randint(1, 20)])
                json_dict['shelves'].append(shelf)
    for x in [0, x_max - 1]:
        for y in range(0, y_max, 2):
            json_dict['workstations'].append([x, y])
            if x == 0:
                json_dict['robots'].append([x + 1, y])
            else:
                json_dict['robots'].append([x - 1, y])
    with open('layouts/mid.json', 'w') as f:
        json.dump(json_dict, f) 


if __name__ == "__main__":
    # read_xmap()
    file = 'geekplus'
    gen_mid_layout()
    file = 'mid'
    with open(f'{file}.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    x_max = json_data['x_max']
    y_max = json_data['y_max']
    colors = [['white'] * x_max for _ in range(y_max)]
    for workstation in json_data['workstations']:
        x, y = workstation
        colors[y_max - 1 - y][x] = 'black'
    for shelf in json_data['shelves']:
        x, y = shelf['coord']
        colors[y_max - 1 - y][x] = 'blue'
    create_grid_visualization(colors, title=f"{file}_layout", save_path=f"output/{file}.png")
    # exit(0)