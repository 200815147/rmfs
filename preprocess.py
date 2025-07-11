import json
import pdb
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

origin_order_df = pd.read_csv('dataset/dynamic/origin_order.csv')
pick_list_df = pd.read_csv('dataset/dynamic/pick_list.csv')
pick_wave_df = pd.read_csv('dataset/dynamic/pick_wave.csv')
dispatch_df = pd.read_csv('dataset/dynamic/dispatch.csv')
shelf_position_df = pd.read_csv('dataset/dynamic/shelf_position.csv')
station_position_df = pd.read_csv('dataset/dynamic/station_position.csv')
sku_df = pd.read_csv('dataset/dynamic/stock_unit.csv')

def diff():
    # 比较原始订单和最终的拣选单
    # 发现差异较大
    # Aggregate total quantity per SKU in each table
    agg_original = origin_order_df.groupby('sku_code')['num'].sum().reset_index().rename(columns={'num': 'total_num_original'})
    agg_pick_list = pick_list_df.groupby('sku_code')['num'].sum().reset_index().rename(columns={'num': 'total_num_pick_list'})

    # Merge the aggregated results on SKU
    merged_skus = pd.merge(agg_original, agg_pick_list, on='sku_code', how='outer').fillna(0)

    # Convert quantities to integers
    merged_skus['total_num_original'] = merged_skus['total_num_original'].astype(int)
    merged_skus['total_num_pick_list'] = merged_skus['total_num_pick_list'].astype(int)

    # Check for discrepancies
    merged_skus['difference'] = merged_skus['total_num_original'] - merged_skus['total_num_pick_list']

    # Display SKUs where the total numbers differ
    discrepancies = merged_skus[merged_skus['difference'] != 0]

    print("Total SKUs compared:", len(merged_skus))
    print("Number of SKUs with matching totals:", len(merged_skus) - len(discrepancies))
    print("Number of SKUs with discrepancies:", len(discrepancies))
    print()

    if discrepancies.empty:
        print("All SKU quantities match exactly between original orders and pick lists.")
    else:
        print("SKUs with differing total quantities:")
        print(discrepancies)

def final_dispatch():
    # 按 pick_list_id 将“派单表”和“拣选单表”合并
    merged = pd.merge(dispatch_df, pick_list_df, on='pick_list_id', how='left')

    # 选取需要的列并重命名
    result = merged[['generate_time', 'station_id', 'pick_list_id', 'sku_code', 'num']].copy()
    result.columns = ['dispatch_time', 'station_id', 'pick_list_id', 'sku_code', 'quantity']

    print(result)

def read_xmap():
    with open("static/map_2_1715935647135.xmap", "r", encoding="utf-8") as f:
        data = json.load(f)
    # (Pdb) data.keys()
    # dict_keys(['exportMapDto', 'md5'])
    # md5 没用

    # (Pdb) data['exportMapDto'].keys()
    # dict_keys(['exportBaseMapDto', 'exportFloorDtoList', 'exportMapNodeDtoList', 'exportMapChargerDtoList', 'exportMapStationDtoList', 'exportFunctionalCellDtoList', 'exportMapAreaDtoList', 'exportMapAreaItemDtoList', 'exportSingleLaneDtoList'])
    # exportBaseMapDto 没用
    # exportFloorDtoList 没用
    # exportMapNodeDtoList 为所有节点
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
    
    print(counter)
    print(min(x_indexes), max(x_indexes))
    print(min(y_indexes), max(y_indexes))
    warehouse_map = np.zeros((max(x_indexes) - min(x_indexes) + 1, max(y_indexes) - min(y_indexes) + 1), dtype=np.int16)
    for shelf in shelves:
        warehouse_map[shelf['indexX'] - min(x_indexes), shelf['indexY'] - min(y_indexes)] = 1
    for station in stations:
        warehouse_map[station['indexX'] - min(x_indexes), station['indexY'] - min(y_indexes)] = 2
    for i in range(max(y_indexes) - min(y_indexes), -1, -1):
        for j in range(max(x_indexes) - min(x_indexes) + 1):
            if warehouse_map[j][i] == 0:
                print(' ', end='')
            elif warehouse_map[j][i] == 1:
                print('*', end='')
            else:
                print('#', end='')
        print('')
    pdb.set_trace()

def merge_sku():

    # 读取库存表 CSV 文件（请根据实际文件名/路径修改）
    inventory = pd.read_csv('dynamic/stock_unit.csv', encoding='utf-8')

    # 按货架 (shelf_code) 和 SKU (sku_code) 分组，并求 available_num 的总和
    agg_inventory = (
        inventory
        .groupby(['shelf_code', 'sku_code'])['available_num']
        .sum()
        .reset_index()
        .rename(columns={'available_num': 'total_available_num'})
    )
    filtered_inventory = agg_inventory[agg_inventory['total_available_num'] > 0]

    # 在本地环境中，你可以直接打印或展示
    print(filtered_inventory)
    
    # —— 方法3：将 total_available_num 按区间分箱，再绘制直方图 —— #
    # 例如将库存数量分为 [1,2], [3,4], [5,6], [7,10], [11,20], [21,50], [51,100], 100+ 等区间
    bins = [0, 2, 4, 6, 10, 20, 50, 100, filtered_inventory['total_available_num'].max()]
    labels = ['1-2', '3-4', '5-6', '7-10', '11-20', '21-50', '51-100', f'101+']
    filtered_inventory['bin_range'] = pd.cut(filtered_inventory['total_available_num'], bins=bins, labels=labels, include_lowest=True)

    bin_counts = filtered_inventory['bin_range'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    bin_counts.plot(kind='bar')
    plt.xlabel('Total Available Num Range')
    plt.ylabel('Number of (Shelf, SKU) Pairs')
    plt.title('Distribution of total_available_num by Range')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('dataset/freq_counts_binned.png', bbox_inches='tight')
    plt.close()


def order_time():
    # 读取拣选单表 CSV 文件，请根据实际路径修改文件名
    pick_lists = origin_order_df

    # 将 generate_time 列转换为 datetime 类型
    pick_lists['generate_time'] = pd.to_datetime(pick_lists['generate_time'])

    # 按小时对生成时间进行分组，并统计每小时内的拣选单数量
    pick_lists['hour'] = pick_lists['generate_time'].dt.floor('h')
    hourly_counts = pick_lists.groupby('hour').size().reset_index(name='order_count')

    # 绘制柱状图，并让柱子之间留白
    plt.figure(figsize=(12, 6))
    # 设置宽度为 0.8 小时，即 48 分钟左右，这样与整点有间隔
    bar_width = pd.Timedelta(minutes=48)
    plt.bar(hourly_counts['hour'], hourly_counts['order_count'], width=bar_width, align='center', edgecolor='black')

    plt.xlabel('Hour')
    plt.ylabel('Number of Pick Lists')
    plt.title('Hourly Distribution of Pick List Generation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图片到本地
    plt.savefig('dataset/hourly_pick_list_distribution_spaced.png', bbox_inches='tight')
    plt.close()


    # 提取每条记录的小时（0-23）
    pick_lists['hour_of_day'] = pick_lists['generate_time'].dt.hour

    # 按小时统计订单数量，并按小时顺序排列
    hourly_counts = (
        pick_lists['hour_of_day']
        .value_counts()
        .sort_index()
        .reset_index(name='order_count')
        .rename(columns={'index': 'hour'})
    )
    # pdb.set_trace()
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(hourly_counts['hour_of_day'], hourly_counts['order_count'], width=0.8, edgecolor='black')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Pick Lists')
    plt.title('Distribution of Pick List Generation by Hour of Day')
    plt.xticks(range(0, 24))  # 标记0-23小时
    plt.tight_layout()

    # 保存图片
    plt.savefig('dataset/hourly_of_day_pick_list_distribution.png', bbox_inches='tight')
    plt.close()

def sku_type_num():
    # 统计唯一的 sku_code 数量
    unique_skus = sku_df['sku_code'].nunique()

    print(f"库存表中共有 {unique_skus} 种不同的 SKU") # 23451

def sku_every_k(k, pick_lists=None):
    # 自定义时间段长度（分钟）
    # k = 30  # 可以修改为任意正整数，如 15、60、120

    # 读取拣选单表
    if pick_lists is None:
        pick_lists = pick_list_df
    pick_lists['generate_time'] = pd.to_datetime(pick_lists['generate_time'])

    # 向下取整到每 k 分钟的时间段
    interval = f'{k}min'
    pick_lists['time_bucket'] = pick_lists['generate_time'].dt.floor(interval)

    # 按时间段分组，统计每段内的拣选单数与 SKU 种类数
    summary = (
        pick_lists
        .groupby('time_bucket')
        .agg(order_count=('pick_list_id', 'count'),
            sku_count=('sku_code', pd.Series.nunique))
        .reset_index()
    )

    # 查看结果
    print(summary.head())
    summary.to_csv(f'dataset/pick_list_sku_stats_{k}min.csv', index=False)

def sku_num():
    # 统计每种 SKU 出现的次数
    sku_counts = pick_list_df['sku_code'].value_counts()

    # 每个出现次数 i 下有多少种 SKU（如出现 5 次的 SKU 有 12 个）
    sku_occurrence_stats = sku_counts.value_counts().sort_index()
    pdb.set_trace()
    # 转为 DataFrame，并重命名列
    sku_occurrence_stats_df = sku_occurrence_stats.reset_index()
    sku_occurrence_stats_df.columns = ['occurrence_count', 'num_skus']

    print(sku_occurrence_stats_df)

def filter_sku(k=11):
    # 步骤 2：统计每个 SKU 出现次数
    sku_counts = pick_list_df['sku_code'].value_counts()

    # 步骤 3：筛选出现次数 >= k 的 SKU
    # k = 5  # 自定义阈值
    frequent_skus = sku_counts[sku_counts >= k].index

    # 步骤 4：保留这些 SKU 的拣选单记录
    filtered_pick_lists = pick_list_df[pick_list_df['sku_code'].isin(frequent_skus)].copy()

    # 步骤 5：保存结果（可选）
    filtered_pick_lists.to_csv(f'dataset/filtered_pick_list_k{k}.csv', index=False)

    # 可选：预览前几行
    print(filtered_pick_lists.head())
    return filtered_pick_lists
# diff()
# final_dispatch()
# read_xmap()
# merge_sku()
# order_time()
# sku_num()
# sku_every_k(10)
# sku_num()
tmp = filter_sku(31)
unique_skus = tmp['sku_code'].nunique()

print(unique_skus) # 23451
sku_every_k(10, tmp)