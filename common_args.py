from enum import Enum

class LOGLEVEL(Enum):
    INFO = 0
    WARN = 1

class MapState(Enum):
    EMPTY = 0
    SHELF_EMPTY = 1
    # 0 空地
    # 1 货架空地

class RobotState(Enum):
    PICK = 0
    DELIVER = 1
    RETURN = 2
    END = 3
    # 0 在出发点或完成送回货架，前往下一个货架
    # 1 拿到货架，前往工作站
    # 2 完成运送，放回货架
    # 3 结束了，不动
    # 0 -> 1 -> 2 -> 0 -> 1 -> 2 ... 2 -> 3

class env_attr:
    x_max = 8
    y_max = 5
    n_map_state = 5 

    n_robots = 2
    n_robot_state = len(RobotState)

    n_workstations = 1

    n_shelves = 8
    n_sku_types = 8
    
    inf = 1000000
    max_frame = 500

    pick_reward = 5
    deliver_reward = 5
    return_reward = 10

if __name__ == "__main__":
    pass