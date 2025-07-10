import argparse
from enum import Enum
from types import SimpleNamespace

def merge_dict(a, b):
    a = a.copy()
    a.update(b)
    return a

def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--embd-dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--only-critic-trainable",
        action="store_true"
    )
    parser.add_argument(
        "--no-need-boat-action",
        action="store_true",
        help="boat action will use heuristic algorithms."
    )
    parser.add_argument(
        "--n-consider-goods",
        type=int,
        default=15,
        help="number of goods to be considered by a robot when decoding."
    )



def get_model_config(args):
    return {
        "hidden_dim": args.hidden_dim,
        "embd_dim": args.embd_dim,
        # "compile": True,
        "nhead": 4,
        "num_layers": args.num_layers,
        "only_critic_trainable": args.only_critic_trainable,
        "no_need_boat_action": args.no_need_boat_action,
        "n_consider_goods": args.n_consider_goods
    }


def add_env_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--map_file",
        type=str,
        default="maps/map8.txt",
        help="map file path."
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Enable reward_shaping_split",
    )
    parser.add_argument(
        "--idle-penalty",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=20,
        help="number of frame accumulated to action."
    )
    parser.add_argument(
        "--seed_l",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed_r",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--final-reward",
        action="store_true",
        default="accumulate rewards to the final reward."
    )
    parser.add_argument(
        "--fastmoney",
        action="store_true",
        help="when goods are deliveried to berth, we will have reward."
    )
    parser.add_argument(
        "--event",
        action="store_true",
        help="if event occurs, we make action regardless of frame-interval."
    )


def get_env_config(args):
    return {"map_file": args.map_file, "base_dir": "./", "seed_l": args.seed_l, "seed_r": args.seed_r,
            "reward_shaping_split": args.split,
            "reward_shaping_idle_penalty": args.idle_penalty,
            "frame_interval": args.frame_interval,
            "final_reward": args.final_reward,
            "fastmoney": args.fastmoney,
            "event": args.event
            }

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

    n_robots = 3
    n_robot_state = len(RobotState)

    n_workstations = 4

    n_shelves = 8
    n_sku_types = 8
    
    inf = 1000000
    max_frame = 500

if __name__ == "__main__":
    pass