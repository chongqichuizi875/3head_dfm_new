import argparse
from copy import deepcopy
import os
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='Path to JSON configuration file', default=None)
    parser.add_argument("--C", type=int, default=0.25,
                            help="elapsed time in ES-DFM")
    parser.add_argument("--D", type=int, default=0.6,
                            help="yahoo and coat Non-delayed propotion. \
                            0 means almost all delayed, few observed, \
                            1 means few delayed, all samples can be observed")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--datapath", type=str, default="data")
    parser.add_argument("--data", type=str,
                            help="Using yahoo, coat or criteo",
                            choices=["yahoo",
                                    "coat",
                                    "criteo"])
    parser.add_argument("--save_path", type=str, default="run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int,
                            default=1048576)
    parser.add_argument("--epoch", type=int, default=100,
                            help="training epoch of pretraining")
    parser.add_argument("--l2_reg", type=float, default=1e-6,
                            help="l2 regularizer strength")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--method", type=str, default="dfm",
                        choices=["dfm",
                                 "esdfm",
                                 "3head",
                                 "ora"])
    parser.add_argument("--build_data", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train",
                                 "eval",
                                 "both"])
    parser.add_argument("--hard_share", type=bool, default=True)
    parser.add_argument("--save_model", type=bool, default=False)
    args = parser.parse_args()
    return args

def generate_params(args):
    params = deepcopy(vars(args))
    return params