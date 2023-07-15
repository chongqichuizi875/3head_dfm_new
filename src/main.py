from arg_parser import arg_parser, generate_params
import json
from utils import *
import torch
import numpy as np
import os
from training import run

if __name__ == "__main__":
    args = arg_parser()
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        for h_param in config:
            setattr(args, h_param, config[h_param])
    params = generate_params(args)
    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    
    # save path
    i = 0
    base_dir_name = params['data']+'-'+params['method']+'-seed'+str(params['seed'])
    dir_name = base_dir_name
    while os.path.exists(os.path.join(params['save_path'], dir_name)):
        i += 1
        dir_name = f'{base_dir_name}_{i}'
    dump_path = os.path.join(params['save_path'], dir_name)
    os.makedirs(dump_path)
    param_file = os.path.join(dump_path, "params.json")
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)
    params["dump_path"] = dump_path
    # train
    run(params)