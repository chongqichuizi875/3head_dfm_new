import argparse
import json
import os
import pathlib
from copy import deepcopy
from pretrain import run
# from stream_train_test import stream_run
import numpy as np
import torch


def run_params(args):
    params = deepcopy(vars(args))
    params["model"] = "MLP_SIG"
    params["optimizer"] = "Adam"
    if args.data_cache_path != "None":
        pathlib.Path(args.data_cache_path).mkdir(parents=True, exist_ok=True)
    if args.data == "criteo":
        if args.mode == "pretrain":
            if args.method == "Pretrain":
                params["loss"] = "cross_entropy_loss"
                params["dataset"] = "baseline_prtrain"
            elif args.method == "DFM":
                params["loss"] = "delayed_feedback_loss"
                params["dataset"] = "dfm_prtrain"
                params["model"] = "MLP_EXP_DELAY"
            elif args.method == "FSIW":
                params["loss"] = "cross_entropy_loss"
                params["dataset"] = args.fsiw_pretraining_type + "_cd_" + str(args.CD)
                params["model"] = "MLP_FSIW"
            elif args.method == "ES-DFM":
                params["loss"] = "tn_dp_pretraining_loss"
                params["dataset"] = "tn_dp_pretrain_cut_hour_" + str(args.C)
                params["model"] = "MLP_tn_dp"
            elif args.method == "3head":
                params["loss"] = "3head_dfm_loss"
                params["dataset"] = "tn_dp_pretrain_cut_hour_" + str(args.C)
                params["model"] = "MLP_tn_dp_3head"
            else:
                raise ValueError(
                    "{} method do not need pretraining other than Pretrain".format(args.method))
        else:
            if args.method == "Pretrain":
                params["loss"] = "none_loss"
                params["dataset"] = "last_30_train_test_oracle"
            elif args.method == "Oracle":
                params["loss"] = "cross_entropy_loss"
                params["dataset"] = "last_30_train_test_oracle"
            elif args.method == "DFM":
                params["loss"] = "delayed_feedback_loss"
                params["dataset"] = "last_30_train_test_dfm"
            elif args.method == "FSIW":
                params["loss"] = "fsiw_loss"
                params["dataset"] = "last_30_train_test_fsiw"
            elif args.method == "ES-DFM":
                params["loss"] = "esdfm_loss"
                params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                    str(args.C)
            elif args.method == "Vanilla":
                params["loss"] = "cross_entropy_loss"
                params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                    str(1)
            elif args.method == "FNW":
                params["loss"] = "fake_negative_weighted_loss"
                params["dataset"] = "last_30_train_test_fnw"
            elif args.method == "FNC":
                params["loss"] = "cross_entropy_loss"
                params["dataset"] = "last_30_train_test_fnw"
            elif args.method == "3head":
                params["loss"] = "3head_dfm_loss"
                params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                                    str(args.C)
    elif args.data == "coat":
        if args.method == "ES-DFM":
            params["loss"] = "esdfm_loss"
            params["dataset"] = "coat"
            params["model"] = "MLP_tn_dp"
        elif args.method == "3head":
            params["loss"] = "3head_dfm_loss"
            params["dataset"] = "coat"
            params["model"] = "MLP_tn_dp_3head"
    elif args.data == "yahoo":
        if args.method == "ES-DFM":
            params["loss"] = "esdfm_loss"
            params["dataset"] = "yahoo"
            params["model"] = "MLP_tn_dp"
        elif args.method == "3head":
            params["loss"] = "3head_dfm_loss"
            params["dataset"] = "yahoo"
            params["model"] = "MLP_tn_dp_3head"

    return params

if __name__ == "__main__":
    torch.set_grad_enabled(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='Path to JSON configuration file')
    parser.add_argument("--method", help="delayed feedback method",
                        choices=["FSIW",
                                 "DFM",
                                 "ES-DFM",
                                 "FNW",
                                 "FNC",
                                 "Pretrain",
                                 "Oracle",
                                 "Vanilla",
                                 "3head"],
                        type=str)
    parser.add_argument(
        "--mode", type=str, choices=["pretrain", "stream"], help="training mode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--CD", type=int, default=7,
                        help="counterfactual deadline in FSIW")
    parser.add_argument("--C", type=int, default=0.25,
                        help="elapsed time in ES-DFM")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", type=str,
                        help="Using yahoo, coat or criteo",
                        choices=["yahoo",
                                 "coat",
                                 "criteo"])
    parser.add_argument("--data_cache_path", type=str, default="None")
    parser.add_argument("--model_ckpt_path", type=str,
                        help="path to save pretrained model")
    # pretrained model path
    parser.add_argument("--pretrain_fsiw0_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained fsiw0 model,  \
                            necessary for the streaming evaluation of FSIW method")
    parser.add_argument("--pretrain_fsiw1_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained fsiw1 model,  \
                            necessary for the streaming evaluation of FSIW method")
    parser.add_argument("--pretrain_baseline_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained baseline model(Pretrain),  \
                            necessary for the streaming evaluation of \
                                FSIW, ES-DFM, FNW, FNC, Pretrain, Oracle, Vanilla method")
    parser.add_argument("--pretrain_dfm_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained DFM model,  \
                            necessary for the streaming evaluation of \
                                DFM method")
    parser.add_argument("--pretrain_esdfm_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained ES-DFM model,  \
                            necessary for the streaming evaluation of \
                            ES-DFM method")
    parser.add_argument("--pretrain_3head_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained 3head DFM model,  \
                                                    necessary for the streaming evaluation of \
                                                    3head DFM method"
                        )
    parser.add_argument("--fsiw_pretraining_type", choices=["fsiw0", "fsiw1"], type=str, default="None",
                        help="FSIW needs two pretrained weighting model")

    parser.add_argument("--batch_size", type=int,
                        default=1024)
    parser.add_argument("--epoch", type=int, default=5,
                        help="training epoch of pretraining")
    parser.add_argument("--l2_reg", type=float, default=1e-6,
                        help="l2 regularizer strength")

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        for h_param in config:
            setattr(args, h_param, config[h_param])

    params = run_params(args)
    torch.manual_seed(params['seed'])
    np.random.seed(args.seed)

    # create save path
    i = 0
    base_dir_name = params['dataset']+'-'+params['method']+'-seed'+str(params['seed'])
    dir_name = base_dir_name
    while os.path.exists(os.path.join(params['save_path'], dir_name)):
        i += 1
        dir_name = f'{base_dir_name}_{i}'
    dump_path = os.path.join(params['save_path'], dir_name)
    os.makedirs(dump_path)
    param_file = os.path.join(dump_path, "params.json")
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    if params['mode'] == "pretrain":
        run(params)
    # else:
    #     stream_run(params)