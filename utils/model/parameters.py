# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: set data_params, learning_params, net_params for modeling training
#
# Created at 03/01/2023

import torch as th

OBJ_MAP = {
    "latbuck20": ["latbuck20"],
    "tid": ["tid"],
    "latency": ["latency"],
    "stage_lat": ["stage_lat"],
    "stage_dt": ["stage_dt"],
    "stage_both": ["stage_lat", "stage_dt"]
}

ALL_OP_FEATS = ["ch1_type", "ch1_cbo", "ch1_enc"]
DEFAULT_DTYPE = th.float32
DEFAULT_DEVICE = th.device("cpu")

def get_gpus(gpu):
    if len(gpu.split(",")) == 1:
        gpu = int(gpu)
        try:
            cuda = gpu >= 0
        except:
            raise Exception(f"invalid gpu value {gpu}")
        device = th.device("cuda:{}".format(gpu)) if cuda else th.device("cpu")
        if cuda:
            th.cuda.set_device(gpu)
        gpus = None
    else:
        gpus = [int(i) for i in gpu.split(",")]
        assert gpus[0] == min(gpus), f"module must have its parameters and buffers on device cuda:{gpus[0]} " \
                                     f"(device_ids[0]) but found one of them on device {min(gpus)}"
        device = th.device("cuda:{}".format(min(gpus)) if th.cuda.is_available() else "cpu")
    return gpus, device


def set_data_params(args):
    assert args.ch1_type in ("on", "off")
    assert args.ch1_cbo in ("on", "off")
    assert args.ch1_enc in ("off", "w2v", "d2v")
    assert args.ch2 in ("on", "off")
    assert args.ch3 in ("on", "off")
    assert args.ch4 in ("on", "off")
    if args.granularity == "Q":
        assert args.obj in ("latency", "latbuck20", "tid")
    elif args.granularity == "QS":
        assert args.obj in ("stage_lat", "stage_dt", "stage_both")
    assert args.model_name in ("GTN", "RAAL", "QF", "TL", "AVGMLP")
    assert args.debug in (0, 1)

    return {
        "ch1_type": args.ch1_type,
        "ch1_cbo": args.ch1_cbo,
        "ch1_enc": args.ch1_enc,
        "ch2": args.ch2,
        "ch3": args.ch3,
        "ch4": args.ch4,
        "obj": args.obj,
        "model_name": args.model_name,
        "debug": False if args.debug == 0 else True
    }


def set_learning_params(args):
    gpus, device = get_gpus(args.gpu)
    learning_params = {
        "device": device,
        "gpu": gpus,
        "num_workers": args.nworkers,
        "batch_size": args.bs,
        "epochs": args.epochs,
        "seed": args.seed,
        "init_lr": 0.0007,
        "min_lr": 1e-6,
        "weight_decay": 0.0,
        "ckp_interval": 100,
        "loss_type": "wmape",
        "loss_ws": None
    }
    if args.init_lr is not None:
        learning_params["init_lr"] = args.init_lr
    if args.min_lr is not None:
        learning_params["min_lr"] = args.min_lr
    if args.weight_decay is not None:
        learning_params["weight_decay"] = args.weight_decay
    if args.ckp_interval is not None:
        learning_params["ckp_interval"] = args.ckp_interval
    if args.loss_type is not None:
        assert args.loss_type in {"msle", "wmape", "mae", "mape"}
        learning_params['loss_type'] = args.loss_type
    if len(OBJ_MAP[args.obj]) > 1:
        assert args.loss_ws is not None
        loss_ws = [float(w) for w in args.loss_ws.split(",")]
        assert len(loss_ws) == len(OBJ_MAP[args.obj])
        total = sum(loss_ws)
        learning_params["loss_ws"] = {m: w / total for m, w in zip(OBJ_MAP[args.obj], loss_ws)}
    return learning_params


def set_net_params(args):
    net_params = {
        "ped": 8,  # 8, positional encoding dimension
        "L_gtn": 5,  # 10
        "L_mlp": 4,
        "n_heads": 8,
        "hidden_dim": 128,
        "out_dim": 128,
        "mlp_dim": 128,
        "residual": True,
        "readout": "mean",
        "dropout": 0.0,
        "dropout2": 0.0,
        "batch_norm": True,
        "layer_norm": False,
        "ch1_type_dim": 8,
        "ch1_cbo_dim": 4,
        "ch1_enc_dim": 32,
        "out_norm": None,
        "agg_dim": None
    }

    if args.ped is not None:
        net_params["ped"] = args.ped
    if args.L_gtn is not None:
        net_params["L_gtn"] = args.L_gtn
    if args.L_mlp is not None:
        net_params["L_mlp"] = args.L_mlp
    if args.n_heads is not None:
        net_params["n_heads"] = args.n_heads
    if args.hidden_dim is not None:
        net_params["hidden_dim"] = args.hidden_dim
    if args.out_dim is not None:
        net_params["out_dim"] = args.out_dim
    if args.mlp_dim is not None:
        net_params["mlp_dim"] = args.mlp_dim
    if args.residual is not None:
        net_params["residual"] = False if args.residual == 0 else True
    if args.readout is not None:
        net_params["readout"] = args.readout
    if args.dropout is not None:
        net_params["dropout"] = args.dropout # for GTN
    if args.dropout2 is not None:
        net_params["dropout2"] = args.dropout2 # for MLP
    if args.batch_norm is not None:
        net_params["batch_norm"] = False if args.batch_norm == 0 else True
    if args.layer_norm is not None:
        net_params["layer_norm"] = False if args.layer_norm == 0 else True
    if args.ch1_type_dim is not None:
        net_params["ch1_type_dim"] = args.ch1_type_dim
    if args.ch1_cbo_dim is not None:
        net_params["ch1_cbo_dim"] = args.ch1_cbo_dim
    if args.ch1_enc_dim is not None:
        net_params["ch1_enc_dim"] = args.ch1_enc_dim
    if args.out_norm is not None:
        assert args.out_norm in ["BN", "LN", "IsoBN", "None"], ValueError(args.out_norm)
        net_params["out_norm"] = args.out_norm if args.out_norm != "None" else None
    if args.agg_dim is not None:
        net_params["agg_dim"] = args.agg_dim if args.agg_dim != "None" else None

    assert net_params["ch1_cbo_dim"] == 4, ValueError(net_params["ch1_cbo_dim"])
    return net_params


def set_params(args):
    data_params = set_data_params(args)
    learning_params = set_learning_params(args)
    net_params = set_net_params(args)
    return data_params, learning_params, net_params