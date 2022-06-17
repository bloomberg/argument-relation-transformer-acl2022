import glob
import json
import os
import random

import numpy as np
import torch

DOMAINS = ["ampere", "ukp", "cdcp", "abst_rct", "echr"]
AL_METHODS = [
    "random",
    "max-entropy",
    "bald",
    "disc",
    "distance",
    "vocab",
    "no-disc",
    "coreset",
]


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_best_ckpt(ckpt_path):
    """Find the most recent path and load the model with best f1"""
    ckpt_paths = glob.glob(ckpt_path + "epoch=*")

    if len(ckpt_paths) == 0:
        return None, None

    def _get_f1_score(_path):
        basename = os.path.basename(_path)
        f1_score = basename.split(".ckpt")[0]
        f1_score = f1_score.split("_f1=")[1]
        return float(f1_score)

    ckpt_sorted = sorted(ckpt_paths, key=lambda x: _get_f1_score(x), reverse=True)
    return ckpt_sorted[0], _get_f1_score(ckpt_sorted[0])


def get_epoch_num_from_path(path):
    base = os.path.basename(path)
    base = base.split("-")[0][6:]
    return int(base)


def load_latest_ckpt_from_globs(ckpt_list):
    ckpt_list = sorted(
        ckpt_list, key=lambda x: get_epoch_num_from_path(x), reverse=True
    )
    return ckpt_list[0]


def load_latest_ckpt(exp_name, task, domain, epoch_id=-1):

    if epoch_id == -1:
        all_ckpts = glob.glob(f"checkpoints/{task}/{domain}/{exp_name}/epoch*")
    else:
        all_ckpts = glob.glob(
            f"checkpoints/{task}/{domain}/{exp_name}/epoch={epoch_id}-*"
        )

    ckpt_list = sorted(
        all_ckpts, key=lambda x: get_epoch_num_from_path(x), reverse=True
    )
    assert len(ckpt_list) > 0, f"no checkpoint found for {exp_name}, epoch={epoch_id}"
    return ckpt_list[0], get_epoch_num_from_path(ckpt_list[0])


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            r = {key: _apply(value) for key, value in x.items()}
            return r
            # return {
            #     key: _apply(value)
            #     for key, value in x.items()
            # }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def load_vocab(domain, min_freq=0):
    path = f"vocab/{domain}.txt"
    vocab = dict()
    for ln in open(path):
        word, freq = ln.strip().split("\t")
        freq = int(freq)
        if freq < min_freq:
            break

        vocab[word] = freq
    return vocab
