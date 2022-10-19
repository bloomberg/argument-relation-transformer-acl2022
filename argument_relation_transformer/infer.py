"""Methods for evaluation and inference"""
import argparse
import datetime
import glob
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import precision_recall_fscore_support

from argument_relation_transformer.system import ArgumentRelationClassificationSystem
from argument_relation_transformer.utils import DOMAINS, find_best_ckpt


def run_evaluation(output_path, label_path, exp_name, task):
    """Given the goldstandard data, calculate macro-F1 for relation, and binary F1 for link prediction.
    Also report how many links are predicted vs. how many exist in labels.
    """
    true_data = dict()  # doc_id -> (tail, head) -> label
    pred_data = dict()  # doc_id -> (tail, head) -> prediction
    y_true, y_pred = [], []
    link_y_true, link_y_pred = [], []
    acc = []

    for ln in open(label_path):
        cur_obj = json.loads(ln)
        doc_id = cur_obj["doc_id"]
        if isinstance(doc_id, list):
            doc_id = "_".join(doc_id)

        true_data[doc_id] = dict()
        n_text = len(cur_obj["text"])

        support_pairs, attack_pairs = [], []
        head_set = set()
        for item in cur_obj["relations"]:
            head = item["head"]
            tail = item["tail"]
            rel_type = item["type"]

            if rel_type == "support":
                support_pairs.append((tail, head))
            else:
                attack_pairs.append((tail, head))
            head_set.add(head)

        # we assume the head are known, and iterate over all possible tails
        # to create the gold-standard set (true_data)
        for head_id in sorted(head_set):
            for i in range(n_text):
                if i == head_id:
                    continue
                cur_pair = (i, head_id)
                if cur_pair in support_pairs:
                    label = "support"
                elif cur_pair in attack_pairs:
                    label = "attack"
                else:
                    label = "no-rel"
                true_data[doc_id][cur_pair] = label

    for ln in open(output_path):
        cur_obj = json.loads(ln)
        doc_id = cur_obj["doc_id"]

        if isinstance(doc_id, list):
            doc_id = "_".join(doc_id)
        if doc_id not in pred_data:
            pred_data[doc_id] = dict()
        for pair in cur_obj["candidates"]:
            pair_idx = (pair["tail"], pair["head"])
            pair_pred = pair["prediction"]
            pred_data[doc_id][pair_idx] = pair_pred

    pred_cnt, true_cnt = 0, 0
    support_cnt, attack_cnt, link_cnt = 0, 0, 0

    for doc_id, true_pairs in true_data.items():
        pred_pairs = pred_data[doc_id] if doc_id in pred_data else {}
        for t in true_pairs:
            true_cnt += 1
            if t not in pred_pairs:
                pred = "no-rel"
            else:
                pred = pred_pairs[t]
                pred_cnt += 1
            y_true.append(true_pairs[t])
            y_pred.append(pred)
            acc.append(1 if pred == true_pairs[t] else 0)

            if task == "binary":
                if pred == "link":
                    link_cnt += 1
                    support_cnt += 1
            else:
                if pred == "support":
                    support_cnt += 1
                    link_cnt += 1
                elif pred == "attack":
                    attack_cnt += 1
                    link_cnt += 1

            link_y_true.append(0 if true_pairs[t] == "no-rel" else 1)
            link_y_pred.append(0 if pred == "no-rel" else 1)

    print(f"{true_cnt} pairs found in label, {pred_cnt} predicted")
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    binary_prec, binary_rec, binary_f1, _ = precision_recall_fscore_support(
        link_y_true, link_y_pred, average="binary", zero_division=0
    )

    print(
        f"Macro F1: {macro_f1}\tBinary F1: {binary_f1}\tPredicted: {pred_cnt}\tTotal: {true_cnt}"
    )

    fout = open(output_path + ".scores", "w")
    results = {
        "macro_prec": macro_prec,
        "macro_rec": macro_rec,
        "macro_f1": macro_f1,
        "binary_prec": binary_prec,
        "binary_rec": binary_rec,
        "binary_f1": binary_f1,
        "pred_samples": pred_cnt,
        "labeled_samples": true_cnt,
        "accuracy": np.mean(acc),
        "support_ratio": support_cnt / true_cnt,
        "attack_ratio": attack_cnt / true_cnt,
        "link_ratio": link_cnt / true_cnt,
    }
    fout.write(json.dumps(results))
    fout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./data/")
    parser.add_argument("--ckptdir", type=str, default="./checkpoints/")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=DOMAINS)

    parser.add_argument(
        "--end-to-end", action="store_true", help="Whether to assume heads are given"
    )
    parser.add_argument("--eval-set", type=str, choices=["val", "test"], default="test")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=20)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckptdir, args.exp_name)
    candidate_ckpt = glob.glob(f"{ckpt_path}/*.ckpt")
    assert len(candidate_ckpt) > 0, f"No checkpoint found under {ckpt_path}"
    if len(candidate_ckpt) > 1:
        best_ckpt, _ = find_best_ckpt(ckpt_path)
    else:
        best_ckpt = candidate_ckpt[0]
    print(f"Loading checkpoint from {best_ckpt}")

    eval_output_path = f"outputs/{args.exp_name}.jsonl"
    trainer = pl.Trainer.from_argparse_args(args, gpus=1)
    model = ArgumentRelationClassificationSystem.load_from_checkpoint(
        checkpoint_path=best_ckpt
    )
    trainer.test(model)

    run_evaluation(
        output_path=eval_output_path,
        label_path=f"{args.datadir}/{args.dataset}_{args.eval_set}.jsonl",
        exp_name=args.exp_name,
        task=model.task,
    )


if __name__ == "__main__":
    main()
