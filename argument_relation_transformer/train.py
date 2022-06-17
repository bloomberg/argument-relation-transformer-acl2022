"""Training the model, either using standard supervised training or active learning"""
import argparse
import datetime
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from argument_relation_transformer.infer import run_evaluation
from argument_relation_transformer.system import ArgumentRelationClassificationSystem
from argument_relation_transformer.utils import (
    AL_METHODS,
    DOMAINS,
    find_best_ckpt,
    get_epoch_num_from_path,
    set_seeds,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str,
        default="data/",
        help="Parent level directory containing jsonl format dataset",
    )
    parser.add_argument(
        "--ckptdir",
        type=str,
        default="checkpoints/",
        help="Directory to save model checkpoints and active learning sample indices",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DOMAINS,
        help="Available domains for argument relation prediction task",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="A string identifier of an experiment run, it is recommended to include `model` hyperparameters and random seed in it",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--end-to-end", action="store_true", help="Whether to assume heads are given"
    )
    parser.add_argument(
        "--huggingface-path",
        type=str,
        default="./huggingface/",
        help="Directory where the pre-trained huggingface transformers are saved",
    )

    ## Hyper-parameters
    # optimizer
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--scheduler-type", type=str, choices=["linear", "constant"])

    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=20)

    parser.add_argument(
        "--from-al",
        action="store_true",
        help="If set to True, train over samples selected by AL method.",
    )
    parser.add_argument("--al-method", type=str, choices=AL_METHODS)
    parser.add_argument(
        "--current-sample-size",
        type=int,
        help="If set --from-al to True, this will be used to identify the selected training set.",
    )

    args = parser.parse_args()
    set_seeds(args.seed)

    if args.dataset in ["echr", "cdcp"]:
        print(f"dataset {args.dataset} has only two classes, convert task into binary")
        args.task = "binary"
    else:
        args.task = "ternary"

    trainer_args = dict()
    if torch.cuda.is_available():
        trainer_args["gpus"] = 1
        trainer_args["log_gpu_memory"] = True
        print("use GPU training")
    else:
        trainer_args["gpus"] = 0
        print("use CPU training")

    if args.from_al:
        checkpoint_path = f"{args.ckptdir}/{args.exp_name}_model-trained-on-{args.current_sample_size}/"
    else:
        checkpoint_path = f"{args.ckptdir}/{args.exp_name}/"
    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)

    tb_logger = TensorBoardLogger(f"tb_logs/", name=f"{args.exp_name}")

    if args.task == "binary":
        ckpt_fname = "{epoch}-{val_loss:.4f}-{val_acc:.4f}-{val_link_f1:.4f}"
        monitor = "val_link_f1"
    else:
        ckpt_fname = "{epoch}-{val_loss:.4f}-{val_acc:.4f}-{val_macro_f1:.4f}"
        monitor = "val_macro_f1"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=ckpt_fname,
        monitor=monitor,
        mode="max",
        save_top_k=1,
    )
    trainer_args["logger"] = tb_logger
    trainer_args["callbacks"] = [checkpoint_callback]
    trainer = pl.Trainer.from_argparse_args(args, **trainer_args)

    if args.from_al:
        print(
            f">> Train model over actively selected samples from {args.al_method}.{args.current_sample_size}.jsonl"
        )
        al_selected_data_path = os.path.join(
            args.ckptdir,
            args.exp_name,
            f"{args.al_method}.{args.current_sample_size}.jsonl",
        )
        model = ArgumentRelationClassificationSystem(args, al_selected_data_path)
    else:
        model = ArgumentRelationClassificationSystem(args)

    trainer.fit(model)

    # run test, evaluation, and save results
    # predictions will be saved to `outputs/{args.exp_name}.jsonl`
    # scores will be saved to `scores/{args.exp_name}.scores`
    best_ckpt, _ = find_best_ckpt(ckpt_path=checkpoint_path)
    print(
        f">> Test on {best_ckpt}, results will be saved to `outputs/{args.exp_name}.jsonl`"
    )
    results = trainer.test(ckpt_path=best_ckpt)[0]
    results["epoch"] = get_epoch_num_from_path(best_ckpt)
    run_evaluation(
        output_path=os.path.join("outputs", f"{args.exp_name}.jsonl"),
        label_path=os.path.join(args.datadir, f"{args.dataset}_test.jsonl"),
        exp_name=args.exp_name,
        task=model.task,
    )


if __name__ == "__main__":
    main()
