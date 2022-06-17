import argparse
import json
import os

import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    RobertaTokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from argument_relation_transformer.dataset import ArgumentRelationDataset
from argument_relation_transformer.modeling import RobertaForDocumentSpanClassification


class ArgumentRelationClassificationSystem(pl.LightningModule):
    def __init__(self, hparams, al_selected_data_path=None):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)

        self.datadir = hparams.datadir
        self.dataset_name = hparams.dataset
        self.batch_size = hparams.batch_size
        self.window_size = hparams.window_size
        self.adam_epsilon = hparams.adam_epsilon
        self.learning_rate = hparams.learning_rate
        self.scheduler_type = hparams.scheduler_type
        self.warmup_steps = hparams.warmup_steps
        self.exp_name = hparams.exp_name
        self.seed = hparams.seed
        self.task = hparams.task
        self.end_to_end = hparams.end_to_end

        self.sampled_ids = None
        if al_selected_data_path is not None:
            self.sampled_ids = []
            for ln in open(al_selected_data_path):
                _id, _label = json.loads(ln)
                self.sampled_ids.append(_id)

        if hparams.huggingface_path is None:
            model_name_or_path = "roberta-base"
        else:
            model_name_or_path = os.path.join(hparams.huggingface_path, "roberta-base")

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
        self.model = RobertaForDocumentSpanClassification.from_pretrained(
            model_name_or_path,
            num_labels=3 if self.task == "ternary" else 2,
            config=model_name_or_path,
        )

    def training_step(self, batch, batch_idx):
        logits, loss = self.model(**batch)
        pred = logits.argmax(-1)
        labels = batch["labels"]

        accuracy = (pred == labels)[labels != -1].float().mean()
        pred_masked = pred[labels > -1].tolist()
        if len(pred_masked) > 0:

            if self.task == "binary":
                self.log(
                    "train_step_pos_ratio",
                    sum(pred_masked) / len(pred_masked),
                    on_step=True,
                    prog_bar=False,
                    logger=True,
                )
            else:
                supp_cnt = pred_masked.count(1)
                att_cnt = pred_masked.count(2)
                supp_ratio = supp_cnt / len(pred_masked)
                att_ratio = att_cnt / len(pred_masked)
                self.log(
                    "train_step_support_ratio",
                    supp_ratio,
                    on_step=True,
                    prog_bar=False,
                    logger=True,
                )
                self.log(
                    "train_step_attack_ratio",
                    att_ratio,
                    on_step=True,
                    prog_bar=False,
                    logger=True,
                )

        self.log("train_loss", loss, on_step=True, prog_bar=False, logger=True)
        self.log("train_acc", accuracy, on_step=True, prog_bar=False, logger=True)
        for i, param in enumerate(self.opt.param_groups):
            self.log(
                f"lr_group_{i}", param["lr"], on_step=True, prog_bar=False, logger=True
            )
        return {"loss": loss, "pred": pred, "labels": labels}

    def test_step(self, batch, batch_idx):
        logits, loss = self.model(**batch)

        pred = logits.argmax(-1)
        labels = batch["labels"]
        accuracy = (pred == labels)[labels != -1].float().mean()
        pred_unmasked = pred[labels != -1].tolist()

        if len(pred_unmasked) > 0:
            if self.task == "binary":
                self.log(
                    "test_pos_ratio",
                    sum(pred_unmasked) / len(pred_unmasked),
                    on_step=False,
                    prog_bar=False,
                    logger=True,
                )
            else:
                supp_cnt = pred_unmasked.count(1)
                att_cnt = pred_unmasked.count(2)
                supp_ratio = supp_cnt / len(pred_unmasked)
                att_ratio = att_cnt / len(pred_unmasked)
                self.log(
                    "test_support_ratio",
                    supp_ratio,
                    on_step=False,
                    prog_bar=False,
                    logger=True,
                )
                self.log(
                    "test_attack_ratio",
                    att_ratio,
                    on_step=False,
                    prog_bar=False,
                    logger=True,
                )

        self.log("test_loss", loss, on_step=False, prog_bar=False, logger=True)
        self.log("test_acc", accuracy, on_step=False, prog_bar=False, logger=True)

        # recover the original predictions for more accurate evaluation
        pred_results = dict()  # (src, tgt) -> [pred, label]
        for ids, p, l, i_str in zip(batch["ids"], pred, labels, batch["input_str"]):
            doc_id, head_prop_id, rel_dir = ids
            cur_samples = len(i_str) - 1
            if rel_dir == "backward":
                effective_l = l[:cur_samples].tolist()
                effective_p = p[:cur_samples].tolist()
                for tail_i in range(cur_samples):
                    tail_real_idx = head_prop_id - cur_samples + tail_i
                    pred_results[(doc_id, tail_real_idx, head_prop_id)] = (
                        effective_p[tail_i],
                        effective_l[tail_i],
                    )
            else:
                effective_l = l[1 : cur_samples + 1].tolist()
                effective_p = p[1 : cur_samples + 1].tolist()
                for tail_i in range(cur_samples):
                    tail_real_idx = head_prop_id + tail_i + 1
                    pred_results[(doc_id, tail_real_idx, head_prop_id)] = (
                        effective_p[tail_i],
                        effective_l[tail_i],
                    )

        return {
            "loss": loss,
            "acc": accuracy,
            "pred": pred,
            "labels": labels,
            "results": pred_results,
        }

    def validation_step(self, batch, batch_idx):
        logits, loss = self.model(**batch)

        pred = logits.argmax(-1)
        labels = batch["labels"]
        accuracy = (pred == labels)[labels != -1].float().mean()
        pred_unmasked = pred[labels != -1].tolist()

        if len(pred_unmasked) > 0:
            if self.task == "binary":
                self.log(
                    "val_pos_ratio",
                    sum(pred_unmasked) / len(pred_unmasked),
                    on_step=False,
                    prog_bar=False,
                    logger=True,
                )
            else:
                supp_cnt = pred_unmasked.count(1)
                att_cnt = pred_unmasked.count(2)
                supp_ratio = supp_cnt / len(pred_unmasked)
                att_ratio = att_cnt / len(pred_unmasked)
                self.log(
                    "val_support_ratio",
                    supp_ratio,
                    on_step=False,
                    prog_bar=False,
                    logger=True,
                )
                self.log(
                    "val_attack_ratio",
                    att_ratio,
                    on_step=False,
                    prog_bar=False,
                    logger=True,
                )

        self.log("val_loss", loss, on_step=False, prog_bar=False, logger=True)
        self.log("val_acc", accuracy, on_step=False, prog_bar=False, logger=True)

        return {"loss": loss, "acc": accuracy, "pred": pred, "labels": labels}

    def validation_epoch_end(self, validation_step_outputs):
        y_true, y_pred = [], []
        for out in validation_step_outputs:
            for p, l in zip(out["pred"], out["labels"]):
                p = p[l > -1]
                l = l[l > -1]
                y_pred.extend(p.tolist())
                y_true.extend(l.tolist())

        if self.task == "binary":
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary"
            )
            self.log("val_link_f1", f1, on_epoch=True, logger=True)
            self.log("val_link_prec", prec, on_epoch=True, logger=True)
            self.log("val_link_rec", rec, on_epoch=True, logger=True)
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro"
            )
            self.log("val_macro_f1", f1, on_epoch=True, logger=True)
            self.log("val_macro_prec", prec, on_epoch=True, logger=True)
            self.log("val_macro_rec", rec, on_epoch=True, logger=True)

    def test_epoch_end(self, test_step_outputs):
        LABEL_NAMES = ["no-rel", "support", "attack"]
        y_true, y_pred = [], []
        total_results = dict()  # doc -> [tail, head] -> (pred, label)
        for out in test_step_outputs:
            for p, l in zip(out["pred"], out["labels"]):
                p = p[l > -1]
                l = l[l > -1]
                y_pred.extend(p.tolist())
                y_true.extend(l.tolist())

            for k, v in out["results"].items():
                doc_id, tail, head = k
                if doc_id not in total_results:
                    total_results[doc_id] = dict()
                total_results[doc_id][(tail, head)] = (
                    LABEL_NAMES[v[0]],
                    LABEL_NAMES[v[1]],
                )

        # log results to disk
        output_path = f"outputs/{self.exp_name}.jsonl"
        if not os.path.exists("outputs/"):
            os.makedirs("./outputs/")
        fout = open(output_path, "w")
        for doc, pairs in total_results.items():
            _pairs = [
                {"tail": tail, "head": head, "prediction": p, "label": l}
                for ((tail, head), (p, l)) in pairs.items()
            ]
            fout.write(json.dumps({"doc_id": doc, "candidates": _pairs}) + "\n")
        fout.close()
        if self.task == "binary":
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary"
            )
            self.log("test_link_f1", f1, on_epoch=True, logger=True)
            self.log("test_link_prec", prec, on_epoch=True, logger=True)
            self.log("test_link_rec", rec, on_epoch=True, logger=True)
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro"
            )
            self.log("test_macro_f1", f1, on_epoch=True, logger=True)
            self.log("test_macro_prec", prec, on_epoch=True, logger=True)
            self.log("test_macro_rec", rec, on_epoch=True, logger=True)

    def training_epoch_end(self, outputs) -> None:
        y_true, y_pred = [], []
        for out in outputs:
            for p, l in zip(out["pred"], out["labels"]):
                p = p[l > -1]
                l = l[l > -1]
                y_pred.extend(p.tolist())
                y_true.extend(l.tolist())

        if self.task == "binary":
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary"
            )
            self.log("train_link_f1", f1, on_epoch=True, logger=True)
            self.log("train_link_prec", prec, on_epoch=True, logger=True)
            self.log("train_link_rec", rec, on_epoch=True, logger=True)
            self.log(
                "train_link_pos_ratio",
                sum(y_pred) / len(y_pred),
                on_epoch=True,
                logger=True,
            )
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro"
            )
            self.log("train_macro_f1", f1, on_epoch=True, logger=True)
            self.log("train_macro_prec", prec, on_epoch=True, logger=True)
            self.log("train_macro_rec", rec, on_epoch=True, logger=True)
            supp_ratio = y_pred.count(1) / len(y_pred)
            att_ratio = y_pred.count(2) / len(y_pred)
            self.log("train_support_ratio", supp_ratio, on_epoch=True, logger=True)
            self.log("train_attack_ratio", att_ratio, on_epoch=True, logger=True)
            self.log(
                "train_link_ratio", supp_ratio + att_ratio, on_epoch=True, logger=True
            )

    def get_dataloader(self, set_type, shuffle):
        dataset = ArgumentRelationDataset(
            dataset_name=self.dataset_name,
            datadir=self.datadir,
            set_type=set_type,
            tokenizer=self.tokenizer,
            end_to_end=self.end_to_end,
            window_size=self.window_size,
            seed=self.seed,
            sampled_ids=self.sampled_ids if set_type == "train" else None,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collater,
            shuffle=shuffle,
            num_workers=0,
        )
        return dataloader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader(set_type="val", shuffle=False)

    def test_dataloader(self, test_set="test", use_pipeline=False):
        return self.get_dataloader(set_type=test_set, shuffle=False)

    def total_steps(self):
        return (self.dataset_size / self.hparams.batch_size) * self.hparams.max_epochs

    def setup(self, stage):
        self.train_loader = self.get_dataloader("train", shuffle=True)
        self.dataset_size = len(self.train_loader.dataset)

    def get_lr_scheduler(self):
        if self.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.opt,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps(),
            )
        else:
            scheduler = get_constant_schedule_with_warmup(
                self.opt,
                num_warmup_steps=self.warmup_steps,
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        model = self.model
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters()], "weight_decay": 0.0}
        ]
        print(
            f'{len(optimizer_grouped_parameters[0]["params"])} parameters will be trained'
        )

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]
