"""Active Learning routines that select next batch of unlabeled data for annotation."""
import argparse
import glob
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer

from argument_relation_transformer.coreset import CoresetSampler
from argument_relation_transformer.dataset import ArgumentRelationDataset
from argument_relation_transformer.system import ArgumentRelationClassificationSystem
from argument_relation_transformer.utils import AL_METHODS, move_to_cuda, set_seeds


class AL:
    """Active Learning class that implements various acqusition methods."""

    def __init__(self, args, existing_samples):
        self.dataset = args.dataset
        self.datadir = args.datadir
        self.method = args.method
        if self.method == "vocab":
            self.vocab_path = args.vocab_path

        self.interval = args.interval
        self.existing_samples = existing_samples
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.huggingface_path = args.huggingface_path

    def acquire(self, model):
        if self.huggingface_path is not None:
            tokenizer_name_or_path = self.huggingface_path + "roberta-base/"
        else:
            tokenizer_name_or_path = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name_or_path)
        dataset = ArgumentRelationDataset(
            dataset_name=self.dataset,
            datadir=self.datadir,
            set_type="train",
            window_size=20,
            tokenizer=tokenizer,
            seed=self.seed,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collater,
            shuffle=False,
            num_workers=0,
        )

        print(
            f"------------------ SELECTION METHOD: {self.method.upper()} ---------------------"
        )
        if self.method == "disc":
            return self.acquire_discourse_marker(dataloader)
        elif self.method == "no-disc":
            return self.acquire_discourse_marker(dataloader, True)
        elif self.method == "distance":
            return self.acquire_distance(dataloader)
        elif self.method == "vocab":
            return self.acquire_vocab(dataloader)
        elif self.method == "random" or model is None:
            return self.acquire_random(dataloader)
        elif self.method == "max-entropy":
            return self.acquire_entropy(dataloader, model)
        elif self.method == "bald":
            return self.acquire_bald(dataloader, model)
        elif self.method == "coreset":
            return self.acquire_coreset(dataloader, model)

    def acquire_vocab(self, dataloader):
        """Sample propositions based on the novelty of vocabulary.

        Score(prop) = \sum_w {freq}
        """
        from nltk.tokenize import word_tokenize

        vocab_idf = dict()
        for ln in open(self.vocab_path):
            word, df = ln.strip().split("\t")
            vocab_idf[word] = 1 / int(df)

        def _score_proposition(input_str):
            score = 0
            for w in word_tokenize(input_str):
                w = w.lower()
                if str.isalpha(w):
                    w_idf = vocab_idf[w] if w in vocab_idf else 0.0
                    w_score = 1 / (1 + w_idf)
                    score += w_score
            return score

        id2score = dict()
        for (i, batch) in tqdm(
            enumerate(dataloader), desc="sampling using vocabulary novelty"
        ):
            for (j, (_id, _labels)) in enumerate(zip(batch["ids"], batch["labels"])):
                _labels = _labels[_labels > -1]
                for k in range(len(_labels)):
                    new_id = list(_id)
                    new_id.append(k)
                    new_id = (tuple(new_id), _labels[k].item())

                    if new_id in self.existing_samples:
                        continue
                    if "forward" in new_id[0]:
                        cur_input_str = batch["input_str"][j][k + 1]
                    else:
                        cur_input_str = batch["input_str"][j][k]

                    cur_score = _score_proposition(cur_input_str)
                    id2score[new_id] = cur_score
        selected = []
        for item in sorted(id2score.items(), key=lambda x: x[1], reverse=True):
            selected.append(item[0])
            if len(selected) == self.interval:
                break
        self.existing_samples.extend(selected)
        all_labels = [item[1] for item in self.existing_samples]
        label_dist = Counter(all_labels)

        return self.existing_samples, label_dist

    def acquire_distance(self, dataloader):
        """Sample propositions based on their distance to the target, using a Possion distribution."""
        bins = list(range(1, 21))
        current_round_samples = [[] for _ in bins]

        for (i, batch) in tqdm(enumerate(dataloader), desc="sampling using distance"):
            for (j, (_id, _labels)) in enumerate(zip(batch["ids"], batch["labels"])):
                _labels = _labels[_labels > -1]
                for k in range(len(_labels)):
                    new_id = list(_id)
                    new_id.append(k)
                    new_id = (tuple(new_id), _labels[k].item())

                    if new_id in self.existing_samples:
                        continue

                    if "backward" in new_id[0]:
                        cur_distance = len(_labels) - k
                    else:
                        cur_distance = k + 1
                    cur_bin = cur_distance - 1
                    current_round_samples[cur_bin].append(new_id)
        possion_samples = np.random.poisson(lam=4, size=self.interval)
        bin_dist = Counter(possion_samples)
        extra = 0

        selected = []
        for ix in range(len(bins)):
            count = bin_dist[ix]
            cur_pool = current_round_samples[ix]
            if count > len(cur_pool):
                cur_sample = cur_pool
                extra += count - len(cur_pool)
            else:
                cur_sample = random.sample(cur_pool, k=count)
            current_round_samples[ix] = [_i for _i in cur_pool if _i not in cur_sample]
            for item in cur_sample:
                selected.append(item)

        last_extra = 0
        if extra > 0:
            print(
                f"{extra} extra need to be sampled by lambda=2, step={len(self.existing_samples)}"
            )
            extra_poisson_samples = np.random.poisson(lam=2, size=extra)
            extra_bin_dist = Counter(extra_poisson_samples)
            for ix in range(len(bins)):
                count = extra_bin_dist[ix]
                if count >= len(current_round_samples[ix]):
                    cur_sample = current_round_samples[ix]
                    last_extra += count - len(current_round_samples)
                else:
                    cur_sample = random.sample(current_round_samples[ix], k=count)

                cur_pool = current_round_samples[ix]
                current_round_samples[ix] = [
                    _i for _i in cur_pool if _i not in cur_sample
                ]

                for item in cur_sample:
                    selected.append(item)
        if last_extra > 0:

            total = []
            for b in current_round_samples:
                total.extend(b)
            print(
                f"{last_extra} needs to be sampled from the remaining, using random, sampling from the modified universe of {len(total)} samples"
            )
            cur_sample = random.sample(total, k=last_extra)
            selected.extend(cur_sample)

        self.existing_samples.extend(selected)
        all_labels = [item[1] for item in self.existing_samples]
        label_dist = Counter(all_labels)
        return self.existing_samples, label_dist

    def acquire_random(self, dataloader):

        unsampled = []
        for i, batch in tqdm(enumerate(dataloader)):
            for j, (_id, _label) in enumerate(zip(batch["ids"], batch["labels"])):

                # _id: (doc_id, tgt_id, direction)
                # _label: [ 0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                _label = _label[_label > -1]

                for k in range(len(_label)):
                    cur_label = _label[k].item()
                    new_id = list(_id)
                    new_id.append(k)
                    new_id = (tuple(new_id), cur_label)
                    if new_id in self.existing_samples:
                        continue

                    unsampled.append(new_id)

        if len(unsampled) <= self.interval:
            selected_tuples = unsampled
        else:
            selected_tuples = random.sample(unsampled, k=self.interval)

        self.existing_samples.extend(selected_tuples)
        all_labels = [item[1] for item in self.existing_samples]
        label_dist = Counter(all_labels)

        return self.existing_samples, label_dist

    def acquire_entropy(self, dataloader, model):
        scores = dict()  # id -> score
        label_dist = Counter()
        for i, batch in tqdm(enumerate(dataloader)):
            labels = batch["labels"]
            batch_ids = batch.pop("ids")
            batch.pop("input_str")
            batch.pop("disc_token_mask")
            batch = move_to_cuda(batch)
            logits, loss = model.model(**batch)
            categorical = Categorical(logits=logits)
            entropies = categorical.entropy()

            for j, (_id, _label) in enumerate(zip(batch_ids, batch["labels"])):
                sample_entropy = entropies[j]
                sample_entropy = sample_entropy[_label > -1]
                _label = _label[_label > -1]

                for (k, ent) in enumerate(sample_entropy):
                    new_id = list(_id)
                    new_id.append(k)
                    cur_label = _label[k].item()
                    new_id = (tuple(new_id), cur_label)

                    if new_id in self.existing_samples:
                        continue

                    scores[new_id] = ent.item()

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        unsampled = []
        for item in sorted_scores[: self.interval]:
            unsampled.append(item[0])

        self.existing_samples.extend(unsampled)
        all_labels = [item[1] for item in self.existing_samples]
        label_dist = Counter(all_labels)

        return self.existing_samples, label_dist

    @torch.no_grad()
    def acquire_coreset(self, dataloader, model):
        """
        Sample using coreset sampling
        """
        sampler = CoresetSampler(dataloader=dataloader, model=model.model)
        selected = sampler.select_batch_(self.existing_samples)
        self.existing_samples.extend(selected)

        all_labels = [item[1] for item in self.existing_samples]
        label_dist = Counter(all_labels)
        return self.existing_samples, label_dist

    @torch.no_grad()
    def acquire_bald(self, dataloader, model):
        """Run BALD based sampling, code adapted from:

        https://github.com/siddk/vqa-outliers/blob/main/active.py#L728

        Need to run Monte-Carlo dropout in train mode; collect disagreement of k different forward passes
        """
        import numpy as np
        import torch
        from scipy.stats import entropy

        model.train()

        def mc_step(batch, k=10):
            bsz = batch["labels"].shape[0]
            probs, disagreements = [], []
            with torch.no_grad():
                for _ in range(k):
                    logits, _ = model.model(**batch)
                    prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                    prob = prob.reshape(-1, prob.shape[-1])  # [16, 21, 3] -> [16*21, 3]
                    probs.append(prob)
                    disagreements.append(entropy(prob.transpose(1, 0)))

            entropies = entropy(np.mean(probs, axis=0).transpose(1, 0))
            disagreements = np.mean(disagreements, axis=0)
            diff = entropies - disagreements
            return diff.reshape(bsz, -1)

        id_list = []
        score_list = []
        for (i, batch) in tqdm(enumerate(dataloader)):
            batch = move_to_cuda(batch)
            info = mc_step(batch, k=10)  # info: [bsz, window]
            labels = batch["labels"].detach().cpu().numpy()

            for j, (_id, _label) in enumerate(zip(batch["ids"], labels)):
                cur_info = info[j][_label > -1]
                _label = _label[_label > -1]

                for k, _info in enumerate(cur_info):
                    new_id = list(_id)
                    new_id.append(k)
                    cur_label = _label[k].item()
                    new_id = (tuple(new_id), cur_label)
                    if new_id in self.existing_samples:
                        continue

                    id_list.append(new_id)
                    score_list.append(_info)
        added_ids = [
            id_list[x]
            for x in np.argpartition(score_list, -self.interval)[-self.interval :]
        ]
        self.existing_samples.extend(added_ids)
        all_labels = [item[1] for item in added_ids]
        label_dist = Counter(all_labels)

        return self.existing_samples, label_dist

    def acquire_discourse_marker(self, dataloader, exclude_marker=False):
        DISC_MARKER = [
            "because",
            "however",
            "therefore",
            "although",
            "though",
            "nevertheless",
            "nonetheless",
            "thus",
            "hence",
            "consequently",
            "for this reason",
            "due to",
            "in particular",
            "particularly",
            "specifically",
            "but",
            "in fact",
            "actually",
        ]

        unsampled_to_keep = []
        unsampled_to_discard = []
        for (i, batch) in tqdm(enumerate(dataloader)):
            for j, (_id, _label) in enumerate(zip(batch["ids"], batch["labels"])):
                _label = _label[_label > -1]
                for k in range(len(_label)):
                    new_id = list(_id)
                    new_id.append(k)
                    cur_label = _label[k].item()
                    new_id = (tuple(new_id), cur_label)
                    if new_id in self.existing_samples:
                        continue
                    if "forward" in new_id[0]:
                        cur_input_str = batch["input_str"][j][k + 1].lower()
                    else:
                        cur_input_str = batch["input_str"][j][k].lower()

                    contains_disc = False
                    for w in DISC_MARKER:
                        if w in cur_input_str:
                            contains_disc = True
                            break

                    if exclude_marker:
                        if contains_disc:
                            unsampled_to_discard.append(new_id)
                        else:
                            unsampled_to_keep.append(new_id)
                    else:
                        if contains_disc:
                            unsampled_to_keep.append(new_id)
                        else:
                            unsampled_to_discard.append(new_id)

        if self.interval > len(unsampled_to_keep):
            selected = unsampled_to_keep
            additional_count = self.interval - len(unsampled_to_keep)
            selected_extra = random.sample(unsampled_to_discard, k=additional_count)
            selected.extend(selected_extra)
        else:
            selected = random.sample(unsampled_to_keep, k=self.interval)

        self.existing_samples.extend(selected)
        all_labels = [item[1] for item in self.existing_samples]
        label_dist = Counter(all_labels)

        return self.existing_samples, label_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--datadir", type=str, default="./data/")
    parser.add_argument("--ckptdir", type=str, default="checkpoints/")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=AL_METHODS)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument(
        "--current-sample-size",
        type=int,
        required=True,
        help="The number of samples that's annotated now (before AL).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=500,
        help="The number of samples to be selected for annotation.",
    )
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument(
        "--huggingface-path",
        type=str,
        default=None,
        help="Path to local copy of the huggingface model, if not specified, will attempt to download remotely",
    )

    parser.add_argument(
        "--vocab-path",
        type=str,
        help="Path to the vocabulary, only required when `--method` set to `vocab`",
    )
    args = parser.parse_args()

    set_seeds(args.seed)

    output_dir = os.path.join(args.ckptdir, args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.current_sample_size == 0:
        # this is the first iteration, we have no model and no existing samples
        model = None
        existing_samples = []

    else:
        if args.method in ["random", "disc", "distance", "vocab", "no-disc"]:
            # methods that do not require model
            model = None

        else:
            # load model from previous round
            model_dir = os.path.join(
                args.ckptdir,
                f"{args.exp_name}_model-trained-on-{args.current_sample_size}/",
            )
            model_path = glob.glob(model_dir + "*.ckpt")
            assert len(model_path) == 1, model_dir
            model_path = model_path[0]
            print(f"loading model from {model_path}")
            model = ArgumentRelationClassificationSystem.load_from_checkpoint(
                checkpoint_path=model_path
            )
            model.cuda()
            model.eval()

        prev_output_path = os.path.join(
            output_dir, f"{args.method}.{args.current_sample_size}.jsonl"
        )
        existing_samples = []
        for ln in open(prev_output_path):
            _id, _label = json.loads(ln)
            _id = tuple(_id)
            existing_samples.append((_id, _label))

    print(f"{len(existing_samples)} existing samples loaded")

    al = AL(args, existing_samples)
    selected, label_dist = al.acquire(model)

    output_path = os.path.join(
        output_dir, f"{args.method}.{args.current_sample_size+args.interval}.jsonl"
    )
    with open(output_path, "w") as fout:
        for ln in selected:
            fout.write(json.dumps(ln) + "\n")

    # save ratio of labels for analysis
    label_ratio_output_path = output_path.replace(".jsonl", ".label_ratio.jsonl")
    with open(label_ratio_output_path, "w") as fout_log:
        fout_log.write(json.dumps(label_dist))


if __name__ == "__main__":
    main()
