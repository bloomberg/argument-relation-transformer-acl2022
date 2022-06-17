"""Data loading and processing."""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class ArgumentRelationDataset(Dataset):
    """Dataset class that creates context window and batchify the dataset."""

    def __init__(
        self,
        dataset_name,
        datadir,
        set_type,
        tokenizer,
        end_to_end=False,
        window_size=20,
        seed=42,
        sampled_ids=None,
    ):
        """
        Args:
            dataset_name (str): dataset domain, e.g., `ampere`, `ukp`.
            datadir (str): path to where the dataset jsonl files are stored
            set_type (str): one of `train`, `val`, `test`
            tokenizer (transformers.Tokenizer): the tokenizer used to map text to symbols
            end_to_end (bool): if set to True, head propositions are not given
            window_size (int): the number of propositions encoded to the left and right
            seed (int): random seed
            sampled_ids (list): list of sample ids that are to be included, use None if all should be included. This is not None only for active learning setting.

        """
        super().__init__()
        self.dataset_name = dataset_name
        self.datadir = datadir
        self.set_type = set_type
        self.window_size = window_size
        self.end_to_end = end_to_end
        self.tokenizer = tokenizer
        self.seed = seed
        random.seed(seed)

        self.disc_token_id = tokenizer.cls_token_id

        self.ID = []
        self.input_str = []
        self.input_ids = []
        self.labels = []

        # statistics to print
        self.supp_cnt = 0
        self.att_cnt = 0
        self.no_rel_cnt = 0

        self.skipped_supp_label = 0
        self.skipped_att_label = 0

        self.sampled_ids = None
        if sampled_ids is not None:
            print("select subset by AL")
            self.sampled_ids = dict()
            for ln in sampled_ids:
                doc_head_id = tuple(ln[:-1])
                prop_id = ln[-1]
                if doc_head_id not in self.sampled_ids:
                    self.sampled_ids[doc_head_id] = []
                self.sampled_ids[doc_head_id].append(prop_id)

        self._load_data()

    def _load_data(self):
        """Load data split and report statistics"""
        path = os.path.join(self.datadir, f"{self.dataset_name}_{self.set_type}.jsonl")
        print(path)
        for ln in open(path):
            cur_obj = json.loads(ln)
            self._include_bidir_context_with_window_size(cur_obj)

        print("=" * 50)
        print(f"label distribution for ({self.set_type}):")
        total_cnt = self.supp_cnt + self.att_cnt + self.no_rel_cnt
        print(
            f"Support: {self.supp_cnt} ({100 * self.supp_cnt / total_cnt:.2f}%) ({self.skipped_supp_label} skipped due to context limit)",
            end="\t",
        )
        print(
            f"Attack {self.att_cnt} ({100 * self.att_cnt / total_cnt:.2f}%) ({self.skipped_att_label} skipped due to context limit)",
            end="\t",
        )
        print(f"No-rel: {self.no_rel_cnt} ({100 * self.no_rel_cnt / total_cnt:.2f}%)")
        print(f"{len(self.ID)} sequence loaded.")
        print(f"=" * 50)

    def _include_bidir_context_with_window_size(self, doc_obj):
        """Create actual training sample in sequences, by extracting context from left and right.

        Results are stored in the following lists:
            - self.input_ids : store the token ids of the input sequence
            - self.input_str : store the raw string of the input sequence
            - self.labels    : store the label id of the input sequence (0: no-rel, 1: support, 2: attack, -2: head itself)
            - self.ID
        """

        doc_id = doc_obj["doc_id"]
        cur_sents = doc_obj["text"]
        cur_toks = [
            self.tokenizer.encode(sent, add_special_tokens=False) for sent in cur_sents
        ]
        relation_list = doc_obj["relations"]
        head_to_tails = dict()  # head_id -> list of tail ids
        for item in relation_list:
            head = item["head"]
            tail = item["tail"]
            rel_type = item["type"]

            if head not in head_to_tails:
                head_to_tails[head] = []
            head_to_tails[head].append((tail, rel_type))

        for head in range(len(cur_toks)):

            if head in head_to_tails:
                tail_list = head_to_tails[head]

            elif self.end_to_end:
                # do not assume heads are given, include all propositions as potential head
                tail_list = []

            else:
                # assumes heads are given, therefore skip non-head cases
                continue

            self._extract_left_context(doc_id, cur_toks, cur_sents, head, tail_list)
            self._extract_right_context(doc_id, cur_toks, cur_sents, head, tail_list)
        return

    def _extract_right_context(self, doc_id, tokens, strs, head, tail_list):
        """Extract right context with window size, also truncate to at most 500 tokens (not counting disc_token)"""

        # right context is `forward`
        id_proposal = (doc_id, head, "forward")
        right_id = min(head + self.window_size, len(tokens) - 1)
        proposal_tokens = tokens[head : right_id + 1]
        proposal_str = strs[head : right_id + 1]
        labels = [0] * len(proposal_tokens)

        # use -2 to indicate head in `labels`
        labels[0] = -2

        for r_id, rel_type in tail_list:
            if r_id <= head:  # left context
                continue

            # the offset for prop that is immediately to the right of head is 1, head itself would be 0
            positive_offset = r_id - head
            if (
                positive_offset > len(labels) - 1
            ):  # too far right, exceeding window size
                continue

            if rel_type == "support":
                labels[positive_offset] = 1

            elif rel_type == "attack":
                labels[positive_offset] = 2

        cur_lens = sum([len(x) for x in proposal_tokens])
        while cur_lens > 500 - self.window_size:

            # shrink from the rightmost
            cur_lens -= len(proposal_tokens[-1])
            proposal_tokens = proposal_tokens[:-1]
            proposal_str = proposal_str[:-1]
            if labels[-1] == 1:
                self.skipped_supp_label += 1
            elif labels[-1] == 2:
                self.skipped_att_label += 1
            labels = labels[:-1]

        if len(labels) <= 1:
            return

        input_ids = []
        input_str = []
        for _ix, toks in enumerate(proposal_tokens):
            input_ids.append(self.disc_token_id)
            input_ids.extend(toks)
            input_str.append(proposal_str[_ix])

        skip = False
        if self.set_type == "train" and self.sampled_ids is not None:
            uncov_ids = (
                self.sampled_ids[id_proposal] if id_proposal in self.sampled_ids else []
            )
            masked_labels = []
            skip = True
            for i, l in enumerate(labels):
                if l == -2:
                    masked_labels.append(l)
                elif i in uncov_ids:  # this label is uncovered during active learning
                    masked_labels.append(l)
                    skip = False
                else:  # this label is not uncovered, so hidden as -1 (padding)
                    masked_labels.append(-1)
            labels = masked_labels

        if not skip:
            self.input_ids.append(input_ids)
            self.input_str.append(input_str)
            self.labels.append(labels)
            self.ID.append(id_proposal)

            # record statistics
            cur_supp = labels.count(1)
            cur_att = labels.count(2)
            self.supp_cnt += cur_supp
            self.att_cnt += cur_att
            self.no_rel_cnt += labels.count(0)

    def _extract_left_context(self, doc_id, tokens, strs, head, tail_list):
        """Extract left context with window size, also truncate to at most 500 tokens (not counting disc_token)"""

        # left context is `backward`
        id_proposal = (doc_id, head, "backward")
        left_id = max(0, head - self.window_size)
        proposal_tokens = tokens[left_id : head + 1]
        proposal_str = strs[left_id : head + 1]

        labels = [0] * len(proposal_tokens)
        for r_id, rel_type in tail_list:
            if r_id >= head:  # belongs to the right context
                continue

            # the offset for prop that is immediately to the left of head is -2, head itself would be -1
            negative_offset = r_id - head - 1
            if negative_offset < -1 * len(
                labels
            ):  # too far left, exceeding window size
                continue

            if rel_type == "support":
                labels[negative_offset] = 1

            elif rel_type == "attack":
                labels[negative_offset] = 2

        # make sure the entire sequence is within 500 token range
        cur_lens = sum([len(x) for x in proposal_tokens])
        while cur_lens > 500 - self.window_size:
            # shrink from the leftmost
            cur_lens -= len(proposal_tokens[0])
            proposal_tokens = proposal_tokens[1:]
            proposal_str = proposal_str[1:]
            if labels[0] == 1:
                self.skipped_supp_label += 1
            elif labels[0] == 2:
                self.skipped_att_label += 1
            labels = labels[1:]

        # only one proposition is left, likely because some proposition in this dataset is too long
        if len(labels) <= 1:
            return

        # if prop3 is head, encode as:
        # <s> prop1 </s> prop2 </s> prop3 </s>
        # use -2 to indicate the head proposition in `labels`
        labels = labels[:-1] + [-2]
        input_str = []
        input_ids = []
        for _ix, toks in enumerate(proposal_tokens):
            input_ids.append(self.disc_token_id)
            input_ids.extend(toks)
            input_str.append(proposal_str[_ix])

        skip = False
        if self.set_type == "train" and self.sampled_ids is not None:
            uncov_ids = (
                self.sampled_ids[id_proposal] if id_proposal in self.sampled_ids else []
            )
            masked_labels = []
            skip = True
            for i, l in enumerate(labels):
                if l == -2:
                    masked_labels.append(l)
                elif i in uncov_ids:  # this label is uncovered during active learning
                    masked_labels.append(l)
                    skip = False
                else:  # this label is not uncovered, so hidden as -1 (padding)
                    masked_labels.append(-1)
            labels = masked_labels

        if not skip:
            self.input_ids.append(input_ids)
            self.input_str.append(input_str)
            self.labels.append(labels)
            self.ID.append(id_proposal)
            cur_supp = labels.count(1)
            cur_att = labels.count(2)
            self.supp_cnt += cur_supp
            self.att_cnt += cur_att
            self.no_rel_cnt += labels.count(0)

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, index):
        result = {
            "id": self.ID[index],
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
            "input_str": self.input_str[index],
        }
        return result

    def collater(self, samples):
        """Consolidate list of samples into a batch"""
        batch = dict()
        batch["ids"] = [s["id"] for s in samples]
        batch["input_str"] = [s["input_str"] for s in samples]

        batch_size = len(samples)
        max_input_len = max([len(s["input_ids"]) for s in samples])
        input_ids = np.full(
            shape=[batch_size, max_input_len],
            fill_value=self.tokenizer.pad_token_id,
            dtype=np.int,
        )

        # 1 - [disc], 0 - otherwise
        disc_token_mask = np.full(
            shape=[batch_size, max_input_len], fill_value=0, dtype=np.int
        )

        max_sent_num = max([len(s["labels"]) for s in samples])
        sentence_boundary = np.full(
            shape=[batch_size, max_sent_num], fill_value=0, dtype=np.int
        )
        sentence_boundary_mask = np.zeros(
            shape=[batch_size, max_sent_num], dtype=np.int
        )

        # 0 - no-rel; 1 - support; 2 - attack; -1 - (pad, self)
        labels = np.full(shape=[batch_size, max_sent_num], fill_value=-1, dtype=np.int)
        target_ids = torch.zeros([batch_size, 1], dtype=torch.long)

        for ix, s in enumerate(samples):
            cur_input_ids = s["input_ids"]
            input_ids[ix][: len(cur_input_ids)] = cur_input_ids

            cur_labels = [x if x != -2 else -1 for x in s["labels"]]

            disc_token_pos = []
            cur_disc = cur_input_ids.index(self.disc_token_id)
            while cur_disc < len(cur_input_ids) - 1:
                disc_token_pos.append(cur_disc)
                disc_token_mask[ix, cur_disc] = 1
                try:
                    cur_disc = cur_input_ids.index(self.disc_token_id, cur_disc + 1)
                except ValueError:
                    break

            assert len(disc_token_pos) == len(cur_labels)

            sentence_boundary[ix][: len(disc_token_pos)] = disc_token_pos
            sentence_boundary_mask[ix][: len(disc_token_pos)] = 1

            labels[ix][: len(cur_labels)] = cur_labels
            target_ids[ix][0] = s["labels"].index(-2)

        batch["input_ids"] = torch.LongTensor(input_ids)
        assert (
            batch["input_ids"].shape[1] < 512
        ), f'size too large! {batch["input_ids"].shape}'
        batch["labels"] = torch.LongTensor(
            labels
        )  # 0 -> no-real, -1 -> attack, 1 -> support
        batch["sequence_boundary_ids"] = torch.LongTensor(sentence_boundary)
        batch["sequence_boundary_mask"] = torch.LongTensor(sentence_boundary_mask)
        batch["attention_mask"] = (
            batch["input_ids"] != self.tokenizer.pad_token_id
        ).long()
        batch["target_ids"] = target_ids
        batch["disc_token_mask"] = torch.LongTensor(disc_token_mask)
        return batch
