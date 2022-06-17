"""Helper class for the CoreSet active learning strategy."""

import torch
import torch.nn as nn
from tqdm import tqdm

from argument_relation_transformer.utils import move_to_cuda


class CoresetSampler:
    def __init__(self, dataloader, model):
        self.dataloader = dataloader
        self.model = model
        self.min_distances = None
        self.already_selected = []
        self.pdist = nn.PairwiseDistance(p=2)

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        adapted from https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
        """
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [
                d for d in cluster_centers if d not in self.already_selected
            ]
        if cluster_centers:
            x = self.all_features[0][cluster_centers]
            dist = torch.cdist(self.all_features, x).squeeze(0)

            if self.min_distances is None:
                self.min_distances, _ = torch.min(dist, axis=1)
                self.min_distances = self.min_distances.reshape(-1, 1)

            else:
                self.min_distances = torch.min(self.min_distances, dist)

    def select_batch_(self, already_selected):
        """Sample greedily to minimize the maximum distance to a cluster center
        among all unlabeled datapoints.
        """
        features = []
        sample_ix = 0
        already_selected_ix = []
        real_ids = []
        for (i, batch) in tqdm(
            enumerate(self.dataloader), desc=f"Sampling using coreset"
        ):
            batch = move_to_cuda(batch)
            # batch_size x prop_num x dim
            batch_feats = self.model.extract_last_layer(**batch)
            cur_labels = batch["labels"]
            for (j, _id) in enumerate(batch["ids"]):
                valid_prop_ix = 0
                for l_ix, l in enumerate(cur_labels[j]):
                    if l == -1:
                        continue

                    new_id = list(_id)
                    new_id.append(valid_prop_ix)
                    new_id = (tuple(new_id), l.item())
                    if new_id in already_selected:
                        already_selected_ix.append(sample_ix)
                    real_ids.append(new_id)
                    sample_ix += 1
                    valid_prop_ix += 1

                    cur_feat = batch_feats[j, l_ix].unsqueeze(0)
                    features.append(cur_feat)

        self.all_features = torch.cat(features, 0).unsqueeze(0)
        self.update_distances(already_selected_ix, only_new=False, reset_dist=True)
        self.already_selected = already_selected_ix

        selected = []
        for _ in range(500):
            ind = torch.argmax(self.min_distances)
            ind = ind.item()
            assert ind not in self.already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            selected.append(real_ids[ind])

        return selected
