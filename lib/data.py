import os
import json
import numpy as np
import torch

class SADataset():
    def __init__(self, path=None, split_idx_path=None, score_path=None):
        self.features_path = os.path.join(path, 'features.dat')
        self.labels_path = os.path.join(path, 'labels.dat')
        conf_path = os.path.join(path, 'conf.json')
        self.conf = json.load(open(conf_path, 'r'))

        split_idx = torch.load(split_idx_path)
        self.train_idx = split_idx['train']
        self.val_idx = split_idx['valid']
        self.test_idx = split_idx['test']

        self.score_path = score_path

        self.num_nodes = self.conf['num_nodes']
        self.num_features = self.conf['features_shape'][1]
        self.num_classes = self.conf['num_classes']


    def get_labels(self):
        labels = torch.from_numpy(np.fromfile(self.labels_path, dtype=self.conf['labels_dtype'], count=self.num_nodes).reshape(tuple([self.conf['labels_shape'][0]])))
        return labels


    def make_new_shuffled_train_idx(self):
        self.shuffled_train_idx = self.train_idx[torch.randperm(self.train_idx.numel())]



