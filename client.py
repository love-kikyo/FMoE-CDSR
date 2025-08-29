# -*- coding: utf-8 -*-
import os
import gc
import copy
import logging
import numpy as np
import torch
from dataloader import SeqDataloader
from utils.io_utils import ensure_dir


class Client:
    def __init__(self, model_fn, c_id, args, adj, train_dataset, valid_dataset, test_dataset, num_domains):
        # Used for computing the mask in self-attention module
        self.num_items = train_dataset.num_items
        self.c_id = c_id
        self.domain = train_dataset.domain
        self.num_domains = num_domains
        # Used for computing the positional embeddings
        self.max_seq_len = args.max_seq_len
        self.trainer = model_fn(args, self.c_id,
                                self.max_seq_len, num_domains, self.num_items)
        self.model = self.trainer.model
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = args.id if len(args.id) > 1 else "0" + args.id
        self.args = args
        self.adj = adj

        self.train_dataloader = SeqDataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = SeqDataloader(
            valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = SeqDataloader(
            test_dataset, batch_size=args.batch_size, shuffle=False)

        # Compute the number of samples for each client
        self.n_samples_train = len(train_dataset)
        self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)
        # The aggretation weight
        self.train_pop, self.valid_weight, self.test_weight = 0.0, 0.0, 0.0
        # Model evaluation results
        self.metrics_list = [
            {
                'MRR': 0.0,
                'NDCG_5': 0.0,
                'NDCG_10': 0.0,
                'HR_1': 0.0,
                'HR_5': 0.0,
                'HR_10': 0.0
            }
            for i in range(self.num_domains)
        ]

        self.metrics_shared = {
            'MRR': 0.0,
            'NDCG_5': 0.0,
            'NDCG_10': 0.0,
            'HR_1': 0.0,
            'HR_5': 0.0,
            'HR_10': 0.0
        }

    def train_epoch(self, round, args):
        """Trains one client with its own training data for one epoch.

        Args:
            round: Training round.
            args: Other arguments for training.
        """
        self.trainer.model.train()
        for _ in range(args.local_epoch):
            loss = 0
            step = 0
            for _, sessions in self.train_dataloader:
                if args.mu:
                    batch_loss = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args)
                else:
                    batch_loss = self.trainer.train_batch(
                        sessions, self.adj, self.num_items, args)
                loss += batch_loss
                step += 1

            gc.collect()
        logging.info("Epoch {}/{} - client {} -  Training Loss: {:.3f}".format(
            round, args.epochs, self.c_id, loss / step))
        return self.n_samples_train

    def evaluation(self, mode="valid"):
        """Evaluates one client with its own valid/test data for one epoch.

        Args:
            mode: `valid` or `test`.
        """
        if mode == "valid":
            dataloader = self.valid_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader

        self.trainer.model.eval()
        self.trainer.model.graph_convolution(self.adj)
        pred_list = [[] for i in range(self.num_domains)]
        pred_shared = []
        for _, sessions in dataloader:
            predictions_list, predictions_shared = self.trainer.test_batch(
                sessions)
            for i in range(self.num_domains):
                pred_list[i] = pred_list[i] + predictions_list[i]
            pred_shared = pred_shared + predictions_shared

        gc.collect()
        for i in range(self.num_domains):
            self.metrics_list[i]['MRR'], self.metrics_list[i]['NDCG_5'], self.metrics_list[i]['NDCG_10'], self.metrics_list[i]['HR_1'], self.metrics_list[i]['HR_5'], self.metrics_list[i]['HR_10'], \
                = self.cal_test_score(pred_list[i])
        self.metrics_shared['MRR'], self.metrics_shared['NDCG_5'], self.metrics_shared['NDCG_10'], self.metrics_shared['HR_1'], self.metrics_shared['HR_5'], self.metrics_shared['HR_10'], \
            = self.cal_test_score(pred_shared)
        return self.metrics_list, self.metrics_shared

    def get_old_eval_log(self):
        return self.metrics_list, self.metrics_shared

    @ staticmethod
    def cal_test_score(predictions):
        MRR = 0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        # `pred` indicates the rank of groundtruth items in the recommendation
        # list
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
        return MRR/valid_entity, NDCG_5 / valid_entity, \
            NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / \
            valid_entity, HR_10 / valid_entity

    def get_params(self):
        return copy.deepcopy(self.model.encoder_list[self.c_id].state_dict())

    def set_shared_params(self, model_shared_params):
        for id, shared_params in model_shared_params.items():
            if id != self.c_id:
                self.model.encoder_list[id].load_state_dict(shared_params)

    def save_params(self):
        method_ckpt_path = os.path.join(self.checkpoint_dir,
                                        "domain_" +
                                        "".join([domain[0]
                                                for domain
                                                 in self.args.domains]),
                                        "FMoE_DCSR_" + self.model_id)
        ensure_dir(method_ckpt_path, verbose=True)
        ckpt_filename = os.path.join(
            method_ckpt_path, "client%d.pt" % self.c_id)
        params = self.trainer.model.state_dict()
        try:
            torch.save(params, ckpt_filename)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(self.checkpoint_dir,
                                     "domain_" +
                                     "".join([domain[0]
                                             for domain in self.args.domains]),
                                     "FMoE_DCSR_" + self.model_id,
                                     "client%d.pt" % self.c_id)
        try:
            checkpoint = torch.load(ckpt_filename)
        except IOError:
            print("[ Fail: Cannot load model from {}. ]".format(ckpt_filename))
            exit(1)
        if self.trainer.model is not None:
            self.trainer.model.load_state_dict(checkpoint)
