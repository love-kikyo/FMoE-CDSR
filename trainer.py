# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from models.moe_gsan_model import MoeGSAN
from models import config
from utils import train_utils
from losses import NCELoss, HingeLoss, JSDLoss, Discriminator


class Trainer(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def train_batch(self, *args, **kwargs):
        raise NotImplementedError

    def test_batch(self, *args, **kwargs):
        raise NotImplementedError

    def update_lr(self, new_lr):
        train_utils.change_lr(self.optimizer, new_lr)


class ModelTrainer(Trainer):
    def __init__(self, args, c_id, max_seq_len, num_domains, num_items):
        self.args = args
        self.num_domains = num_domains
        self.c_id = c_id
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.model = MoeGSAN(
            c_id, args, num_domains, num_items).to(self.device)
        self.discri = Discriminator(
            config.hidden_size, max_seq_len).to(self.device)

        self.bce_criterion = nn.BCEWithLogitsLoss(
            reduction="none").to(self.device)
        self.cs_criterion = nn.CrossEntropyLoss(
            reduction="none").to(self.device)
        self.cl_criterion = NCELoss(
            temperature=args.temperature).to(self.device)
        self.jsd_criterion = JSDLoss().to(self.device)
        self.hinge_criterion = HingeLoss(margin=0.3).to(self.device)

        self.params = list(self.model.parameters())
        params_to_remove = []
        for i in range(self.num_domains):
            if i != self.c_id:
                params_to_remove.extend(
                    list(self.model.encoder_list[i].parameters()))
        self.params = [param for param in self.params if all(
            id(param) != id(exclude_param) for exclude_param in params_to_remove)]

        self.optimizer = train_utils.get_optimizer(
            args.optimizer, self.params, args.lr)
        self.step = 0

    def train_batch(self, sessions, adj, num_items, args):
        """Trains the model for one batch.

        Args:
        sessions: Input user sequences.
        adj: Adjacency matrix of the local graph.
        num_items: Number of items in the current domain.
        args: Other arguments for training.
        """
        self.optimizer.zero_grad()

        # Here the items are first sent to GNN for convolution, and then
        # the resulting embeddings are sent to the self-attention module.
        # Note that each batch must be convolved once, and the
        # item_embeddings input to the convolution layer are updated from
        # the previous batch.
        self.model.graph_convolution(adj)

        sessions = [torch.LongTensor(x).to(self.device) for x in sessions]
        # seq: (batch_size, seq_len), ground: (batch_size, seq_len),
        # ground_mask:  (batch_size, seq_len),
        # js_neg_seqs: (batch_size, seq_len),
        # contrast_aug_seqs: (batch_size, seq_len)
        # Here `js_neg_seqs` is used for computing similarity loss,
        # `contrast_aug_seqs` is used for computing contrastive infomax
        # loss
        seq, ground, ground_mask, contrast_aug_seqs = sessions
        result_list, result_shared, \
            mu_list, logvar_list, z_list, aug_z_list = self.model(
                seq, aug_seqs=contrast_aug_seqs)

        loss = self.moe_gsan_loss_fn(result_list, result_shared,
                                        mu_list, logvar_list,
                                        ground, z_list,
                                        aug_z_list, ground_mask,
                                        num_items, self.step)

        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def moe_gsan_loss_fn(self, result_list, result_shared,
                          mu_list, logvar_list,
                          ground, z_list,
                          aug_z_list, ground_mask,
                          num_items, step):
        """Overall loss function of FMoE_CDSR (our method).
        """

        recons_loss_sum = 0
        for i in range(self.num_domains):
            recons_loss = self.cs_criterion(
                result_list[i].reshape(-1, num_items + 1),
                ground.reshape(-1))  # (batch_size * seq_len, )
            recons_loss = (recons_loss *
                           (ground_mask.reshape(-1))).mean()
            recons_loss_sum = recons_loss_sum + recons_loss

        recons_loss_shared = self.cs_criterion(
            result_shared.reshape(-1, num_items + 1),
            ground.reshape(-1))  # (batch_size * seq_len, )
        recons_loss_shared = (recons_loss_shared *
                              (ground_mask.reshape(-1))).mean()

        kld_loss_sum = 0
        for i in range(self.num_domains):
            kld_loss = -0.5 * \
                torch.sum(1 + logvar_list[i] - mu_list[i] ** 2 -
                          logvar_list[i].exp(), dim=-1).reshape(-1)
            kld_loss = (kld_loss * (ground_mask.reshape(-1))).mean()
            kld_loss_sum += kld_loss

        alpha = 1.0  # 1.0 for all scenarios

        kld_weight = self.kl_anneal_function(
            self.args.anneal_cap, step, self.args.total_annealing_step)

        beta = 1.0  # 1.0 for FKCB and BMG, 0.1 for SGH

        contrastive_loss_sum = 0
        for i in range(self.num_domains):
            user_representation1 = z_list[i][:, -1, :]
            user_representation2 = aug_z_list[i][:, -1, :]
            contrastive_loss = self.cl_criterion(
                user_representation1, user_representation2)
            contrastive_loss = contrastive_loss.mean()
            contrastive_loss_sum += contrastive_loss

        loss = alpha * ((recons_loss_sum + recons_loss_shared) +
                        kld_weight * kld_loss_sum) +\
            + beta * (contrastive_loss_sum)

        return loss

    def kl_anneal_function(self, anneal_cap, step, total_annealing_step):
        """
        step: increment by 1 for every forward-backward step.
        total annealing steps: pre-fixed parameter control the speed of
        anealing.
        """
        # borrows from https://github.com/timbmg/Sentence-VAE/blob/master/train.py
        return min(anneal_cap, step / total_annealing_step)

    @staticmethod
    def flatten(source):
        return torch.cat([value.flatten() for value in source])

    def prox_reg(self, params1, params2, mu):
        params1_values, params2_values = [], []
        # Record the model parameter aggregation results of each branch
        # separately
        for branch_params1, branch_params2 in zip(params1, params2):
            branch_params2 = [branch_params2[key]
                              for key in branch_params1.keys()]
            params1_values.extend(branch_params1.values())
            params2_values.extend(branch_params2)

        # Multidimensional parameters should be compressed into one dimension
        # using the flatten function
        s1 = self.flatten(params1_values)
        s2 = self.flatten(params2_values)
        return mu/2 * torch.norm(s1 - s2)

    def test_batch(self, sessions):
        """Tests the model for one batch.

        Args:
            sessions: Input user sequences.
        """
        sessions = [torch.LongTensor(x).to(self.device) for x in sessions]

        # seq: (batch_size, seq_len), ground_truth: (batch_size, ),
        # neg_list: (batch_size, num_test_neg)
        seq, ground_truth, neg_list = sessions
        # result: (batch_size, seq_len, num_items)
        result_list, result_shared = self.model(seq, self.step)

        pred_list = []
        pred_shared = []
        for result in result_list:
            pred = []
            for id in range(len(result)):
                # result[id, -1]: (num_items, )
                score = result[id, -1]
                cur = score[ground_truth[id]]
                # score_larger = (score[neg_list[id]] > (cur + 0.00001))\
                # .data.cpu().numpy()
                score_larger = (score[neg_list[id]] > (cur)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                pred.append(true_item_rank)
            pred_list.append(pred)

        for id in range(len(result_shared)):
            # result[id, -1]: (num_items, )
            score = result_shared[id, -1]
            cur = score[ground_truth[id]]
            # score_larger = (score[neg_list[id]] > (cur + 0.00001))\
            # .data.cpu().numpy()
            score_larger = (score[neg_list[id]] > (cur)).data.cpu().numpy()
            true_item_rank = np.sum(score_larger) + 1
            pred_shared.append(true_item_rank)

        return pred_list, pred_shared
