# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import SelfAttention
from .gnn import GCNLayer
from . import config
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_items, args, id):
        super(Encoder, self).__init__()
        self.encoder_mu = SelfAttention(num_items, args, id)
        self.encoder_logvar = SelfAttention(num_items, args, id)

    def forward(self, seqs, seqs_data):
        """
        seqs: (batch_size, seq_len, hidden_size)
        seqs_data: (batch_size, seq_len)
        """
        mu = self.encoder_mu(seqs, seqs_data)
        logvar = self.encoder_logvar(seqs, seqs_data)
        return mu, logvar


class Moe(nn.Module):
    def __init__(self, num_experts, input_dim):
        super(Moe, self).__init__()
        self.num_experts = num_experts

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size * 4),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size * 2),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size * 2, num_experts)
        )

    def forward(self, z_moe):
        z_moe_pool = z_moe.mean(dim=[0, 1])  # (input_dim)
        gate_logits = self.gate_net(z_moe_pool)  # (num_domains)
        gate_weights = F.softmax(gate_logits, dim=-1)
        return gate_weights  # (num_domains)


class MoeGSAN(nn.Module):
    def __init__(self, c_id, args, num_domains, num_items):
        super(MoeGSAN, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.num_domains = num_domains
        self.c_id = c_id
        # Item embeddings cannot be shared between clients, because the number
        # of items in each domain is different.
        self.item_emb_list = nn.ModuleList(
            [nn.Embedding(num_items + 1, config.hidden_size, padding_idx=num_items) for i in range(self.num_domains)])

        self.pos_emb_list = nn.ModuleList(
            [nn.Embedding(args.max_seq_len, config.hidden_size) for i in range(self.num_domains)])

        self.GNN_encoder_list = nn.ModuleList(
            [GCNLayer(args) for i in range(self.num_domains)])

        self.encoder_list = nn.ModuleList(
            [Encoder(num_items, args, i) for i in range(self.num_domains)])

        self.linear_list = nn.ModuleList([nn.Linear(
            config.hidden_size, num_items) for i in range(self.num_domains)])
        self.linear_pad_list = nn.ModuleList(
            [nn.Linear(config.hidden_size, 1) for i in range(self.num_domains)])

        self.LayerNorm_list = nn.ModuleList(
            [nn.LayerNorm(config.hidden_size, eps=1e-12) for i in range(self.num_domains)])
        self.dropout_list = nn.ModuleList(
            [nn.Dropout(config.dropout_rate) for i in range(self.num_domains)])

        self.moe = Moe(
            self.num_domains, config.hidden_size * self.num_domains)

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def graph_convolution(self, adj):
        self.item_graph_embs_list = []
        for i in range(self.num_domains):
            self.item_index = torch.arange(
                0, self.item_emb_list[i].num_embeddings, 1).to(self.device)
            item_embs = self.my_index_select_embedding(
                self.item_emb_list[i], self.item_index)
            self.item_graph_embs_list.append(
                self.GNN_encoder_list[i](item_embs, adj))

    def get_position_ids(self, seqs):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        return position_ids

    def add_position_embedding(self, seqs, seq_embeddings, c_id):
        position_ids = self.get_position_ids(seqs)
        position_embeddings = self.pos_emb_list[c_id](position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm_list[c_id](seq_embeddings)
        seq_embeddings = self.dropout_list[c_id](seq_embeddings)
        return seq_embeddings  # (batch_size, seq_len, hidden_size)

    def forward(self, seqs, aug_seqs=None):
        # `item_graph_embs` stores the embeddings of all items.
        # Here we need to select the embeddings of items appearing in the
        # sequence
        seqs_emb_list = []
        for i in range(self.num_domains):
            seqs_emb = self.my_index_select(
                self.item_graph_embs_list[i], seqs) + self.item_emb_list[i](seqs)
            # (batch_size, seq_len, hidden_size)
            seqs_emb *= self.item_emb_list[i].embedding_dim ** 0.5
            seqs_emb = self.add_position_embedding(
                seqs, seqs_emb, i)  # (batch_size, seq_len, hidden_size)
            seqs_emb_list.append(seqs_emb)

        # Here is a shortcut operation that adds up the embeddings of items
        # convolved by GNN and those that have not been convolved.
        if self.training:
            aug_seqs_emb_list = []
            for i in range(self.num_domains):
                aug_seqs_emb = self.my_index_select(
                    self.item_graph_embs_list[i], aug_seqs) + self.item_emb_list[i](aug_seqs)
                # (batch_size, seq_len, hidden_size)
                aug_seqs_emb *= self.item_emb_list[i].embedding_dim ** 0.5
                aug_seqs_emb = self.add_position_embedding(
                    aug_seqs, aug_seqs_emb, i)  # (batch_size, seq_len, hidden_size)
                aug_seqs_emb_list.append(aug_seqs_emb)

        mu_list = []
        logvar_list = []
        z_list = []
        for i in range(self.num_domains):
            mu, logvar = self.encoder_list[i](
                seqs_emb_list[i], seqs)
            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(mu)

        if self.training:
            aug_z_list = []
            for i in range(self.num_domains):
                aug_mu, aug_logvar = self.encoder_list[i](
                    aug_seqs_emb_list[i], aug_seqs)
                aug_z_list.append(aug_mu)

        result_list = []
        for i in range(self.num_domains):
            result = (self.linear_list[i](z_list[i]))
            result_pad = self.linear_pad_list[i](z_list[i])
            result_list.append(torch.cat((result, result_pad), dim=-1))

        z_moe_list = []
        for i in range(self.num_domains):
            z_moe_list.append(z_list[i].detach())

        z_moe = torch.cat(z_moe_list, dim=-1)
        result_weight = self.moe(z_moe)

        result_moe = torch.zeros_like(result_list[self.c_id])
        for i in range(self.num_domains):
            result_moe += result_weight[i] * result_list[i].detach()
                
        if self.training:
            return result_list, \
                result_moe, \
                mu_list, logvar_list, z_list, aug_z_list
        else:
            return result_list, \
                result_moe
