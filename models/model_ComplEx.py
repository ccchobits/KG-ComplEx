import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class ComplEx(nn.Module):
    def __init__(self, configs, n_ent, n_rel):
        super(ComplEx, self).__init__()

        self.depth = configs.dim
        self.reg = configs.reg
        self.ent_re_embed = nn.Embedding(n_ent, self.depth)
        self.ent_im_embed = nn.Embedding(n_ent, self.depth)
        self.rel_re_embed = nn.Embedding(n_rel, self.depth)
        self.rel_im_embed = nn.Embedding(n_rel, self.depth)

        self.dropout_ent_re = nn.Dropout(configs.dropout)
        self.dropout_ent_im = nn.Dropout(configs.dropout)
        self.dropout_rel_re = nn.Dropout(configs.dropout)
        self.dropout_rel_im = nn.Dropout(configs.dropout)

        self.all_params = [self.ent_re_embed, self.ent_im_embed, self.rel_re_embed, self.rel_im_embed]

    def initialize(self):
        for emb in self.all_params:
            nn.init.xavier_normal_(emb.weight)

        self.rel_re_embed.weight.data = F.normalize(self.rel_re_embed.weight.data, dim=1)
        self.rel_im_embed.weight.data = F.normalize(self.rel_im_embed.weight.data, dim=1)

    def get_score(self, heads, tails, rels, clamp=True):
        heads_re = self.dropout_ent_re(self.ent_re_embed(heads))
        heads_im = self.dropout_ent_im(self.ent_im_embed(heads))
        tails_re = self.dropout_ent_re(self.ent_re_embed(tails))
        tails_im = self.dropout_ent_im(self.ent_im_embed(tails))
        rels_re = self.dropout_rel_re(self.rel_re_embed(rels))
        rels_im = self.dropout_rel_im(self.rel_im_embed(rels))

        score = torch.sum(rels_re * heads_re * tails_re, dim=-1) + torch.sum(rels_re * heads_im * tails_im, dim=-1) \
            + torch.sum(rels_im * heads_re * tails_im, dim=-1) - torch.sum(rels_im * heads_im * tails_re, dim=-1)
        if clamp:
            score = torch.clamp(score, -20, 20)
        return score

    def forward(self, x, labels):
        self.ent_re_embed.weight.data = F.normalize(self.ent_re_embed.weight.data, dim=1)
        self.ent_im_embed.weight.data = F.normalize(self.ent_im_embed.weight.data, dim=1)

        # shape: (batch_size,)
        heads, tails, rels = x[:, 0], x[:, 1], x[:, 2]
        scores = self.get_score(heads, tails, rels)

        if self.reg == 0.:
            return F.softplus(-labels * scores).mean()
        return F.softplus(-labels * scores).mean() + self.reg * self.get_regularization()

    def get_regularization(self):
        penalty = 0
        for param in self.all_params:
            penalty += torch.sum(param.weight.data ** 2) / 2.
        return penalty 

