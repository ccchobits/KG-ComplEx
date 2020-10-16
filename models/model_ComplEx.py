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

        self.all_params = [self.ent_re_embed, self.ent_im_embed, self.rel_re_embed, self.rel_im_embed]
        self.initialize()

    def initialize(self):
        for emb in self.all_params:
            nn.init.xavier_normal_(emb)

        self.rel_re_embed.weight.data = F.normalize(self.rel_re_embed.weight.data, dim=1)
        self.rel_im_embed.weight.data = F.normalize(self.rel_im_embed.weight.data, dim=1)

    def get_score(self, heads, tails, rels, clamp=True):
        score = torch.sum(self.rel_re_embed(rels) * self.ent_re_embed(heads) * self.ent_re_embed(tails), dim=-1) \
            + torch.sum(self.rel_re_embed(rels) * self.ent_im_embed(heads) * self.ent_im_embed(tails), dim=-1) \
            + torch.sum(self.rel_im_embed(rels) * self.ent_re_embed(heads) * self.ent_im_embed(tails), dim=-1) \
            - torch.sum(self.rel_im_embed(rels) * self.ent_im_embed(heads) * self.ent_re_embed(tails), dim=-1)
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

