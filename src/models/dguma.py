import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils.utils import *
from collections import defaultdict
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
import math
from scipy.sparse import lil_matrix
import random
import json
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class DGUMA(GeneralRecommender):
    def __init__(self, config, dataset, test_data,local_time):
        super().__init__(config, dataset, test_data,local_time)
        self.sparse = True
        self.bm_loss = config['bm_loss']
        self.um_loss = config['um_loss']
        self.vt_loss = config['vt_loss']
        self.reg_weight_1 = config['reg_weight_1']
        self.reg_weight_2 = config['reg_weight_2']
        self.bm_temp = config['bm_temp']
        self.um_temp = config['um_temp']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.torchBiAdj = self.makeTorchAdj(self.interaction_matrix)
        self.user_embedding = nn.Parameter(init(torch.empty(self.n_users, self.embedding_dim)))
        self.item_id_embedding = nn.Parameter(init(torch.empty(self.n_items, self.embedding_dim)))

        self.extended_image_user = nn.Parameter(init(torch.empty(self.n_users, self.embedding_dim)))
        self.extended_image_user_1 = nn.Parameter(init(torch.empty(self.n_users, self.embedding_dim)))
        self.extended_text_user = nn.Parameter(init(torch.empty(self.n_users, self.embedding_dim)))
        self.extended_text_user_1 = nn.Parameter(init(torch.empty(self.n_users, self.embedding_dim)))

        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.data_name = config['dataset']

        image_adj_file = os.path.join(self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        #  Enhancing User-Item Graph
        self.inter = self.find_inter(self.image_original_adj, self.text_original_adj)
        self.ii_adj = self.add_edge(self.inter)
        self.norm_adj = self.get_adj_mat(self.ii_adj.tolil())
        self.R1 = self.R.tocoo().astype(np.float32)
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(config['gnn_layer'])])
        trnData = TrnData(self.interaction_matrix, self.n_items)
        self.Diff_trnLoader = dataloader.DataLoader(trnData, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.diffusionData = DiffusionData(torch.FloatTensor(self.interaction_matrix.A))
        self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=self.batch_size, shuffle=True,
                                                     num_workers=0)

        self.image_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.image_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid(),
        )
        self.image_space_trans = nn.Sequential(
            self.image_reduce_dim,
            self.image_trans_dim
        )

        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_space_trans = nn.Sequential(
            self.text_reduce_dim,
            self.text_trans_dim
        )

        self.separate_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.image_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5
        self.edgeDropper = SpAdjDropEdge(config['keepRate'])

        if config['trans'] == 1:
            self.image_trans = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.embedding_dim),
            )
            self.text_trans = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.embedding_dim),
            )
        elif config['trans'] == 0:
            self.image_trans = nn.Parameter(init(torch.empty(size=(self.v_feat.shape[1], self.embedding_dim))))
            self.text_trans = nn.Parameter(init(torch.empty(size=((self.t_feat.shape[1], self.embedding_dim)))))
        elif config['trans'] == 2:
            self.image_trans = self.image_space_trans
            self.text_trans = self.text_space_trans
        elif config['trans'] == 3:
            self.image_trans = nn.Parameter(init(torch.empty(size=(self.v_feat.shape[1], self.embedding_dim))))
            self.text_trans = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        else:
            self.image_trans = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            self.text_trans = nn.Parameter(init(torch.empty(size=(self.t_feat.shape[1], self.embedding_dim))))

        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))

    def find_inter(self, image_adj, text_adj):
        inter_file = os.path.join(self.dataset_path, 'inter.json')
        if os.path.exists(inter_file):
            with open(inter_file) as f:
                inter = json.load(f)
        else:
            j = 0
            inter = defaultdict(list)
            img_sim = []
            txt_sim = []
            for i in range(0, len(image_adj._indices()[0])):
                img_id = image_adj._indices()[0][i]
                txt_id = text_adj._indices()[0][i]
                assert img_id == txt_id
                id = img_id.item()
                img_sim.append(image_adj._indices()[1][j].item())
                txt_sim.append(text_adj._indices()[1][j].item())

                if len(img_sim) == 10 and len(txt_sim) == 10:
                    it_inter = list(set(img_sim) & set(txt_sim))
                    inter[id] = [v for v in it_inter if v != id]
                    img_sim = []
                    txt_sim = []

                j += 1

            with open(inter_file, "w") as f:
                json.dump(inter, f)

        return inter



    def add_edge(self, inter):
        sim_rows = []
        sim_cols = []
        for id, vs in inter.items():
            if len(vs) == 0:
                continue
            for v in vs:
                sim_rows.append(int(id))
                sim_cols.append(v)

        sim_rows = torch.tensor(sim_rows)
        sim_cols = torch.tensor(sim_cols)
        sim_values = [1] * len(sim_rows)

        item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)), shape=(self.n_items, self.n_items), dtype=np.int)
        return item_adj

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self, item_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat[self.n_users:, self.n_users:] = item_adj
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            rowsum = np.maximum(rowsum, 1e-8)

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)

        norm_adj_mat = norm_adj_mat.tolil()

        self.R = norm_adj_mat[:self.n_users, self.n_users:]

        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def conv_ui(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

        return all_embeddings

    def conv_ii(self, ii_adj, single_modal):
        for i in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    def forward_cl_MM_2(self, adj, image_adj, text_adj, config, train=False):
        def cross_modal_interaction(img_feat, txt_feat):
            attn = torch.matmul(img_feat, txt_feat.T) / np.sqrt(img_feat.size(-1))
            attn = F.softmax(attn, dim=-1)
            cross_feat = torch.matmul(attn, txt_feat)
            return img_feat + 0.5 * cross_feat

        if config['trans'] == 0:
            image_feats = torch.mm(self.v_feat.detach(), self.image_trans)
            text_feats = torch.mm(self.t_feat.detach(), self.text_trans)
        elif config['trans'] == 1 or config['trans'] == 2:
            image_feats = self.image_trans(self.v_feat.detach())
            text_feats = self.text_trans(self.t_feat.detach())
        elif config['trans'] == 3:
            image_feats = torch.mm(self.v_feat.detach(), self.image_trans)
            text_feats = self.text_trans(self.t_feat.detach())
        else:
            text_feats = torch.mm(self.t_feat.detach(), self.text_trans)
            image_feats = self.image_trans(self.v_feat.detach())

        fused_image = cross_modal_interaction(image_feats, text_feats)
        fused_text = cross_modal_interaction(text_feats, image_feats)

        def enhanced_gcn(feats, adj):
            all_embeds = []
            current = torch.cat([self.user_embedding, F.normalize(feats)])
            for gcn in self.gcnLayers:
                current = gcn(adj, current)
                current = F.layer_norm(current, (current.size(-1),))
                all_embeds.append(current)
            return torch.stack(all_embeds).mean(dim=0)

        embeds1 = enhanced_gcn(fused_image, image_adj)
        embeds2 = enhanced_gcn(fused_text, text_adj)

        user1, item1 = embeds1[:self.n_users], embeds1[self.n_users:]
        user2, item2 = embeds2[:self.n_users], embeds2[self.n_users:]
    
        return user1, item1, user2, item2

    def forward_3(self, adj, image_adj, text_adj, config, train=False):
        if config['trans'] == 0:
            image_feats = torch.mm(self.v_feat.detach(), self.image_trans)
            text_feats = torch.mm(self.t_feat.detach(), self.text_trans)
        elif config['trans'] == 1 or config['trans'] == 2:
            image_feats = self.image_trans(self.v_feat.detach())
            text_feats = self.text_trans(self.t_feat.detach())
        elif config['trans'] == 3:
            image_feats = torch.mm(self.v_feat.detach(), self.image_trans)
            text_feats = self.text_trans(self.t_feat.detach())
        else:
            text_feats = torch.mm(self.t_feat.detach(), self.text_trans)
            image_feats = self.image_trans(self.v_feat.detach())
        weight = self.softmax(self.modal_weight)
        embedsImageAdj = torch.concat([self.user_embedding, self.item_id_embedding])
        embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

        embedsImage = torch.concat([self.user_embedding, F.normalize(image_feats)])
        embedsImage = torch.spmm(adj, embedsImage)


        embedsImage_ = torch.concat([embedsImage[:self.n_users], self.item_id_embedding])
        embedsImage_ = torch.spmm(adj, embedsImage_)
        embedsImage += embedsImage_

        embedsTextAdj = torch.concat([self.user_embedding, self.item_id_embedding])
        embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

        embedsText = torch.concat([self.user_embedding, F.normalize(text_feats)])
        embedsText = torch.spmm(adj, embedsText)

        embedsText_ = torch.concat([embedsText[:self.n_users], self.item_id_embedding])
        embedsText_ = torch.spmm(adj, embedsText_)
        embedsText += embedsText_

        embedsImage += config['ris_adj_lambda'] * embedsImageAdj
        embedsText += config['ris_adj_lambda'] * embedsTextAdj

        embedsModal = weight[0] * embedsImage + weight[1] * embedsText

        embeds = embedsModal
        embedsLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)

        embeds = embeds + config['ris_lambda'] * F.normalize(embedsModal)

        user_embeds, item_embeds = embeds[:self.n_users], embeds[self.n_users:]

        #  Encoding Multiple Modalities
        image_item_embeds = torch.multiply(self.item_id_embedding, self.image_space_trans(self.image_embedding.weight))
        text_item_embeds = torch.multiply(self.item_id_embedding, self.text_space_trans(self.text_embedding.weight))


        extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds)


        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)

        extended_image_embeds = self.conv_ui(adj, self.extended_image_user, explicit_image_item)
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)

        extended_text_embeds = self.conv_ui(adj, self.extended_text_user, explicit_text_item)

        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2
        # Attributes Separation for Better Integration
        image_weights, text_weights = torch.split(
            self.softmax(
                torch.cat([
                    self.separate_coarse(explicit_image_embeds),
                    self.separate_coarse(explicit_text_embeds)
                ], dim=-1)
            ),
            1,
            dim=-1
        )
        coarse_grained_embeds = image_weights * explicit_image_embeds + text_weights * explicit_text_embeds

        fine_grained_image = torch.multiply(self.image_behavior(extended_id_embeds),
                                            (explicit_image_embeds - coarse_grained_embeds))
        fine_grained_text = torch.multiply(self.text_behavior(extended_id_embeds),
                                           (explicit_text_embeds - coarse_grained_embeds))
        integration_embeds = (fine_grained_image + fine_grained_text + coarse_grained_embeds) / 3

        all_embeds = extended_id_embeds + integration_embeds

        if train:
            return all_embeds, (integration_embeds, extended_id_embeds, extended_it_embeds), (
                explicit_image_embeds, explicit_text_embeds),extended_image_embeds, extended_text_embeds,embedsImage,embedsText,self.item_id_embedding
        return all_embeds

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        reg_loss = self.reg_weight_1 * regularizer

        return mf_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)

        return torch.mean(cl_loss)

    def cal_noise_loss(self, id, emb, temp):

        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(self.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        emb_view1 = add_perturbation(emb)
        emb_view2 = add_perturbation(emb)
        emb_loss = self.InfoNCE(emb_view1[id], emb_view2[id], temp)

        return emb_loss

    def align_vt(self, embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)

        vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()

        return vt_loss

    def getUserEmbeds(self):
        return self.user_embedding

    def getItemEmbeds(self):
        return self.item_id_embedding

    def getImageFeats(self, config):
        if config['trans'] == 0 or  config['trans'] == 3:
            image_feats = torch.mm(self.v_feat.detach(), self.image_trans)
            return image_feats
        else:
            return self.image_trans(self.v_feat.detach())

    def getTextFeats(self, config):
        if config['trans'] == 0 or  config['trans'] == 4:
            text_feats = torch.mm(self.t_feat.detach(), self.text_trans)
            return text_feats
        else:
            return self.text_trans(self.t_feat.detach())

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        a = sp.csr_matrix((self.n_users, self.n_users))
        b = sp.csr_matrix((self.n_items, self.n_items))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()


class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return item, index

    def __len__(self):
        return len(self.data)


class TrnData(data.Dataset):
    def __init__(self, coomat, item):
        self.item = item
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(self.item)
                if (u, iNeg) not in self.dokmat:
                    break
                self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32) / (
                self.time_emb_dim // 2)).cuda()
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h

class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            if beta_fixed:
                self.betas[0] = 0.0001
            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(
                min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
        return np.array(betas)

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor(
                [steps - 1] * x_start.shape[0]).cuda()
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean
        return x_t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t,
                                         x_start.shape) * x_start + self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.cuda()
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t,
                                                x.shape) * model_output + self._extract_into_tensor(
            self.posterior_mean_coef2, t, x.shape) * x)

        return model_mean, model_log_variance

    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
        batch_size = x_start.size(0)

        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)

        mse = self.mean_flat((x_start - model_output) ** 2)

        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        diff_loss = weight * mse

        usr_model_embeds = torch.mm(model_output, model_feats)
        usr_id_embeds = torch.mm(x_start, itmEmbeds)

        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

        return diff_loss, gc_loss

    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate

    def forward(self, adj):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)




class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)