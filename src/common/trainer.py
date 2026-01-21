import itertools
import scipy.sparse as sp
import torch
import torch.optim as optim
from scipy.sparse import coo_matrix

import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger
from utils.logger import log

from utils.utils import *
from utils.topk_evaluator import TopKEvaluator
from models.dguma import Denoise, GaussianDiffusion

class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0


        self.diffusion_model = GaussianDiffusion(config['noise_scale'], config['noise_min'], config['noise_max'],
                                                 config['steps']).cuda()
        out_dims = eval(config['dims']) + [model.R.shape[1]]
        in_dims = out_dims[::-1]
        self.denoise_model_image = Denoise(in_dims, out_dims, config['d_emb_size'], norm=False).cuda()
        self.denoise_model_text = Denoise(in_dims, out_dims, config['d_emb_size'], norm=False).cuda()

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer,self.denoise_opt_image, self.denoise_opt_text= self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        self.optimizer_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.image_scheduler = optim.lr_scheduler.LambdaLR(self.denoise_opt_image, lr_lambda=fac)
        self.text_scheduler = optim.lr_scheduler.LambdaLR(self.denoise_opt_text, lr_lambda=fac)

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)
        self.item_tensor = None
        self.tot_item_num = None

        self.epoch_times = []

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            denoise_opt_image = optim.Adam(self.denoise_model_image.parameters(), lr=self.learning_rate,weight_decay=0)
            denoise_opt_text = optim.Adam(self.denoise_model_text.parameters(), lr=self.learning_rate,weight_decay=0)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            denoise_opt_image = optim.SGD(self.denoise_model_image.parameters(), lr=self.learning_rate,weight_decay=0)
            denoise_opt_text = optim.SGD(self.denoise_model_text.parameters(), lr=self.learning_rate,weight_decay=0)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            denoise_opt_image = optim.Adagrad(self.denoise_model_image.parameters(), lr=self.learning_rate,weight_decay=0)
            denoise_opt_text = optim.Adagrad(self.denoise_model_text.parameters(), lr=self.learning_rate,weight_decay=0)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            denoise_opt_image = optim.RMSprop(self.denoise_model_image.parameters(), lr=self.learning_rate,weight_decay=0)
            denoise_opt_text = optim.RMSprop(self.denoise_model_text.parameters(), lr=self.learning_rate,weight_decay=0)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            denoise_opt_image = optim.Adam(self.denoise_model_image.parameters(), lr=self.learning_rate)
            denoise_opt_text = optim.Adam(self.denoise_model_text.parameters(), lr=self.learning_rate)
        return optimizer,denoise_opt_image,denoise_opt_text

    def train(self):
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            train_loss = self._train_one_epoch(epoch)

            val_metrics = self._evaluate(self.val_data)
            val_ndcg = val_metrics['ndcg@10']

            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)

            if val_ndcg > self.best_metric:
                self.best_metric = val_ndcg
                self.best_epoch = epoch
                self._save_checkpoint(epoch)

            print(f'Epoch {epoch}, Loss: {train_loss:.4f}, NDCG@10: {val_ndcg:.4f}, Time: {epoch_time:.2f}s')

        avg_time = np.mean(self.epoch_times[:self.best_epoch])
        print(f'达到最佳性能（Epoch {self.best_epoch}）的平均耗时: {avg_time:.2f}s')

        return self.best_metric, avg_time

    def _train_epoch(self, train_data, epoch_idx):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        config = self.config
        self.model.train()

        self.model.Diff_trnLoader.dataset.negSampling()
        epDiLoss_image, epDiLoss_text = 0, 0
        diffusionLoader = self.model.diffusionLoader
        for i, batch in enumerate(diffusionLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
            iEmbeds = self.model.getItemEmbeds().detach()

            image_feats = self.model.getImageFeats(config).detach()
            text_feats = self.model.getTextFeats(config).detach()
            self.denoise_opt_image.zero_grad()
            self.denoise_opt_text.zero_grad()

            diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item,
                                                                                  iEmbeds, batch_index, image_feats)
            diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item,
                                                                                iEmbeds, batch_index, text_feats)

            loss_image = diff_loss_image.mean() + gc_loss_image.mean() * config['e_loss']
            loss_text = diff_loss_text.mean() + gc_loss_text.mean() * config['e_loss']

            epDiLoss_image += loss_image.item()
            epDiLoss_text += loss_text.item()

            loss = loss_image + loss_text
            loss.backward()

            self.denoise_opt_image.step()
            self.denoise_opt_text.step()

        log('')
        log('Start to re-build UI matrix')

        with torch.no_grad():
            u_list_image = []
            i_list_image = []
            edge_list_image = []

            u_list_text = []
            i_list_text = []
            edge_list_text = []

            for _, batch in enumerate(diffusionLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                # image
                denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item,
                                                               config['sampling_steps'], config['sampling_noise'])
                top_item, indices_ = torch.topk(denoised_batch, k=config['rebuild_k'])

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        u_list_image.append(int(batch_index[i].cpu().numpy()))
                        i_list_image.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_image.append(1.0)

                # text
                denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, config['sampling_steps'],
                                                               config['sampling_noise'])
                top_item, indices_ = torch.topk(denoised_batch, k=config['rebuild_k'])

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        u_list_text.append(int(batch_index[i].cpu().numpy()))
                        i_list_text.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_text.append(1.0)

            # image
            u_list_image = np.array(u_list_image)
            i_list_image = np.array(i_list_image)
            edge_list_image = np.array(edge_list_image)
            self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image,self.model.ii_adj)
            self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)
            self.R_image = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)


            # text
            u_list_text = np.array(u_list_text)
            i_list_text = np.array(i_list_text)
            edge_list_text = np.array(edge_list_text)
            self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text,self.model.ii_adj)
            self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)
            self.R_text = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)

        log('UI matrix built!')
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses = self.calculate_loss(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            loss.backward()

            self.optimizer.step()

            loss_batches.append(loss.detach())
        return total_loss, loss_batches
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            for param_group in self.optimizer.param_groups:
               self.logger.info('======lr: ' + str(param_group['lr']))

            self.optimizer_scheduler.step()
            self.image_scheduler.step()
            self.text_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result
                    

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            scores = self.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        all_embeds = self.model.forward_3(self.model.norm_adj,self.image_UI_matrix,self.text_UI_matrix,self.config)
        restore_user_e, restore_item_e = torch.split(all_embeds, [self.model.n_users, self.model.n_items], dim=0)
        u_embeddings = (restore_user_e[user])
        scores = (torch.matmul(u_embeddings, restore_item_e.transpose(0, 1)))
        return scores
    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
    def buildUIMatrix(self, u_list, i_list, edge_list,item_adj):
        mat = coo_matrix((edge_list, (u_list, i_list)), shape=(self.model.n_users, self.model.n_items), dtype=np.float32)
        a = sp.csr_matrix((self.model.n_users, self.model.n_users))
        b = sp.csr_matrix((self.model.n_items,self.model.n_items))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        norm_adj_mat = mat.tolil()

        self.R = norm_adj_mat[:self.model.n_users, self.model.n_users:]

        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        embeds_1, embeds_2, embeds_3,extended_image_embeds, extended_text_embeds,image_embeds,text_embeds,item_id_embedding = self.model.forward_3(self.model.norm_adj,self.image_UI_matrix,self.text_UI_matrix,self.config, train=True)
        users_embeddings, items_embeddings = torch.split(embeds_1, [self.model.n_users, self.model.n_items], dim=0)

        integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
        extended_it_embeds_1 = (image_embeds +  text_embeds)/2

        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        integration_users, integration_items = torch.split(integration_embeds, [self.model.n_users, self.model.n_items], dim=0)
        extended_id_user, extended_id_items = torch.split(extended_id_embeds, [self.model.n_users, self.model.n_items], dim=0)
        bpr_loss, reg_loss_1 = self.model.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        bm_loss = self.model.bm_loss * (
                    self.model.InfoNCE(integration_users[users], extended_id_user[users], self.model.bm_temp) + self.model.InfoNCE(
                integration_items[pos_items], extended_id_items[pos_items], self.model.bm_temp))


        usr1, itm1, usr2, itm2 = self.model.forward_cl_MM_2(self.model.norm_adj, self.image_UI_matrix,self.text_UI_matrix, self.config, train=True)
        bm_loss_3  = self.model.bm_loss * (
                self.model.InfoNCE(usr1[users], usr2[users],
                                   self.model.bm_temp) + self.model.InfoNCE(
            itm1[pos_items], itm2[pos_items], self.model.bm_temp))
       

        al_loss = bm_loss+bm_loss_3
        extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.model.n_users, self.model.n_items], dim=0)
        extended_it_user_1, extended_it_items_2 = torch.split(extended_it_embeds_1, [self.model.n_users, self.model.n_items], dim=0)

        c_loss = self.model.InfoNCE(extended_it_user[users], integration_users[users], self.model.um_temp)
        c_loss_1 = self.model.InfoNCE(extended_it_user_1[users], integration_users[users], self.model.um_temp)
        c_loss_2 = self.model.InfoNCE(extended_it_user_1[users], extended_it_user[users], self.model.um_temp)
        um_loss = self.model.um_loss * (c_loss_2+c_loss_1+c_loss)

        reg_loss_2 = self.model.reg_weight_2 * self.model.sq_sum(extended_it_items[pos_items]) / self.model.batch_size
        reg_loss_3 = self.model.reg_weight_2 * self.model.sq_sum(extended_it_items_2[pos_items]) / self.model.batch_size
        reg_loss = reg_loss_1+reg_loss_2+reg_loss_3

        return bpr_loss + al_loss + um_loss + reg_loss




