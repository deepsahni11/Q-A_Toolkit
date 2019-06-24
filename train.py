import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import code
import gc
from torch.optim import Adam
from helper_functions import *
from dataset_iterator import *
# from Models.config import *


class Train_Model:

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.parameters_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = optim.Adam(self.parameters_trainable, lr = self.config.lr)

        self.glove_path = os.path.join(config.data_dir, "glove_word_embeddings.pkl")
        self.num_epochs = config.num_epochs
        self.data_dir = config.data_dir
        self.names = config.names
        self.batch_size = config.batch_size
        self.print_every = config.print_every
        self.max_context_length = config.max_context_length
        self.max_question_length = config.max_question_length

    def get_data(self, batch, is_train=True):

        question_word_index_batch = batch.question_word_index_batch

        context_word_index_batch = batch.context_word_index_batch

        span_tensor_batch = batch.span_tensor_batch

        if is_train:
            return context_word_index_batch, question_word_index_batch,span_tensor_batch
        else:
            return context_word_index_batch, question_word_index_batch

    def get_grad_norm(self, parameters, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def get_param_norm(self, parameters, norm_type=2):
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def train_one_batch(self, batch, model, optimizer, parameters):
        model.train()
        optimizer.zero_grad()
        context_word_index_batch, question_word_index_batch,  span_tensor_batch = self.get_data(batch)



        context_word_index_padded_per_batch = pad_data(context_word_index_batch)
        question_word_index_padded_per_batch = pad_data(question_word_index_batch)


        context_ids = np.array(context_word_index_padded_per_batch) # shape (batch_size, context_len)
        context_mask_per_batch = (context_ids != 0).astype(np.int32) # shape (batch_size, context_len)
        context_word_mask_per_batch_new = torch.from_numpy(context_mask_per_batch)

        question_ids = np.array(question_word_index_padded_per_batch) # shape (batch_size, context_len)
        question_mask_per_batch = (question_ids != 0).astype(np.int32) # shape (batch_size, context_len)
        question_word_mask_per_batch_new = torch.from_numpy(question_mask_per_batch)


        loss, _, _ = model(context_word_index_padded_per_batch,context_word_mask_per_batch_new, question_word_index_padded_per_batch, question_word_mask_per_batch_new, span_tensor_batch)

        l2_reg = None
        for W in parameters:
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        loss = loss + self.config.reg_lambda * l2_reg

        print(loss.grad_fn)
        print(loss)

#         loss.backward()

        param_norm = self.get_param_norm(parameters)
        grad_norm = self.get_grad_norm(parameters)

        print("param_norm   " ,end = "")
        print(param_norm)

#         clip_grad_norm_(parameters, config.max_grad_norm)
        optimizer.step()

        return loss.item(), param_norm, grad_norm


    def train(self):


        model = self.model
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

        optimizer = Adam(parameters, lr= self.config.lr, amsgrad=True)

        num_parameters = sum(p.numel() for p in parameters)
        logging.info("Number of params: %d" % num_parameters)

        exp_loss, best_dev_f1, best_dev_em = None, None, None

        epoch = 0
        global_step = 0

        logging.info("Beginning training loop...")
        for epoch in range(1):
            epoch_tic = time.time()
            for batch in get_batch_generator(self.data_dir, self.names, self.batch_size, self.max_context_length, self.max_question_length):

                global_step += 1
                iter_tic = time.time()


                loss, param_norm, grad_norm = self.train_one_batch(batch, model, optimizer, parameters)

                print("loss for batch" + str(global_step) + " = " + str(loss))

                iter_toc = time.time()
                iter_time = iter_toc - iter_tic




            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc - epoch_tic))

        sys.stdout.flush()
