import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import code
import gc
import dataset_iterator_drop

import Embedding_Layer(1).embedding as embedding
import Encoding_Layer_3.encoding_layer as encoding_layer
import Cross_Interaction_Layer_4.bilinear_compute as bilinear_compute
import Embedding_Combination_Layer(2).encoding_combination_layer as encoding_combination_layer
import Cross_Interaction_Layer_4.query_aware_document_representation as query_aware_document_representation
import Cross_Interaction_Layer_4.document_aware_query_representation as document_aware_query_representation
import Self-Interaction_Layer(5).self_match as self_match
import dataset_iterator_squad
import Output_Layer_10.predict_start as predict_start
import Output_Layer_10.predict_end as predict_end
import Output_Layer_10.mid_processingoutputlayer as mid_processingoutputlayer

import os 
import squad_eval
from numpy import genfromtxt
from torch.autograd import Variable
from torch.nn import Embedding
from torch import zeros, from_numpy, Tensor, LongTensor, FloatTensor
#from argparse import ArgumentParser
import pickle

import torch.backends.cudnn as cudnn
cudnn.benchmark = False 

class TrainModel:

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.parameters_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
		self.outdir="D:/Downloads/SQuAD/"
		
        if (self.config.optim == "Adadelta"):
            self.optimizer = optim.Adadelta(self.parameters_trainable, lr = self.config.lr, rho=0.95)
        else:
            self.optimizer = optim.Adam(self.parameters_trainable, lr=self.config.lr, rho = 0.95)
        self.word_to_index = pickle.load(open(os.path.join(config.emb_dir, "dictionaries.pkl")))["word"]
        self.index_to_word = {v:k for (k,v) in self.word_to_index.iteritems()}

    def update_param(self, loss):
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()



    def load_data(self, data_dir="D:/Downloads/SQuAD/"):

        self.dataset_iterator = dataset_iterator_squad.DataBatch(data_dir)


    def eval(self, dataset, whole_dataset=True, name="valid", epoch="0"):

        self.model.eval()
        num_steps = int(dataset.number_of_examples/float(self.config.batch_size)) + 1


        if not whole_dataset:
            num_steps = 2


        predictions = []
        ground_truths = []
        total_loss = 0
        for i in range(num_steps):

            if name is "train" or name is "validation":
                temp_batch = self.dataset_iterator.next_batch(dataset, 5, is_train=True)
            else:    
                temp_batch = self.dataset_iterator.next_batch(dataset, 5, is_train=False)

            answer_start_batch = Variable(torch.LongTensor(temp_batch["answer_start"]))
            answer_start_batch = torch.squeeze(answer_start_batch)
            
            answer_end_batch = Variable(torch.LongTensor(temp_batch["answer_end"]))
            answer_end_batch = torch.squeeze(answer_end_batch)
            

            content_batches = temp_batch["content"]
            for l,v in content_batches.items():
                content_batches[l] = Variable(torch.LongTensor(v))
                content_batches[l].requires_grad = False

            query_batches = temp_batch["query"]

            for l,v in query_batches.items():
                query_batches[l] = Variable(torch.LongTensor(v)) 
                query_batches[l].requires_grad = False


            pred_start, pred_end = self.model(content_batches, query_batches)
            step_loss = self.loss_fn(pred_start, answer_start_batch)
            step_loss += self.loss_fn(pred_end, answer_end_batch)
            total_loss += self_loss.data

            _, pred_begin = torch.max(pred_start, 1)
            _, pred_end   = torch.max(pred_end,   1)

            pred = torch.stack([pred_begin, pred_end], dim=1)

            temp_batch_data = temp_batch["context_words"]
            
            temp_batch_gt   = temp_batch["ground_truths"]
            for i, (begin, end) in enumerate(pred.cpu().data.numpy()):
                
                temp = temp_batch_data[i][0]
                
                ans = temp
                ans = "".join(str(ans))
                
                if whole_dataset == False:
                    print ("Pred: " , ans, " GT : " , temp_batch_gt[i])
                predictions.append(ans)
                ground_truths.append(temp_batch_gt[i])
        
        exact_match, f1 = evaluate(ground_truths, predictions)
        
        self.model.train()
        print ("Scores of {} are : {} {} ".format(name,f1,exact_match))
        #gc.collect()
        return total_loss/num_steps, f1,exact_match 



    def run_epoch(self, epoch_num):

        train_dataset = self.dataset_iterator.dataset["train"]
        valid_dataset = self.dataset_iterator.dataset["validation"]
        test_dataset  = self.dataset_iterator.dataset["test"]

        num_steps = int(train_dataset.number_of_examples/float(self.config.batch_size)) + 1

        total_loss = 0
        for i in range(num_steps):
            self.model.zero_grad()
            temp_batch = self.dataset_iterator.next_batch(train_dataset, self.config.batch_size, is_train=True)

            

            answer_start_batch = Variable(torch.LongTensor(temp_batch["answer_start"]))
            answer_start_batch = torch.squeeze(answer_start_batch)
            answer_end_batch = Variable(torch.LongTensor(temp_batch["answer_end"]))
            answer_end_batch = torch.squeeze(answer_end_batch)


            content_batches = temp_batch["content"]
            for l,v in content_batches.items():
                content_batches[l] = Variable(torch.LongTensor(v))
                content_batches[l].requires_grad = False

            query_batches = temp_batch["query"]

            for l,v in query_batches.items():
                query_batches[l] = Variable(torch.LongTensor(v)) 
                query_batches[l].requires_grad = False


            begin_logits, end_logits = self.model(content_batches, query_batches)

            step_loss = self.loss_fn(begin_logits, answer_start_batch)
            step_loss += self.loss_fn(end_logits, answer_end_batch)
            self.update_param(step_loss)
            print ("Epoch : {} Step : {} Loss :{}".format(epoch_num, i, step_loss.data))
            total_loss += step_loss.data
            
            del temp_batch, step_loss, answer_start_batch, answer_end_batch, begin_logits, end_logits,  content_batches, query_batches
            #gc.collect()  
        return total_loss/(num_steps)


    def run_training(self):

        outdir = self.outdir

        self.load_data(outdir)

	train_dataset = self.dataset_iterator.dataset["train"]
        valid_dataset = self.dataset_iterator.dataset["validation"]
        test_dataset  = self.dataset_iterator.dataset["test"]

        torch.save(self.model.state_dict(), open(os.path.join(outdir, "last_model"), "wb"))
        best_val_epoch = 0
        if self.resume == 'T':
            self.model.load_state_dict(torch.load(open(os.path.join(outdir, "last_model"), "rb")))
            best_valid_loss, best_valid_exact_match, best_valid_f1 = self.eval(valid_dataset)


        else:
            best_valid_loss = 1e+10
            best_valid_f1  = 0
            best_valid_exact_match = 0

        for epoch in range(self.config.epochs):

            train_loss = self.run_epoch(epoch)

            valid_loss, valid_exact_match, valid_f1 = self.eval(valid_dataset, True, "valid", str(epoch))

            torch.save(self.model.state_dict(), open(os.path.join(outdir, "last_model"), "wb"))

            if (valid_f1 > best_valid_f1):
                torch.save(self.model.state_dict(), open(os.path.join(outdir, "best_model"), "wb"))
                best_valid_f1 = valid_f1
                best_val_epoch = epoch


            train_loss, train_f1, train_exact_match = self.eval(valid_dataset, False, "train", str(epoch))
            print ("Epoch: {} Training Loss:  {} Training F1: {} Training Exact Match: {} ".format(epoch, train_loss, train_f1, train_exact_match))
            print ("Epoch: {} Validation Loss:  {} Validation F1: {} Validation Exact Match: {}".format(epoch, valid_loss, valid_f1, valid_exact_match))

            if (epoch - best_val_epoch > self.config.early_stop):
                break

        self.model.load_state_dict(torch.load(open(os.path.join(outdir, "best_model"), "rb")))
        self.model.eval()
        test_loss, test_f1, test_em = self.eval(test_dataset, True, "test", str(best_val_epoch))
        print ("Epoch: {} Test Loss:  {} Training F1: {} Training Exact Match: {} ".format(best_val_epoch, test_loss, test_f1, test_exact_match))
