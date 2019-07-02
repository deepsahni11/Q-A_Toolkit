from dataset_iterator import *
from helper_functions import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
import code
import os

class Train_Model:

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.parameters_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = optim.Adam(self.parameters_trainable, lr=self.config.lr)

        self.glove_path = os.path.join(config.data_dir, "glove_word_embeddings.pkl")
        self.num_epochs = config.num_epochs
        self.data_dir = config.data_dir
        self.names = config.names
        self.batch_size = config.batch_size
        self.print_every = config.print_every
        self.max_context_length = config.max_context_length
        self.max_question_length = config.max_question_length
        self.model_dir = config.model_dir
        self.early_stop = config.early_stop
        self.print_and_validate_every = config.print_and_validate_every
        
    def save_model(self, model, optimizer, loss, global_step, epoch ,prefix):
        # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor
        
        if(prefix == "best_model"):
            model_state = model.state_dict()
            model_state = {k: v for k, v in model_state.items() if 'embedding' not in k}

            state = {
                'global_step': global_step,
                'epoch': epoch,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'current_loss': loss
            }
            model_save_path = os.path.join(self.model_dir, 'best_model' )
            torch.save(state, model_save_path)
            
        elif(prefix == "last_model"):
            model_state = model.state_dict()
            model_state = {k: v for k, v in model_state.items() if 'embedding' not in k}

            state = {
                'global_step': global_step,
                'epoch': epoch,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'current_loss': loss
            }
            model_save_path = os.path.join(self.model_dir, 'last_model' )
            torch.save(state, model_save_path)
            
            
    def get_f1_em_score(self, prefix, num_samples=100):


        f1_total = 0.
        em_total = 0.
        example_num = 0

#         tic = time.time()

        for batch in get_batch_generator(self.data_dir, self.names, self.batch_size, self.max_context_length, self.max_question_length,prefix):

            _,start_pos_prediction, end_pos_prediction = self.test_one_batch(batch)

            start_pos_prediction = start_pos_prediction.tolist()
            end_pos_prediction = end_pos_prediction.tolist()

            for index, (pred_answer_start, pred_answer_end, true_answer_tokens) in enumerate(zip(start_pos_prediction, end_pos_prediction, batch.answer_tokens_batch)):
                    
                example_num += 1
                pred_answer_tokens = batch.context_tokens_batch[index][pred_answer_start : pred_answer_end + 1]
                pred_answer = " ".join(pred_answer_tokens)

                true_answer = " ".join(true_answer_tokens)

                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

#         toc = time.time()
#         logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total
    def get_validation_loss(self,prefix):
#         logging.info("Calculating dev loss...")
#         tic = time.time()
#         loss_per_batch, batch_lengths = [], []
        total_validation_loss = 0.0
        validation_set_size = 0
        for batch in get_batch_generator(self.data_dir, self.names, self.batch_size, self.max_context_length, self.max_question_length,prefix):

            validation_batch_loss, _, _ = self.test_one_batch(batch)
            validation_set_size += batch.batch_size
            total_validation_loss += validation_batch_loss
#             batch_lengths.append(curr_batch_size)
#             i += 1
#             if i == 10:
#                 break
#         total_num_examples = sum(batch_lengths)
#         toc = time.time()
#         print(validation_set_size)

        validation_loss = total_validation_loss / validation_set_size
#         print "Computed validation loss = %f " % (validation_loss)

        return validation_loss

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
    
    def test_one_batch(self, batch):
        
        self.model.eval()

        context_word_index_batch, question_word_index_batch,  span_tensor_batch = self.get_data(batch)

#         print(context_word_index_batch)


        context_word_index_padded_per_batch = Variable(pad_data(context_word_index_batch))
#         print(context_word_index_padded_per_batch)
        context_word_index_padded_per_batch.requires_grad = False
        question_word_index_padded_per_batch = Variable(pad_data(question_word_index_batch))
        question_word_index_padded_per_batch.requires_grad = False


        context_ids = np.array(context_word_index_padded_per_batch) # shape (batch_size, context_len)
        context_mask_per_batch = (context_ids != 0).astype(np.int32) # shape (batch_size, context_len)
        context_word_mask_per_batch_new = Variable(torch.from_numpy(context_mask_per_batch))
        context_word_mask_per_batch_new.requires_grad = False

        question_ids = np.array(question_word_index_padded_per_batch) # shape (batch_size, context_len)
        question_mask_per_batch = (question_ids != 0).astype(np.int32) # shape (batch_size, context_len)
        question_word_mask_per_batch_new = Variable(torch.from_numpy(question_mask_per_batch))
        question_word_mask_per_batch_new.requires_grad = False

        span_tensor_batch = Variable(span_tensor_batch)
        
        span_tensor_batch.requires_grad = False

        loss,start_index_prediction, end_index_prediction = self.model(context_word_index_padded_per_batch,context_word_mask_per_batch_new, question_word_index_padded_per_batch, question_word_mask_per_batch_new, span_tensor_batch)
    
        self.model.train()

        return loss.item(),start_index_prediction, end_index_prediction
        
    def train_one_batch(self, batch):



        self.optimizer.zero_grad()
        context_word_index_batch, question_word_index_batch,  span_tensor_batch = self.get_data(batch)



        context_word_index_padded_per_batch = Variable(pad_data(context_word_index_batch))
        context_word_index_padded_per_batch.requires_grad = False
        question_word_index_padded_per_batch = Variable(pad_data(question_word_index_batch))
        question_word_index_padded_per_batch.requires_grad = False


        context_ids = np.array(context_word_index_padded_per_batch) # shape (batch_size, context_len)
        context_mask_per_batch = (context_ids != 0).astype(np.int32) # shape (batch_size, context_len)
        context_word_mask_per_batch_new = Variable(torch.from_numpy(context_mask_per_batch))
        context_word_mask_per_batch_new.requires_grad = False

        question_ids = np.array(question_word_index_padded_per_batch) # shape (batch_size, context_len)
        question_mask_per_batch = (question_ids != 0).astype(np.int32) # shape (batch_size, context_len)
        question_word_mask_per_batch_new = Variable(torch.from_numpy(question_mask_per_batch))
        question_word_mask_per_batch_new.requires_grad = False

        span_tensor_batch = Variable(span_tensor_batch)
        
        span_tensor_batch.requires_grad = False

        loss, _, _ = self.model(context_word_index_padded_per_batch,context_word_mask_per_batch_new, question_word_index_padded_per_batch, question_word_mask_per_batch_new, span_tensor_batch)


        
#         print(loss)
        
#         l2_reg = None
#         for W in self.parameters:
            
#             if l2_reg is None:
#                 l2_reg = W.norm(2)
#             else:
#                 l2_reg = l2_reg + W.norm(2)
#         loss = loss + config.reg_lambda * l2_reg
        

        
        loss.backward()
        

        
        param_norm = self.get_param_norm(self.parameters_trainable)
        grad_norm = self.get_grad_norm(self.parameters_trainable)
        
        
#         clip_grad_norm_(parameters, config.max_grad_norm)
        self.optimizer.step()
    

        return loss.item(), param_norm, grad_norm
    
    
    def train(self):


        num_parameters = sum(p.numel() for p in self.parameters_trainable)

        best_validation_f1, best_validation_em = None, None
        best_validation_epoch = 0
        epoch = 0
        global_step = 0

        loss_array = []
        logging.info("Beginning training loop...")
        for epoch in range(200):
            total_loss = 0.0
            epoch_tic = time.time()
            for batch in get_batch_generator(self.data_dir, self.names, self.batch_size, self.max_context_length, self.max_question_length,"train"):
                
                global_step += 1
                iter_tic = time.time()
                
                train_batch_loss, param_norm, grad_norm = self.train_one_batch(batch)

#                 total_loss = total_loss + loss
#                 loss_array.append(total_loss)
    
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic
            
                if global_step % self.print_and_validate_every == 0:
                    
#                     print(self.get_validation_loss("validation"))
                    validation_batch_loss = self.get_validation_loss("validation")
                    


                    train_batch_f1, train_batch_em = self.get_f1_em_score("train", num_samples=100)


                    validation_batch_f1, validation_batch_em = self.get_f1_em_score("validation", num_samples=100)




                    if best_validation_f1 is None or validation_batch_f1 > best_validation_f1:
                        best_validation_f1 = validation_batch_f1

                    if best_validation_em is None or validation_batch_em > best_validation_em:
                        best_validation_em = validation_batch_em
                        best_validation_epoch = epoch+1
                        self.save_model(self.model, self.optimizer, validation_batch_loss, global_step, epoch, "best_model")
                        
                    print ("Epoch : {} Step : {} Train_batch Loss : {} Validation_batch Loss :{} " .format(epoch+1, global_step, train_batch_loss, validation_batch_loss))
            
                    print("Train_batch F1:{} Train_batch EM:{} Validation_batch_F1: {} Best_validation_batch F1:{} Best_validation_batch EM :{} ".format(train_batch_f1,train_batch_em,validation_batch_f1,best_validation_f1,best_validation_em))

                
            if (epoch - best_validation_epoch > self.early_stop):
                break
            self.save_model(self.model, self.optimizer, train_batch_loss, global_step, epoch+1 ,"last_model")
#             self.save_model(model, optimizer, loss, global_step, epoch, bestmodel_dir)
#             torch.save(self.model.state_dict(), open(os.path.join(outdir, "last_model"), "wb"))
#             print("total loss for epoch number = " + str(epoch+1) + " = " + str(total_loss))

#             epoch_toc = time.time()
            print("End of epoch %i." % (epoch+1))

        sys.stdout.flush()