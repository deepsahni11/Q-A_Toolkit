


from __future__ import absolute_import
from __future__ import division
from torch import nn
import torch
import random
import re
import time
import os
import pickle
import tqdm as tqdm
import numpy as np
torch.manual_seed(4)
np.random.seed(4)

import numpy as np
from six.moves import xrange

class Batch(nn.Module):
    """A class to hold the information needed for a training batch"""
    def __init__(self,names,context_word_index_batch,context_char_index_batch,question_word_index_batch,question_char_index_batch, span_tensor_batch,context_tokens_batch,questions_tokens_batch,answer_tokens_batch):

        super(Batch, self).__init__()
        self.names = names
        self.context_word_index_batch = context_word_index_batch
        self.question_word_index_batch = question_word_index_batch
        self.context_char_index_batch = context_char_index_batch
        self.question_char_index_batch = question_char_index_batch
        self.span_tensor_batch = span_tensor_batch
        self.context_tokens_batch = context_tokens_batch
        self.questions_tokens_batch = questions_tokens_batch
        self.answer_tokens_batch = answer_tokens_batch
        self.batch_size = len(self.context_word_index_batch)

def index_files_using_char_to_index(filename, _dict, max_words, max_chars):

    f = open(filename, "r", encoding="utf-8")
    lines = f.readlines()
    lines = [l.lower() for l in lines]
    encoded_lines = []
    for l in lines:
        tokens = l.split()
        tokens = tokens[:max_words]
        encoded_tokens = []
        for t in tokens:
            l = list(t)
            l = l[:max_chars] ## there is a max limit for the length of characters = max_chars
            encoded_chars = []
            for j in l:
                if j in _dict:
                    encoded_chars.append(_dict[j])
                else:
                    encoded_chars.append(0)  ## if the character id not in dictionary put '0' in its place
            encoded_tokens.append(encoded_chars)
        encoded_lines.append(encoded_tokens)

    return encoded_lines

def index_files_using_word_to_index(filename, _dict, max_words):

    f = open(filename, "r", encoding="utf-8")

    lines = f.readlines()
    lines  = [l.lower() for l in lines]
    encoded_lines = []
    for l in lines:
        tokens = l.split()
        tokens = tokens[:max_words]
        temp = []
        for t in tokens:
            if t in _dict:
                temp.append(_dict[t])
            else:
                temp.append(1)

        encoded_lines.append(temp[:])

    return encoded_lines



def refill_batches(batches,batch_size,names, max_context_length, max_question_length,context_word_index,context_char_index,question_word_index,question_char_index,span_tensor,context_tokens,question_tokens,answer_tokens):

    """

    Adds more batches into the "batches" list.
    Inputs:
      batches: list to add batches to

      names: list containing strings of file names ["train_context","train_question"] or ["validation_context","validation_question"]
      data_dir : paths to {train/dev}.{context/question/answer} data files
      batch_size: integer ==> how big to make the batches
      max_context_length, max_question_length: max length of context and question respectively

    """
#     print ("Refilling batches...")
    tic = time.time()
    examples = []



        # add to examples
    examples.append((context_word_index,context_char_index, question_word_index,question_char_index, span_tensor,context_tokens,question_tokens,answer_tokens))




    # Make into batches and append to the list batches
    for batch_start in xrange(0, len(examples[0][0]), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        context_word_index_batch = examples[0][0][batch_start:batch_start+batch_size]
        context_char_index_batch = examples[0][1][batch_start:batch_start+batch_size]
        question_word_index_batch = examples[0][2][batch_start:batch_start+batch_size]
        question_char_index_batch = examples[0][3][batch_start:batch_start+batch_size]
        span_tensor_batch = examples[0][4][batch_start:batch_start+batch_size]
        context_tokens_batch = examples[0][5][batch_start:batch_start+batch_size]
        questions_tokens_batch = examples[0][6][batch_start:batch_start+batch_size]
        answer_tokens_batch = examples[0][7][batch_start:batch_start+batch_size]

#         print("Batch " + str(batch_start + 1) + " loaded")

        batches.append((context_word_index_batch,context_char_index_batch, question_word_index_batch,question_char_index_batch,span_tensor_batch,context_tokens_batch,questions_tokens_batch,answer_tokens_batch))




    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
#     print ("Refilling batches took %.2f seconds" % (toc-tic))
    return batches


def get_batch_generator(data_dir,names, batch_size, max_context_length, max_question_length,max_char_length,prefix):
    """
    This function returns a generator object that yields batches.
    The last batch in the dataset will be a partial batch.
    Read this to understand generators and the yield keyword in Python: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    Inputs:
      names: list containing strings of file names = ["train_context","train_question"] or ["validation_context","validation_question"]
      data_dir : paths to {train/dev}.{context/question/answer} data files
      batch_size: integer ==> how big to make the batches
      max_context_length, max_question_length: max length of context and question respectively

    """

    with open(os.path.join(data_dir , "dictionaries.pkl"), "rb") as input_file:
        dictionaries = pickle.load(input_file)
    word_to_index = dictionaries["word_to_index"]
    char_to_index = dictionaries["char_to_index"]

    context_path_train = os.path.join(data_dir, prefix + ".context")
    question_path_train = os.path.join(data_dir, prefix +  ".question")
    answer_path_train = os.path.join(data_dir, prefix +  ".answer_text")

    context_tokens = open(context_path_train, "r", encoding="utf-8").readlines()
    question_tokens =  open(question_path_train, "r", encoding="utf-8").readlines()
    answer_tokens = open(answer_path_train, "r", encoding="utf-8").readlines()

    context_word_index_old = index_files_using_word_to_index(context_path_train, word_to_index, max_context_length)
    question_word_index_old = index_files_using_word_to_index(question_path_train, word_to_index, max_question_length)

    context_char_index_old = index_files_using_char_to_index(context_path_train, char_to_index, max_context_length, max_char_length)
    question_char_index_old = index_files_using_char_to_index(question_path_train, char_to_index, max_question_length, max_char_length)

    answer_start_path = os.path.join(data_dir  + prefix +  ".answer_start")
    answer_start_list = open(answer_start_path, "r", encoding="utf-8").readlines()

    answer_end_path = os.path.join(data_dir + prefix +  ".answer_end")
    answer_end_list = open(answer_end_path, "r", encoding="utf-8").readlines()


    context_tokens = context_tokens[0:100]
    question_tokens = question_tokens[0:100]
    answer_tokens = answer_tokens[0:100]

    answer_end = torch.from_numpy(np.array([int(i) for i in answer_end_list])).long()
    answer_start = torch.from_numpy(np.array([int(i) for i in answer_start_list])).long()
    answer_start = torch.unsqueeze(answer_start, 1)
    answer_end = torch.unsqueeze(answer_end, 1)

    span_tensor_old = torch.cat((answer_start, answer_end), 1)
    span_tensor = span_tensor_old[0:100]
    context_word_index = context_word_index_old[0:100]
    question_word_index = question_word_index_old[0:100]
    context_char_index = context_char_index_old[0:100]
    question_char_index = question_char_index_old[0:100]



    batches = []
    count = 0

    while (True):
        count = count + 1
        if len(batches) == 0: # add more batches
            if(count > 2):
                break
            batches = refill_batches(batches,batch_size,names, max_context_length, max_question_length,context_word_index,context_char_index,question_word_index,question_char_index,span_tensor,context_tokens,question_tokens,answer_tokens)
        if len(batches) == 0:
            break

        # Get next batch. These are all lists length batch_size
        (context_word_index_batch,context_char_index_batch, question_word_index_batch,question_char_index_batch,span_tensor_batch,context_tokens,question_tokens,answer_tokens) = batches.pop(0)


        if(len(context_word_index_batch) == 0):
            break



        # Make into a Batch object
        batch = Batch(names,context_word_index_batch, context_char_index_batch,question_word_index_batch,question_char_index_batch, span_tensor_batch,context_tokens,question_tokens,answer_tokens)

        yield batch

    return
