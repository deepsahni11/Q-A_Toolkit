


from __future__ import absolute_import
from __future__ import division

import torch
import random
import re
import time
import os
import pickle
import tqdm as tqdm

import numpy as np
from six.moves import xrange

from helper_functions import *

class Batch():
    """A class to hold the information needed for a training batch"""
    def __init__(self,names,context_word_index_batch,question_word_index_batch, span_tensor_batch):

        self.names = names
        self.context_word_index_batch = context_word_index_batch

        self.question_word_index_batch = question_word_index_batch
        self.span_tensor_batch = span_tensor_batch
        self.batch_size = len(self.context_word_index_batch)





def refill_batches(batches,batch_size,names, max_context_length, max_question_length,context_word_index,question_word_index,span_tensor):

    """

    Adds more batches into the "batches" list.
    Inputs:
      batches: list to add batches to

      names: list containing strings of file names ["train_context","train_question"] or ["validation_context","validation_question"]
      data_dir : paths to {train/dev}.{context/question/answer} data files
      batch_size: integer ==> how big to make the batches
      max_context_length, max_question_length: max length of context and question respectively

    """
    print ("Refilling batches...")
    tic = time.time()
    examples = []



        # add to examples
    examples.append((context_word_index, question_word_index, span_tensor))




    # Make into batches and append to the list batches
    for batch_start in xrange(0, len(examples[0][0]), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        context_word_index_batch = examples[0][0][batch_start:batch_start+batch_size]
        question_word_index_batch = examples[0][1][batch_start:batch_start+batch_size]
        span_tensor_batch = examples[0][2][batch_start:batch_start+batch_size]


        batches.append((context_word_index_batch, question_word_index_batch,span_tensor_batch))




    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
    print ("Refilling batches took %.2f seconds" % (toc-tic))
    return batches


def get_batch_generator(data_dir, names, batch_size, max_context_length, max_question_length):
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

    context_path_train = os.path.join(data_dir, "train.context")
    question_path_train = os.path.join(data_dir, "train.question")


    context_word_index_old = index_files_using_word_to_index(context_path_train, word_to_index, max_context_length)
    question_word_index_old = index_files_using_word_to_index(question_path_train, word_to_index, max_question_length)


    with open(data_dir + "//" + "answer_end_pkl.pkl", "rb") as input_file:
        answer_end_pkl = pickle.load(input_file)
    with open(data_dir + "//" + "answer_start_pkl.pkl", "rb") as input_file:
        answer_start_pkl = pickle.load(input_file)


    answer_end = torch.from_numpy(np.array([int(i) for i in answer_end_pkl])).long()
    answer_start = torch.from_numpy(np.array([int(i) for i in answer_start_pkl])).long()
    answer_start = torch.unsqueeze(answer_start, 1)
    answer_end = torch.unsqueeze(answer_end, 1)

    span_tensor_old = torch.cat((answer_start, answer_end), 1)
    span_tensor = span_tensor_old[67:83]
    context_word_index = context_word_index_old[67:83]
    question_word_index = question_word_index_old[67:83]



    batches = []
    count = 0

    while (True):
        count = count + 1
        if len(batches) == 0: # add more batches
            if(count > 2):
                break
            batches = refill_batches(batches,batch_size,names, max_context_length, max_question_length,context_word_index,question_word_index,span_tensor)
        if len(batches) == 0:
            break

        # Get next batch. These are all lists length batch_size
        (context_word_index_batch, question_word_index_batch,span_tensor_batch) = batches.pop(0)


        if(len(context_word_index_batch) == 0):
            break



        # Make into a Batch object
        batch = Batch(names,context_word_index_batch, question_word_index_batch, span_tensor_batch)

        yield batch

    return
