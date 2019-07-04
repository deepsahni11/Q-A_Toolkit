import random
import nltk
import numpy as np
import _pickle as cPickle
import sys
import copy
import os.path
#import tensorflow as tf
from vocab import *
#import Drop_padding 

def generate_answer(content_word_list, answer_start, answer_end):

    if (answer_end < answer_start):
        return content_word_list[answer_start]

    else:
        temp = content_word_list[answer_start:answer_end]
        return " ".join(temp)

class DataSplit:

    def __init__(self, name, content_features, query_features, answer_start, answer_end, ground_truths,no_of_sample):

        self.name   = name
        self.content_features = content_features
        self.query_features = query_features
        self.answer_start = answer_start
        self.answer_end = answer_end
        self.number_of_examples = no_of_sample 
        self.ground_truths = ground_truths

        self.global_count_train = 0
        self.global_count_test  = 0



class DataBatch:

    def __init__(self, data_dir="D:/Downloads/SQuAD/"):

        self.dataset = {}
        for i in ["train", "validation","test"]:
            content_features = {}
            query_features = {}
            
            for j in ["", "char_"]:
                content_features["token"+j] = cPickle.load(open(os.path.join(data_dir, i + ".context_" + j + "pkl.pkl"),"rb"))
                query_features["token"+j]   = cPickle.load(open(os.path.join(data_dir, i + ".question_" + j + "pkl.pkl"),"rb"))
                answer_start        = cPickle.load(open(os.path.join(data_dir, i + ".answer_start.txt_pkl.pkl"),"rb"))
                answer_end          = cPickle.load(open(os.path.join(data_dir, i + ".answer_end.txt_pkl.pkl"),"rb"))
                ground_truths       = cPickle.load(open(os.path.join(data_dir, i + ".answer_text.txt_pkl.pkl"),"rb"))
                no_of_samples       = len(answer_end)
            content_features["context_words"]       = cPickle.load(open(os.path.join(data_dir, i + ".context_context_words_pkl.pkl"),"rb"))
            self.dataset[i]         = DataSplit(name = i,content_features=content_features, query_features=query_features, 
                                        answer_start = answer_start, answer_end = answer_end, ground_truths = ground_truths,
                                        no_of_sample=no_of_samples)
            #print(ground_truths[0])
    
    def find_max_length(self, data, idx, batch_size):

        """ Finds the maximum sequence length for data of 
            size batch_size
            Args:
                data: The data from which sequences will be chosen
                idx: The pointer from which retrieval should be done.
                batch_size: Number of examples to be taken to find max.
        """
        #data = data[idx:idx + batch_size]
        temp = 0
        for i, _ in enumerate(data):
            #print data[i]
            if len(data[i]) > temp:
                temp = len(data[i])
        #print ("the max is ", temp)
        return temp

    def pad_data(self,data, idx, batch_size):

        """ Pad the data to max_length given
            Args: 
                data: Data that needs to be padded
                max_length : The length to be achieved with padding
            Returns:
                padded_data : Each sequence is padded to make it of length
                              max_length.
        """

        padded_data = []
        #nprint ("data is" , len(data))
        max_length =  self.find_max_length(data, idx, batch_size)

        for lines in data:
            if (len(lines) < max_length):
                temp = np.lib.pad(lines, (0,max_length - len(lines)),
                    'constant', constant_values=0)
            else:
                temp = lines[:max_length]
            padded_data.append(temp)
        #print(padded_data)
        return padded_data

            
    def make_batch(self, data, batch_size, idx, max_length):


        batch = []
        batch = data[idx:idx+batch_size]
        idx = idx + batch_size
        max_length = 0 
        for b in batch:
            temp = len(b)
            if (temp > max_length):
                max_length = temp

        #index = 0
        #temp = count + batch_size
        while (len(batch) < batch_size):
            #print (max_length)
            batch.append(np.zeros(max_length, dtype = int))
            idx = 0
        #print ("batch is", len(batch))    
        batch = self.pad_data(batch,idx, batch_size)

        #batch = np.transpose(batch)
        batch = np.asarray(batch)
        return batch, idx
    
    def make_batch_gt(self, data, batch_size, idx):
        batch = []
        batch = data[idx:idx + batch_size]
        idx = idx 
        while(len(batch) < batch_size):
            batch.append([""])
            idx = 0

        #print(batch)
        return batch, idx
    
    def pad_data_char(self, data, num_of_words, max_chars):

        padded_data = []

        for lines in data:
            if (len(lines)< max_chars):
                temp = np.lib.pad(lines, (0,max_chars - len(lines)), 
                                  'constant', constant_values=0)
            else:
                temp = lines[:max_chars]
            padded_data.append(temp)


        if (len(padded_data) < num_of_words):
            padded_data.append(np.zeros(max_chars, dtype = int))

        return padded_data

    def make_batch_char(self, data, batch_size, idx, max_length, num_chars):

        """ Make a matrix of size [batch_size * max_length]
            for given dataset
            Args:
                data: Make batch from this dataset
                batch_size : batch size
                idx : pointer from where retrieval will be done
                max_length : maximum length to be padded into
            Returns
                batch: A matrix of size [batch_size * max_length]
                count: The point from where the next retrieval is done.
        """
        batch=[]
        batch = data[idx:idx+batch_size]
        idx = idx + batch_size



        num_of_words = 0
        for b in batch:
            temp = len(b)
            if (temp > num_of_words):
                num_of_words = temp


        dim1 = batch_size
        dim2 = num_of_words
        dim3 = num_chars
        #index = 0
        #temp = count + batch_size

        temp_batch = np.zeros((dim1,dim2,dim3), dtype=int)


        if (len(temp_batch) < batch_size):
            idx = 0
            
        for i in range(len(batch)):
            temp = batch[i]
            for j in range(num_of_words):
                if j < len(temp):
                    temp1 = temp[j]
                    for k in range(num_chars):
                        if k < len(temp1):
                            temp_batch[i][j][k] = temp1[k]

        return np.asarray(temp_batch), idx
        """
        mod_batch = []
        for b in batch:
            temp = self.pad_data_char(b, num_of_words, max_chars=16)
            print (np.shape(temp))
            mod_batch.append(temp)
        while (len(mod_batch) < batch_size):
            mod_batch.append(np.zeros((num_of_words, num_chars), dtype = int))
            idx = 0
            
        #mod_batch = self.pad_data(mod_batch,idx, batch_size)
        #batch = np.transpose(mod_batch)
        batch = np.asarray(mod_batch)
        print ("Token char", num_of_words, np.shape(batch), len(np.shape(batch)))
        return batch, idx
        """

    
    def next_batch(self, dt, batch_size, is_train=True):

        idx = 0
        if (is_train is True):
            idx = dt.global_count_train
        else:
            idx = dt.global_count_test

        # idx_temp to account for the fact that the number of examples
        # might not be a multiple of batch_size. We need to keep track of overflow
        # from the number of datapoints.

        contents_batch = []
        #contents_batch_char = []

        querys_batch = []
        #querys_batch_char = []
        content_batches = {}
        query_batches  = {}

        
        #print ("lengths", len(dt.content_features[i]), len(dt.content_features[i][0]))
        temp_batch_content, idx_temp = self.make_batch(dt.content_features["token"], batch_size, idx, max_length=200)
        content_batches["token"] = temp_batch_content
        cPickle.dump(temp_batch_content, open("temp.pkl", "wb"))
        temp_batch_query, idx_temp   = self.make_batch(dt.query_features["token"], batch_size, idx, max_length=30)
        query_batches["token"] = temp_batch_query

        temp_batch, idx_temp = self.make_batch_char(dt.content_features["tokenchar_"], batch_size, idx, max_length=200, num_chars=15)
        content_batches["tokenchar_"] = temp_batch

        temp_batch_query, idx_temp = self.make_batch_char(dt.query_features["tokenchar_"], batch_size, idx, max_length=200, num_chars=15)
        query_batches["tokenchar_"] = temp_batch_query


        answer_start, _ = self.make_batch(dt.answer_start, batch_size, idx, max_length = 1)
        answer_end, _   = self.make_batch(dt.answer_end, batch_size, idx, max_length = 1)
        
        ground_truths, _ = self.make_batch_gt(dt.ground_truths, batch_size, idx)
        context_words, _ = self.make_batch_gt(dt.content_features["context_words"], batch_size, idx)
        
        #print(ground_truths)
        
        if (is_train == True): 
            dt.global_count_train = idx_temp % dt.number_of_examples
        else:
            dt.global_count_test = idx_temp % dt.number_of_examples
        
        #rint (content_batches["token"])
        #print ("New INdex", idx)
        #print(ground_truths)
        #print(context_words)
        return {"content": content_batches, "query" : query_batches, "answer_start" : answer_start, "answer_end": answer_end,"ground_truths": ground_truths, "context_words": context_words}

  
