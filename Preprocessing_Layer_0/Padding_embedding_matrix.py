import random
import nltk
import numpy as np
import pickle
import sys
import copy
import os.path



class Data_pad:

    def __init__(self, data_path="E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD"):

        
        self.data_path = data_path
        self.out_prefix = "train"
        

 
   
    def find_max_length(self, data):

        """ Finds the maximum sequence length for data 
            Args:
                data: The data from which sequences will be chosen
               
        """
        temp = 0
        for i, _ in enumerate(data):
            
            if (len(data[i]) > temp):
                temp = len(data[i])
        
        return temp
    
    def pickle_padded_sequence(self, prefix):
        self.out_prefix = prefix
       
        with codecs.open(self.data_path + "\\" + prefix + "_word_index.context_pkl.pkl", "rb") as input_file:
            context_word_index = pickle.load(input_file)
        with codecs.open(self.data_path + "\\" + prefix + "_word_index.context_pkl.pkl", "rb") as input_file:
            question_word_index = pickle.load(input_file)

        self.pad_data(self.out_prefix,context_word_index)
        self.pad_data(self.out_prefix,question_word_index)

    def pad_data(self,prefix,data):

        """ Pad the data to max_length given
            Args: 
                data: Data that needs to be padded
                max_length : The length to be achieved with padding
            Returns:
                padded_data : Each sequence is padded to make it of length
                              max_length.
        """

        self.out_prefix = prefix
        padded_data = []
        max_length =  self.find_max_length(data)

        for lines in tqdm.tqdm(data):
            if (len(lines) < max_length):
                temp = np.lib.pad(lines, (0,max_length - len(lines)),
                    'constant', constant_values=0)
            else:
                temp = lines[:max_length]
            padded_data.append(temp)

        padded_data = torch.from_numpy(np.array(padded_data)).int()
        pickle.dump(padded_data, codecs.open(os.path.join(self.data_path, self.out_prefix + "_word_index_padded.pkl"), "wb")) 
   
