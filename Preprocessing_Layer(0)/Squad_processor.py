import os
import tqdm 
import numpy as np
import json


class Squad_preprocessor():
    def __init__(self,tokenizer,data_directory = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD"):
        self.data_directory = data_directory
        self.glove_directory = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\glove.6B"
        self.train_file = "train_v2.json"
        self.validation_file = "validation_v2.json"
        self.out_prefix = "train"
        self.tokenizer = tokenizer
        self.num_train_examples = 0
        self.context_lengths = None
        self.vocab = {}
        
    def load_data(self,filename = "train_v2.json"):
        full_path = os.path.join(self.data_directory,filename)
        
        with open(full_path) as datafile:
            self.data = json.load(datafile)
            
#         print(len(self.data["data"]))
            
    def break_file(self, prefix, filename = "train_v2.json", count_examples = False):
        self.load_data(filename)
        self.out_prefix = prefix
        
        ##### creating data directories for different parts of the data namely:
        ## 1) context
        ## 2) question
        ## 3) answer_text
        ## 4) answer_start
        ## 5) answer_end
      
        
        ###       the SQuAD dataset has the following layout:
        # "data" ---> "title", "paragraphs" 
        #                            |
        #                            -----> "context" , "qas"
        #                                                 |
        #                                                 -----> "answers", "id", "is_impossible", "question"
        #
        #    ie. one context has several questions and their respective answers     

        
        with open(os.path.join(self.data_directory, self.out_prefix +'.context'), 'w', encoding='utf-8') as context_file, \
             open(os.path.join(self.data_directory, self.out_prefix +'.question'), 'w', encoding='utf-8') as question_file, \
             open(os.path.join(self.data_directory, self.out_prefix + '.answer_text'), 'w', encoding= 'utf-8') as answer_text_file, \
             open(os.path.join(self.data_directory, self.out_prefix + '.answer_start'), 'w', encoding= 'utf-8') as answer_start_file, \
             open(os.path.join(self.data_directory, self.out_prefix + '.answer_end'), 'w', encoding= 'utf-8') as answer_end_file:
                   
                    for article_idx in tqdm.tqdm(range(len(self.data["data"]))):
                        paragraphs = self.data["data"][article_idx]["paragraphs"] ## all the paragraphs in data directory

                        for paragraph_idx in range(len(paragraphs)):
                            context = paragraphs[paragraph_idx]["context"] ## each context in a given paragraph directory
                            context = context.lower()
                            context_tokens = self.tokenizer(context)

                            ## each context has a range of "answers", "id", "is_impossible", "question" 

                            qas = paragraphs[paragraph_idx]["qas"] ##  "qas" referrring to a single "context"

                            for qas_idx in range(len(qas)):  ### disecting the "qas" into "answers", "id", "is_impossible", "question" 
                                question = qas[qas_idx]["question"]  
                                question = question.lower()
                                question_tokens = self.tokenizer(question)

                                ## we select the first answer id from the range of answers we are given for a particular question
                                
                                if(len(qas[qas_idx]["answers"]) == 0 ):
                                    
                                    answer_text_tokens = "<unk>"
                                    word_level_answer_start = -1
                                    word_level_answer_end = -1
                                    
                                    
                                    
                                else:
                                    
                                    answer_id = 0
                                    answer_text = qas[qas_idx]["answers"][answer_id]["text"]
#                                     print(answer_text)
                                    
                                    answer_text = answer_text.lower()
                                    answer_text_tokens = self.tokenizer(answer_text) ## we atke the first option as the answer

                                    char_level_answer_start = qas[qas_idx]["answers"][answer_id]["answer_start"]
                                    word_level_answer_start = len(context[:char_level_answer_start].split())
                                    word_level_answer_end = word_level_answer_start + len(answer_text.split()) - 1


                                context_file.write(' '.join(token for token in context_tokens)+'\n')
                                question_file.write(' '.join(token for token in question_tokens)+'\n')
                                answer_text_file.write(' '.join(token for token in answer_text_tokens)+'\n')
                                answer_start_file.write(str(word_level_answer_start)+ "\n")
                                answer_end_file.write(str(word_level_answer_end) + "\n")

    
    def conduct_preprocess(self):
        self.break_file("train", self.train_file, True)
        self.break_file("validation", self.validation_file, False)
        
            
            
    
    