import os
import tqdm 
import numpy as np
import json
from pandas.io.json import json_normalize


class Squad_preprocessor():
    def __init__(self,tokenizer,data_directory = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\DROP"):
        self.data_directory = data_directory
        self.glove_directory = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\glove.6B"
        self.train_file = "train.json"
        self.validation_file = "dev.json"
        self.out_prefix = "train"
        self.tokenizer = tokenizer
        self.num_train_examples = 0
        self.context_lengths = None
        self.vocab = {}
        
    def load_data(self,filename = "train.json"):
        full_path = os.path.join(self.data_directory,filename)
        
        with open(full_path) as datafile:
            self.data = json.load(datafile)
            
#         print(len(self.data["data"]))
            
    def break_file(self, prefix, filename = "train.json", count_examples = False):
        self.load_data(filename)
        self.out_prefix = prefix
        l=[]

        with open('E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\DROP/drop_dataset/train.json') as fh:
            data = json.load(fh)
    
        for each in data:
            l.append(each)
        pas=[]
        code=[]
        for each in l:
            pas.append(data[each]['passage'])
            code.append(each)
        train_df=pd.DataFrame()
        train_df['passage']=pd.Series(pas)
        train_df['code']=pd.Series(code)
        ans=[]
        code=[]
        ques=[]
        for each in l:
            for i in data[each]['qa_pairs']:
                ques.append(i['question'])
                if i['answer']['number']!='':    ###If the answer is a number
                    ans.append(i['answer']['number'])
                elif i['answer']['spans']!=[]:   ###If the answer is a sentence
                    ans.append(i['answer']['spans'][-1])
                else:                            ###If the answer is a date
                    ans.append(i['answer']['date']['day']+"-"+i['answer']['date']['month']+"-"+i['answer']['date']['year'])
            code.append(each)
        di={}
        di['question']=ques
        di['code']=code
        di['answer']=ans
        ans_df=pd.DataFrame(di)
        
        file=open(prefix+"_context.txt","w",encoding="utf-8")
        for index,row in valid_df.iterrows():
            pas=tokenize(row['passage'])
            file.write(' '.join(pas)+'\n')
    
    
        file=open(prefix+"_question.txt","w",encoding="utf-8")
        for index,row in ans_df.iterrows():
            pas=tokenize(row['question'])
            file.write(' '.join(pas)+'\n')


        file=open(prefix+"_answer.txt","w",encoding="utf-8")
        for index,row in ans_df.iterrows():
            pas=row['answer']
            file.write(pas+'\n')
        #ans_df.head()       
        #valid_df.head()
    
    def conduct_preprocess(self):
        self.break_file("train", self.train_file, True)
        self.break_file("validation", self.validation_file, False)