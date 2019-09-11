# from Models.Bidirectional_attention_flow_model import *
from Models.dynamic_coattention_model import *
from Preprocessing_Layer_0.Vocabulary_builder import *
from Preprocessing_Layer_0.Squad_processor import *
from Preprocessing_Layer_0.Embedding_Matrix import *



# from Models.config_bidaf import *
from Models.config_dcn import *
from train import *
import os




preprocess = Squad_preprocessor(nltk.word_tokenize,config.data_dir)
preprocess.conduct_preprocess()

# data_directory = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD"
context_file = open(os.path.join(config.data_dir, 'train.context'), 'r', encoding='utf-8').readlines()
question_file = open(os.path.join(config.data_dir, 'train.question'), 'r', encoding='utf-8').readlines()
answer_text_file = open(os.path.join(config.data_dir,  'train.answer_text'), 'r', encoding= 'utf-8').readlines()
answer_start_file = open(os.path.join(config.data_dir,  'train.answer_start'), 'r', encoding= 'utf-8').readlines()
answer_end_file = open(os.path.join(config.data_dir,  'train.answer_end'), 'r', encoding= 'utf-8').readlines()



with open(os.path.join(config.data_dir, "train.context" ), 'w', encoding='utf-8') as f:
    for item in context_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(config.data_dir, "train.question" ), 'w', encoding='utf-8') as f:
    for item in question_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(config.data_dir, "train.answer_text" ), 'w', encoding='utf-8') as f:
    for item in answer_text_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(config.data_dir, "train.answer_start" ), 'w', encoding='utf-8') as f:
    for item in answer_start_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(config.data_dir, "train.answer_end" ), 'w', encoding='utf-8') as f:
    for item in answer_end_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(config.data_dir, "validation.context"), 'w', encoding='utf-8') as f:
    for item in context_file[:8000]:
        f.write("%s" % item)
with open(os.path.join(config.data_dir, "validation.question" ), 'w', encoding='utf-8') as f:
    for item in question_file[:8000]:
        f.write("%s" % item)
with open(os.path.join(config.data_dir, "validation.answer_text" ), 'w', encoding='utf-8') as f:
    for item in answer_text_file[:8000]:
        f.write("%s" % item)
with open(os.path.join(config.data_dir, "validation.answer_start"), 'w', encoding='utf-8') as f:
    for item in answer_start_file[:8000]:
        f.write("%s" % item)
with open(os.path.join(config.data_dir, "validation.answer_end" ), 'w', encoding='utf-8') as f:
    for item in answer_end_file[:8000]:
        f.write("%s" % item)


vocab = Vocabulary([config.data_dir + "train.context",config.data_dir + "train.question"],config.data_dir + "vocab.dat")
vocab.create_vocabulary(0,config.vocab_size, config.data_dir)

embedding = Embedding_Matrix(config)
embedding.get_glove_embeddings(config.word_embedding_size, config.char_embedding_size ,config.data_dir)
embedding.index_files_to_char_level_and_word_level(config.data_dir , config.max_words, config.max_chars)
embedding_matrix = pickle.load(open(os.path.join(config.data_dir, "glove_word_embeddings" + ".pkl"), 'rb'))



with autograd.set_detect_anomaly(True):
    # model = Bidaf_Model(config)
    model = DCN_Model(config,embedding_matrix)

    if(config.use_gpu == True):
        model = model.cuda()

    # config = Config()
    # model = model.cpu()
    train_model = Train_Model(config, model)

    train_model = train_model.cuda()
    train_model.train()
