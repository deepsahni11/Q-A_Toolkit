# import os
#
# class Config(object):
#     pass
#
# config = Config()
# # E:\Internships_19\Internship(Summer_19)\Q&A_Toolkit\Dynamic_Coattention_Networks\Models\saved_models
# config.data_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\"
# config.model_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dynamic_Coattention_Networks\\Models\\saved_models\\"
# # config.data_dir = "Dynamic_Coattention_Networks/data/"
# # config.model_dir = "Dynamic_Coattention_Networks/Models/saved_models/"
# config.word_embedding_size = 100
# config.hidden_dim = 100
# config.bidirectional = True
# config.mid_processing = True
# # config.dropout_ratio = 0.15
# config.encoder_type = "bi-lstm"
# config.cross_interaction_type = "bidaf"
# config.self_interaction_type = "bidaf"
# config.decoder_type = "bidaf"
# config.num_layers = 1
# config.use_char_emb = False
# config.use_word_emb = True
# config.fine_tune = False
# config.depth = 2
# config.max_context_length = 600
# config.max_question_length = 30
# config.max_char_length = 5
# config.word_emb_size = 100
# config.char_emb_size = 20
# config.max_words=700
# config.lr = 0.001
# config.dropout = 0.15
# config.dropout_ratio = 0.15
# config.early_stop = 10
# config.vocab_size = 50000
#
#
# #vector with zeros for unknown words
# config.num_iterations = 2
# config.maxout_pool_size=16
# config.max_chars=10
# config.sentinel = False
# config.lr = 0.001
# config.dropout_ratio = 0.15
# config.query_non_linearity = True
# config.max_grad_norm = 5.0
# config.batch_size = 20
# config.num_epochs = 2
# config.print_every = 100
# config.max_number_of_iterations = 5
# config.print_and_validate_every = 2
# # config.optim = "Adadelta"
# # config.decoder = "dcn"
# config.print_every = 100
# config.save_every = 50
# config.eval_every = 100
#
# # config.model_type = 'co-attention'
# config.reg_lambda = 0.00007
# config.names = ["train_context","train_question"]
# config.print_every = 100
# import os



class Config(object):

    pass



config = Config()

# E:\Internships_19\Internship(Summer_19)\Q&A_Toolkit\Dynamic_Coattention_Networks\Models\saved_models

config.data_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\"

config.model_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dynamic_Coattention_Networks\\Models\\saved_models\\"
config.num_epochs = 2
# config.data_dir = "data/"
#
# config.model_dir = "Models/saved_models/"
config.batch_size = 20
config.word_embedding_size = 300
config.names = ["train_context","train_question"]
config.hidden_dim = 512

config.bidirectional = True

config.mid_processing = True
config.print_every = 100
config.save_every = 50
config.eval_every = 100
# config.dropout_ratio = 0.15

config.encoder_type = "bi-lstm"

config.cross_interaction_type = "bidaf"

config.self_interaction_type = "bidaf"
config.print_and_validate_every = 2
config.decoder_type = "bidaf"

config.num_layers = 1

config.use_char_emb = False

config.use_word_emb = True

config.fine_tune = False

config.depth = 2

config.max_context_length = 250

config.max_question_length = 30

config.max_char_length = 5

config.word_emb_size = 300

config.char_emb_size = 20

config.max_words=250

config.lr = 0.0004

config.dropout = 0.2

config.dropout_ratio = 0.2

config.early_stop = 10

config.vocab_size = 50000





#vector with zeros for unknown words

config.num_iterations = 20

config.maxout_pool_size=16

config.max_chars=10

config.sentinel = False

config.dropout_ratio = 0.15

config.query_non_linearity = True

config.max_grad_norm = 5.0
