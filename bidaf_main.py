from numpy import genfromtxt
from torch.autograd import Variable
from torch.nn import Embedding
from torch import zeros, from_numpy, Tensor, LongTensor, FloatTensor
from argparse import ArgumentParser
import torch
import train_model
import bidaf_model
class Config(object):
    pass
	
if __name__ == '__main__':	

	config.max_num_sent=1
    config.max_num_sent_words=740
    config.max_sent_size=740
    config.max_ques_size=36
    config.num_layers=1
    config.word_vocab_size=46531
    config.char_vocab_size=74
    config.max_word_size=15
    config.glove_vec_size=100
    config.char_embed_type='CNN'
    config.word_emb_size=100
    config.rnn_bidirectional=True
    config.early_stop=5

    config.documentAwareQuery_method='method1'
    config.queryAwareDocument_method='method_2'
    config.mid_processing='bidaf'
    config.is_train='True'
    config.resume='False'
    config.epochs=12

    config.model_name="basic"
    config.data_dir="D:/Downloads/SQuAD/"
    config.run_id="0"
    config.outdir=".D:/Downloads/SQuAD/"
    config.emb_dir="D:/Downloads/SQuAD/"
    

    # Device placement config.device", type=str, default="/cpu:0", help="default device for summing gradients. [/cpu:0]")
    config.device_type="gpu"
    config.num_gpus=1

    # Essential training and test options
    config.mode="train"
    config.load=True
    #config.single", type=bool, default=False, help="supervise only the answer sentence? [False]")
    config.debug=False
    #config.load_ema", type=bool, default=True, help="load exponential average of variables when testing?  [True]")
    config.eval=True
    #config.wy", type=bool, default=False, help="Use wy for loss / eval? [False]")
    #config.na", type=bool, default=False, help="Enable no answer strategy and learn bias? [False]")
    config.th=0.5

    # Training / test parameters
    config.batch_size=60
    #config.val_num_batches", type=int, default=100, help="validation num batches [100]")
    #config.test_num_batches", type=int, default=0, help="test num batches [0]")
    config.num_epochs=12
    #config.num_steps", type=int, default=20000, help="Number of steps [20000]")
    #config.load_step", type=int, default=0, help="load step [0]")
    config.optim="Adadelta"
    config.lr=0.5
    config.input_keep_prob=0.8
    config.keep_prob=0.8
    config.wd=0.0
    config.hidden_size=100
    config.char_out_size=100
    config.char_emb_out_size=100
    config.char_emb_size=8
    config.out_channel_dims="100"
    config.filter_heights="5"
    config.finetune=False
    config.highway"=True
    config.highway_num_layers=2
    config.share_cnn_weights=True
    config.share_lstm_weights=True
    config.var_decay=0.999
    config.lstm_layers=1
    config.batch_first=True
    config.print_frequency=100

    # Optimizations
    #config.cluster", type=bool, default=False, help="Cluster data for faster training [False]")
    #config.len_opt", type=bool, default=False, help="Length optimization? [False]")
    #config.cpu_opt", type=bool, default=False, help="CPU optimization? GPU computation can be slower [False]")

    # Logging and saving options
    config.progress=True
    config.log_period=100
    config.eval_period=1000
    config.save_period=1000
    config.max_to_keep=20 
    config.dump_eval=True
    config.dump_answer=True
    config.vis=False
    config.dump_pickle=True
    config.decay=0.9

    # Thresholds for speed and less memory usage
    config.word_count_th=10
    config.char_count_th=50
    config.sent_size_th=400
    config.num_sents_th=8
    config.ques_size_th=30
    config.word_size_th=16
    config.para_size_th=256

    # Advanced training options
    config.lower_word=True
    config.squash=False
    config.swap_memory=True
    config.data_filter="max"
    config.use_glove_for_unk=True
    config.known_if_glove=True
    config.logit_func="tri_linear"
    config.answer_func="linear"
    config.sh_logit_func="tri_linear"

    # Ablation options
    config.use_char_emb=True

    config.use_word_emb=True
    config.q2c_att=True
    config.c2q_att=True
    config.dynamic_att=False
	model = bidaf_model.BiDAF(config)
    
	model().cpu()
	 
    train_model = train_model.TrainModel(config, model)

    train_model.run_training()
