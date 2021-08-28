import torch
import argparse
import numpy as np

from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_config
from data_loader import get_loader

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi','mosei','ur_funny'],
                    help='dataset to use (default: mosei)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')
parser.add_argument('--div_dropout', type=float, default=0.1)

# Embedding
parser.add_argument('--use_bert', action='store_true', help='whether to use bert \
                    to encode text inputs (default: False)')

# Losses
parser.add_argument('--lambda_d', type=float, default=0.1, help='portion of discriminator loss added to total loss (default: 0.1)')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--attn_hidden_size', type=int, default=40,
                    help='The size of hiddens in all transformer blocks')
parser.add_argument('--uni_nlevels', type=int, default=3,
                    help='number of transformer blocks for unimodal attention')
parser.add_argument('--enc_layers', type=int, default=1,
                    help='Layers of GRU or LSTM in sequence encoder')
parser.add_argument('--use_disc', action='store_true',
                    help='whether to add a discriminator to the domain-invariant encoder and the corresponding loss to the final training process')

parser.add_argument('--proj_type', type=str, default='cnn',help='network type for input projection', choices=['LINEAR', 'CNN','LSTM','GRU'])
parser.add_argument('--lksize', type=int, default=3,
                    help='Kernel size of language projection CNN')
parser.add_argument('--vksize', type=int, default=3,
                    help='Kernel size of visual projection CNN')
parser.add_argument('--aksize', type=int, default=3,
                    help='Kernel size of accoustic projection CNN')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()

# set numpy manual seed
# np.random.seed(args.seed)

torch.manual_seed(args.seed)
valid_partial_mode = args.lonly + args.vonly + args.aonly

# configurations for data_loader
dataset = str.lower(args.dataset.strip())
batch_size = args.batch_size

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8,
    # 'ur_funny': 1   # comment this if using BCELoss
    'ur_funny': 2 # comment this if using CrossEntropyLoss
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

####################################################################
#######       Load the dataset (aligned or non-aligned)       ######
####################################################################

print("Start loading the data....")

train_config = get_config(dataset, mode='train', batch_size=args.batch_size, use_bert=args.use_bert)
valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size, use_bert=args.use_bert)
test_config = get_config(dataset, mode='test',  batch_size=args.batch_size, use_bert=args.use_bert)

# print(train_config)

hyp_params = args

# pretrained_emb saved in train_config here
train_loader = get_loader(hyp_params, train_config, shuffle=True)
valid_loader = get_loader(hyp_params, valid_config, shuffle=False)
test_loader = get_loader(hyp_params, test_config, shuffle=False)

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

torch.autograd.set_detect_anomaly(True)

# addintional appending
hyp_params.word2id = train_config.word2id
hyp_params.pretrained_emb = train_config.pretrained_emb

# architecture parameters
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_config.lav_dim
if hyp_params.use_bert:
    hyp_params.orig_d_l = 768
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_config.lav_len
hyp_params.layers = args.nlevels
hyp_params.l_ksize = args.lksize
hyp_params.v_ksize = args.vksize
hyp_params.a_ksize = args.aksize

hyp_params.proj_type = args.proj_type.lower()
hyp_params.num_enc_layers = args.enc_layers

hyp_params.use_cuda = use_cuda
hyp_params.dataset = hyp_params.data = dataset
hyp_params.when = args.when
hyp_params.attn_dim = args.attn_hidden_size
hyp_params.batch_chunk = args.batch_chunk
# hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = train_len, valid_len, test_len
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
# hyp_params.criterion = criterion_dict.get(dataset, 'MAELoss')
hyp_params.criterion = criterion_dict.get(dataset, 'MSELoss')


if __name__ == '__main__':
    solver = Solver(hyp_params, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    solver.train_and_eval()
    exit()

