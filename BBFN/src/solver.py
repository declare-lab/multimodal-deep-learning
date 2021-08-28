import torch
from torch import nn
import sys
import models
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
# from utils import DiffLoss, MSE, SIMSE, CMD
from utils import CMD, MSE
from utils.eval_metrics import *
from utils.tools import *
from models import MULTModel

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model

        # loss function
        self.sim_loss = CMD()

        # initialize the model
        if model is None:
            self.model = model = MULTModel(hp)
        
        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.hp.use_bert:
            if self.hp.pretrained_emb is not None:
                self.model.embedding.embed.weight.data = self.hp.pretrained_emb
            self.model.embedding.embed.requires_grad = False

        if hp.use_cuda:
            model = model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # optimizer
        if self.is_train:
            self.optimizer = getattr(torch.optim, self.hp.optim)(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.hp.lr)
        
        # criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
            # self.criterion = criterion = nn.BCELoss()
        else: # mosi and mosei are regression datasets
            self.criterion = criterion = nn.MSELoss(reduction="mean")
        
        if self.hp.use_disc:
            self.criterion_d = nn.BCELoss()
            self.lambda_d = hp.lambda_d

        # Final list
        for name, param in self.model.named_parameters():
            # Bert freezing customizations 
            if self.hp.data in ["mosei", "mosi"]:
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= 8:
                        param.requires_grad = False
            elif self.hp.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            # print('\t' + name, param.requires_grad)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=20, factor=0.1, verbose=True)


    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = self.criterion
        if self.hp.use_disc:
            criterion_d = self.criterion_d
        else:
            criterion_d = None

        # ctc_a2l_module, ctc_v2l_module = self.ctc_a2l_module, self.ctc_v2l_module
        # ctc_a2l_optimizer, ctc_v2l_optimizer = self.ctc_a2l_optimizer, self.ctc_v2l_optimizer
        # ctc_criterion = self.ctc_criterion


        def train(model, optimizer, criterion, criterion_d, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion):
            epoch_loss = 0
            best_valid_loss = float('inf')

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            start_time = time.time()

            train_losses = []
            valid_losses = []



            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch_data
                
                # if i_batch == 2:
                #     break

                model.zero_grad()
                if ctc_criterion is not None:
                    ctc_a2l_module.zero_grad()
                    ctc_v2l_module.zero_grad()
                    
                if self.hp.use_cuda:
                    with torch.cuda.device(0):
                        text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                        text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                        bert_sent_type.cuda(), bert_sent_mask.cuda()
                        if self.hp.dataset=="ur_funny":
                            y = y.squeeze()
                
                batch_size = y.size(0)
                batch_chunk = self.hp.batch_chunk
                    
                combined_loss = 0
                net = nn.DataParallel(model) if batch_size > 10 else model

                # If parallel fails due to limited space, then increase batch_chunk to decrease GPU utilization
                if batch_chunk > 1:
                    pass
                else:
                    preds, disc_preds, disc_trues = model(text, visual, audio, l, bert_sent, bert_sent_type, bert_sent_mask)
                    # if self.hp.dataset == "ur_funny":
                    #     y = y.unsqueeze(-1)
                    raw_loss = criterion(preds, y)
                    combined_loss = raw_loss

                    if self.hp.use_disc:
                        disc_loss = criterion_d(disc_preds, disc_trues)
                        combined_loss += self.lambda_d * disc_loss
                    combined_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                optimizer.step()
                
                proc_loss += raw_loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += combined_loss.item() * batch_size
                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, avg_loss))
                    proc_loss, proc_size = 0, 0
                    start_time = time.time()
                    
            return epoch_loss / self.hp.n_train

        def evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, criterion_d, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0
        
            results = []
            truths = []

            enc_lv_l_all = []
            enc_lv_v_all = []
            enc_la_l_all = []
            enc_la_a_all = []

            with torch.no_grad():
                for batch in loader:
                    text, vision, audio, y, lengths, bert_sent, bert_sent_type, bert_sent_mask = batch
                    # eval_attr = y.squeeze(dim=-1) # if num of labels is 1

                    if self.hp.use_cuda:
                        with torch.cuda.device(0):
                            text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                            lengths = lengths.cuda()
                            bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                            if self.hp.dataset == 'iemocap':
                                y = y.long()
                        
                            if self.hp.dataset == 'ur_funny':
                                y = y.squeeze()

                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)
                    
                    if (ctc_a2l_module is not None) and (ctc_v2l_module is not None):
                        ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                        ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module
                        audio, _ = ctc_a2l_net(audio)     # audio aligned to text
                        vision, _ = ctc_v2l_net(vision)   # vision aligned to text

                    # we don't need disc loss here anymore
                    preds, disc_preds, disc_truths = model(text, vision, audio, lengths, bert_sent, bert_sent_type, bert_sent_mask)

                    
                    # if self.epoch in [1,2,5,6,7]:
                    #     enc_lv_l_all.append(enc_res['lv_l'].detach().cpu().numpy())
                    #     enc_lv_v_all.append(enc_res['lv_v'].detach().cpu().numpy())
                    #     enc_la_l_all.append(enc_res['la_l'].detach().cpu().numpy())
                    #     enc_la_a_all.append(enc_res['la_a'].detach().cpu().numpy())

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    total_loss += criterion(preds, y).item() * batch_size

                    if self.hp.use_disc and not test:
                        if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti']:
                            total_l1_loss += F.l1_loss(preds, y).item() * batch_size
                        elif self.hp.dataset == "ur_funny":
                            # crossentropyloss only
                            total_l1_loss = total_loss
                        loss_d = criterion_d(disc_preds, disc_truths)
                        # valid loss is MSE + discriminator
                        total_loss += self.lambda_d * loss_d * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
            
            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)
            if not test:
                avg_l1_loss = total_l1_loss / self.hp.n_valid
            else:
                avg_l1_loss = 0.0

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, avg_l1_loss, results, truths

        best_valid = 1e8
        best_mae = 1e8
        patience = 20

        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch

            train(model, optimizer, criterion, criterion_d, None, None, None, None, None)
            val_loss, val_f1_loss, _, _ = evaluate(model, None, None, criterion, criterion_d, test=False)
            test_loss, _, results, truths = evaluate(model, None, None, criterion, criterion_d, test=True)
            
            end = time.time()
            duration = end-start
            scheduler.step(val_loss)    # Decay learning rate by validation loss

            # validation F1
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Valid L1 Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, val_f1_loss, test_loss))
            print("-"*50)
            

            if best_valid > val_loss:
                patience = 20
                best_valid = val_loss
                # for ur_funny we don't care about
                if self.hp.dataset == "ur_funny":
                    eval_humor(results, truths, True)
                elif test_loss < best_mae:
                    best_mae = test_loss
                    if self.hp.dataset in ["mosei_senti", "mosei"]:
                        # np.save('result/preds_epoch{}.npy'.format(self.epoch), results.detach().cpu().numpy())
                        # np.save('result/truths.npy', truths.detach().cpu().numpy())
                        eval_mosei_senti(results, truths, True)
                    elif self.hp.dataset == 'mosi':
                        eval_mosi(results, truths, True)
                    elif self.hp.dataset == 'iemocap':
                        eval_iemocap(results, truths)
                
                    print(f"Saved model at pre_trained_models/{self.hp.name}.pt!")
                    save_model(self.hp, model, name=self.hp.name)
                    # best_valid = test_loss
            else:
                patience -= 1
                if patience == 0:
                    break

        model = load_model(self.hp, name=self.hp.name)
        print(f'Best epoch: {best_epoch}')

        sys.stdout.flush()