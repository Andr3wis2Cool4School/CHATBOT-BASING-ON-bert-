import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time 
import math 

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Trainer(object):
    def __init__(self, config, model):
        self.model_name = config.model_name
        self.clip = config.clip
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.num_epochs = config.n_epochs
        self.model = model 
        self.saved_dir = config.saved_dir
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.tokenizer.pad_token_id)
        self.freeze_bert_params()
        self.init_weights()
        self.printpara()

    def freeze_bert_params(self):
        for name, param in self.model.named_parameters():
            if name.find('bert.') != -1:
                param.requires_grad = False
    

    def printpara(self):
        para = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'The model has {para:,} trainable parameters')

    def init_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                nn.init.normal_(param.data, mean=0, std=0.01)

    def train(self, train_iterator):
        self.model.train()
        print('Batch num: ' + str(len(train_iterator)))

        epoch_loss = 0
        for i, batch in enumerate(train_iterator):
            if i % 50 == 0:
                print('')
            print(i, end=',')

            src = batch.src
            tgt = batch.tgt

            self.optimizer.zero_grad()
            
            output = self.model(src, tgt[:-1], self.teacher_forcing_ratio)

            output_dim = output.shape[-1]

            # tgt = [(tgt len - 1) * batch size]
            # output = [(tgt len - 1) * batch size, output dim]
            loss = self.criterion(output[:-1].view(-1, output_dim), tgt[1:-1].view(-1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_iterator)

    def evaluate(self, validation_iterator):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(validation_iterator):
                src = batch.src
                tgt = batch.tgt

                tu_off_teacher_forcing_ratio = 0
                output = self.model(src, tgt[0:-1], tu_off_teacher_forcing_ratio)

                output_dim = output.shape[-1]

                loss = self.criterion(output.view(-1, output_dim), tgt[1:].view(-1))

                epoch_loss += loss.item()

        return epoch_loss / len(validation_iterator)


    def train_epoch(self, train_iterator):
        
        self.model.load_state_dict(torch.load('saved_dir/BERT-Chatbot.pt'))
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            start_time = time.time()

            train_loss = self.train(train_iterator)
            # validation_loss = self.evaluate(validation_iterator)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'\nEpoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            # print(f'\t Val. Loss: {validation_loss:.3f} |  Val. PPL: {math.exp(validation_loss):7.3f}')

            torch.save(self.model.state_dict(), self.saved_dir + self.model_name+'.pt')



            


