from numpy.core.fromnumeric import shape
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from transformers import BertTokenizer, BertModel
from torchtext.data import Field, BucketIterator, TabularDataset
import warnings 
warnings.simplefilter('ignore', UserWarning)




class Config(object):
    def __init__(self):
        self.model_name = 'BERT-Chatbot'
        self.batch_size = 12
        self.n_epochs = 3
        
        self.bert_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']
        self.bert = BertModel.from_pretrained(self.bert_name)
        self.bert_emb_dim = self.bert.config.to_dict()['hidden_size']
        
        self.bert_n_head = 8
        self.clip = 1
        self.teacher_forcing_ratio = 1

        self.saved_dir = './saved_dir/'


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.SRC = Field(use_vocab=False,
                         tokenize=self.tokenize_and_cut,
                         preprocessing=self.tokenizer.convert_tokens_to_ids,
                         init_token=self.tokenizer.cls_token_id,
                         eos_token=self.tokenizer.sep_token_id,
                         pad_token=self.tokenizer.pad_token_id,
                         unk_token=self.tokenizer.unk_token_id)

        self.TGT = Field(use_vocab=False,
                         tokenize=self.tokenize_and_cut,
                         preprocessing=self.tokenizer.convert_tokens_to_ids,
                         init_token=self.tokenizer.cls_token_id,
                         eos_token=self.tokenizer.sep_token_id,
                         pad_token=self.tokenizer.pad_token_id,
                         unk_token=self.tokenizer.unk_token_id)

        self.data_fields = [('src', self.SRC), ('tgt', self.TGT)]
    

    def tokenize_and_cut(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:self.max_input_length-2]
        return tokens

    def out_data(self):
        train_data, validation_data, test_data = TabularDataset.splits(path = './data/',
                                                                        format='csv',
                                                                        train='chatbotdata1.csv',
                                                                        validation='dev.csv',
                                                                        test='test.csv',
                                                                        skip_header=True,
                                                                        fields= self.data_fields)
        return train_data, validation_data, test_data

    def iter(self, train_data, validation_data, test_data):
        train_iterator, validation_iterator, test_iterator = BucketIterator.splits((train_data, validation_data, test_data),
                                                                                   batch_size=self.batch_size,
                                                                                   sort_key=lambda x: len(x.src),  # function used to group the data
                                                                                   sort_within_batch=False,
                                                                                   device=self.device)
        return train_iterator, validation_iterator, test_iterator 


class PositionalEncoding(nn.Module):
    def __init__(self, model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model, 2).float() * (-math.log(10000.0) / model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransBertEncoder(nn.Module):
    def __init__(self, config, dropout=0.5):
        super().__init__()

        # bert encoder
        self.bert = config.bert

        # transformer encoder, as bert last layer fine-tune
        self.pos_encoder = PositionalEncoding(config.bert_emb_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=config.bert_emb_dim, nhead=config.bert_n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)

    def forward(self, src):
        # src = [src len, batch size]

        with torch.no_grad():
            # embedded = [src len, batch size, emb dim]
            embedded = self.bert(src.transpose(0, 1))[0].transpose(0, 1)
            # embedded = self.bert(torch.transpose(src, 0, 1))[0].transpose(0, 1)
            # print(src)
            # print(type(src))
            # temp0 = src.transpose(0,1)
            # temp = self.bert(temp0)
            # temp1 = temp
            # print(type(temp1))
            # print(temp1)
            # print(len(temp1))
            # embedded = temp1.transpose(0, 1)
           

        # embedded = self.pos_encoder(embedded)

        # src_mask = nn.Transformer().generate_square_subsequent_mask(len(embedded)).to(g_device)

        # outputs = [src len, batch size, hid dim * n directions]
        outputs = self.transformer_encoder(embedded)

        return outputs



class TransBertDecoder(nn.Module):
    def __init__(self, config, dropout=0.5):
        super().__init__()

        # bert encoder
        self.bert = config.bert
        self.bert_emb_dim = config.bert_emb_dim
        self.device = config.device
        self.tokenizer = config.tokenizer
        self.vocab_size = config.vocab_size

        self.pos_decoder = PositionalEncoding(config.bert_emb_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.bert_emb_dim, nhead=config.bert_n_head)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        self.fc_out = nn.Linear(config.bert_emb_dim, config.vocab_size)

    def forward(self, tgt, meaning, teacher_forcing_ratio):
        # tgt = [output_len, batch size]

        output_len = tgt.size(0)
        batch_size = tgt.size(1)
        # decide if we are going to use teacher forcing or not
        teacher_force = random.random() < teacher_forcing_ratio

        if teacher_force and self.training:
            tgt_emb_total = torch.zeros(output_len, batch_size, self.bert_emb_dim).to(self.device)

            for t in range(0, output_len):
                with torch.no_grad():
                    tgt_emb = self.bert(tgt[:t+1].transpose(0, 1))[0].transpose(0, 1)
                tgt_emb_total[t] = tgt_emb[-1]

            tgt_mask = nn.Transformer().generate_square_subsequent_mask(len(tgt_emb_total)).to(self.device)
            decoder_output = self.transformer_decoder(tgt=tgt_emb_total,
                                                      memory=meaning,
                                                      tgt_mask=tgt_mask)
            predictions = self.fc_out(decoder_output)
        else:
            # initialized the input of the decoder with sos_idx (start of sentence token idx)
            output = torch.full((output_len+1, batch_size), self.tokenizer.cls_token_id, dtype=torch.long, device=self.device)
            predictions = torch.zeros(output_len, batch_size, self.vocab_size).to(self.device)

            for t in range(0, output_len):
                with torch.no_grad():
                    tgt_emb = self.bert(output[:t+1].transpose(0, 1))[0].transpose(0, 1)
                    

                # tgt_emb = [t, batch size, emb dim]
                # tgt_emb = self.pos_encoder(tgt_emb)

                tgt_mask = nn.Transformer().generate_square_subsequent_mask(len(tgt_emb)).to(self.device)

                # decoder_output = [t, batch size, emb dim]
                decoder_output = self.transformer_decoder(tgt=tgt_emb,
                                                          memory=meaning,
                                                          tgt_mask=tgt_mask)

                # prediction = [batch size, vocab size]
                prediction = self.fc_out(decoder_output[-1])

                # predictions = [output_len, batch size, vocab size]
                predictions[t] = prediction

                one_hot_idx = prediction.argmax(1)

                # output  = [output len, batch size]
                output[t+1] = one_hot_idx

        return predictions        


class GruEncoder(nn.Module):
    """compress the request embeddings to meaning"""

    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input):
        output, hidden = self.gru(input)
        return hidden


class GruDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, config):
        super().__init__()
        self.gru = nn.GRU(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.bert_emb_dim = config.bert_emb_dim
        self.device = config.device


    def forward(self, src, tgt, hidden):
        # first input to the decoder is the <CLS> tokens
        fc_output = src[0].unsqueeze(0)
        tgt_len = tgt.size(0)
        batch_size = tgt.size(1)

        # tensor to store decoder outputs
        outputs = torch.zeros(tgt_len, batch_size, self.bert_emb_dim).to(self.device)

        for t in range(0, tgt_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            gru_output, hidden = self.gru(fc_output, hidden)

            fc_output = self.fc(gru_output)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = fc_output
        return outputs


class DialogDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # ResNet, dropout on first 3 layers
        input = self.dropout(input)

        output = input + F.relu(self.fc1(input))
        output = self.dropout(output)

        output = output + F.relu(self.fc2(output))
        output = self.dropout(output)

        output = output + self.fc3(output)  # no relu to keep negative values

        return output


class Seq2Seq(nn.Module):
    def __init__(self, transbert_encoder, transbert_decoder, gru_encoder, gru_decoder, dialog_dnn):
        super().__init__()

        self.transbert_encoder = transbert_encoder
        self.transbert_decoder = transbert_decoder

        self.gru_encoder = gru_encoder
        self.gru_decoder = gru_decoder

        self.dialog_dnn = dialog_dnn

    def forward(self, src, tgt, teacher_forcing_ratio):
        request_embeddings = self.transbert_encoder(src)
        request_meaning = self.gru_encoder(request_embeddings)

        
        response_meaning = request_meaning

        response_embeddings = self.gru_decoder(request_embeddings, tgt, response_meaning)
        response = self.transbert_decoder(tgt, response_embeddings, teacher_forcing_ratio)

        return response




