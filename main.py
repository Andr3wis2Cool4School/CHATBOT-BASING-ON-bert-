import random 
import torch 
from model.Bert_chatbot import Config 
from model.Bert_chatbot import *
from train import *
import time
from datetime import timedelta

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def main():
    # Make sure that all random seed is the same 
    random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.deterministic = True

    
    config = Config()

    start_time = time.time()
    train_data, validation_data, test_data = config.out_data()
    train_iterator, validation_iterator, test_iterator = config.iter(train_data, validation_data, test_data)
    time_df = get_time_dif(start_time)
    print('Preparing data using {}'.format(time_df))

    transbert_encoder = TransBertEncoder(config)
    transbert_decoder = TransBertDecoder(config)
    gru_encoder = GruEncoder(2048, config.bert_emb_dim)
    gru_decoder = GruDecoder(2048, config.bert_emb_dim, config)
    dialog_dnn = DialogDNN(2048, 2048, 2048)

    model = Seq2Seq(transbert_encoder, transbert_decoder, gru_encoder, gru_decoder, dialog_dnn).to(config.device)
    print(model)

    trainer = Trainer(config, model)
    trainer.train_epoch(train_iterator)

if __name__ == '__main__':
    main()



    
