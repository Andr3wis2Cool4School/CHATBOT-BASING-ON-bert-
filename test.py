from model.Bert_chatbot import *
import csv 
import torch 
import torch.nn as nn 



config = Config()
transbert_encoder = TransBertEncoder(config)
transbert_decoder = TransBertDecoder(config)
gru_encoder = GruEncoder(2048, config.bert_emb_dim)
gru_decoder = GruDecoder(2048, config.bert_emb_dim, config)
dialog_dnn = DialogDNN(2048, 2048, 2048)

model = Seq2Seq(transbert_encoder, transbert_decoder, gru_encoder, gru_decoder, dialog_dnn).to(config.device)
model.load_state_dict(torch.load('saved_dir/BERT-Chatbot.pt'))

# print chatbot's words
def print_chat(sentences):
    print("chatbot: ", end="")
    for word_embeds in sentences:
        word_embed = word_embeds[0]
        # find one shot index from word embedding
        max_idx_t = word_embed.argmax()
        max_idx = max_idx_t.item()
        word = config.tokenizer.convert_ids_to_tokens(max_idx)
        print(word, end=" ")
    print("")  # new line at the end of sentence


def print_tgt(sentences):
    print("tgt: ", end="")
    for word_embeds in sentences:
        word_embed = word_embeds[0]
        max_idx = word_embed.item()
        word = config.tokenizer.convert_ids_to_tokens(max_idx)
        print(word, end=" ")
    print("")  # new line at the end of sentence




def create_file_store(filename, sentence):
    f_csv = open('saved_dir/' + filename + '.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow([sentence, sentence])

def chat(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.tgt

            teacher_forcing_ratio = 0
            output = model(src, tgt[0:-1], teacher_forcing_ratio)
            print_chat(output)
            print_tgt(tgt)

            output_dim = output.shape[-1]

            # tgt = [(tgt len - 1) * batch size]
            # output = [(tgt len - 1) * batch size, output dim]
            loss = criterion(output.view(-1, output_dim), tgt[1:].view(-1))

            print(f'\t Val. Loss: {loss:.3f} |  Val. PPL: {math.exp(loss):7.3f}')

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def user_talk(num):
    criterion = nn.CrossEntropyLoss(ignore_index=config.tokenizer.pad_token_id)
    for i in range(num):
        user_input = input('user:')
        create_file_store('chat_demo', user_input)
        test_data = TabularDataset(
            path = 'saved_dir/chat_demo.csv',
            format='csv',
            skip_header=False,
            fields = config.data_fields
        )

        test_iterator = BucketIterator(
            test_data,
            batch_size= config.batch_size,
            sort_key = lambda x: len(x.src),
            sort_within_batch=False,
            device=config.device
        )

        chat(model, test_iterator, criterion)



if __name__ == '__main__':
    user_talk(5)

