from tqdm import tqdm
import pandas as pd
import random
import time
import torch
import torch.nn as nn
from torch import optim

from model import AttnDecoderRNN, EncoderRNN, DecoderRNN
from config import (
    DATASET_PATH,
    MODEL_SAVE_PATH,
    MAX_LENGTH,
    TEACHER_FORCING_RATIO,
    SOS_TOKEN,
    EOS_TOKEN,
    HIDDEN_SIZE,
    NUM_OF_ITERATIONS,
    DROPOUT,
    RESULT_PATH
)
from utils import (
    Lang,
    read_dataset,
    timeSince,
    tensorsFromPair,
    split_dataset,
    tensorFromSentence
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(df: pd.DataFrame) -> Lang:
    lang = Lang()
    columns = ['context', 'question', 'generated_qn']
    for col in columns:
        for rec in df[col].to_list():
            lang.addSentence(rec)
    return lang



def evaluate(encoder, decoder, sentence, lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, test_df, lang, n=10, sample=True):
    results = []
    for i in tqdm(range(n)):
        sent = test_df.loc[random.randint(0, len(test_df)-1), ['generated_qn', 'question']] if sample else test_df.loc[i, ['generated_qn', 'question']]
        pair = tensorsFromPair(lang, sent)
        # print('>', pair[0])
        # print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, sent[0], lang)
        output_sentence = ' '.join(output_words)
        # print('<', output_sentence)
        # print('')
        results.append([ sent[0], sent[1], output_sentence ])
    
    return pd.DataFrame(results, columns=['generated_qn', 'question', 'improved_qn'])

if __name__ == '__main__':
    torch.cuda.empty_cache()
    df: pd.DataFrame = read_dataset(DATASET_PATH)
    train_df, test_df = split_dataset(df, test_ratio=0.2)
    train_df, val_df = split_dataset(train_df, test_ratio=0.1)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    print(f'device: {device}')
    lang = prepare_data(pd.concat([train_df, val_df], ignore_index=True))

    hidden_size = HIDDEN_SIZE
    encoder1 = EncoderRNN(lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, lang.n_words, dropout_p=DROPOUT).to(device)


    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    # encoder1.load_state_dict(checkpoint['encoder'])
    # attn_decoder1.load_state_dict(checkpoint['decoder'])

    # print('Loaded Model weights')

    output = evaluateRandomly(encoder1, attn_decoder1, test_df, lang, n=test_df.shape[0], sample=False)
    output.to_csv(RESULT_PATH)