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


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(min(input_length, 50)):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_TOKEN:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, df, lang, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(lang, df.iloc[random.randint(0, len(df)-1), :])
                      for _ in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, n_iters + 1)):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
    }, MODEL_SAVE_PATH)


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
    for i in range(n):
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



if __name__ == "__main__":
    df: pd.DataFrame = read_dataset(DATASET_PATH)
    train_df, test_df = split_dataset(df, test_ratio=0.2)
    train_df, val_df = split_dataset(train_df, test_ratio=0.1)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    print(f'device: {device}')
    lang = prepare_data(pd.concat([train_df, val_df], ignore_index=True))
    # hidden_size = 256
    encoder1 = EncoderRNN(lang.n_words, HIDDEN_SIZE).to(device)
    attn_decoder1 = AttnDecoderRNN(
        HIDDEN_SIZE, lang.n_words, dropout_p=DROPOUT).to(device)
    trainIters(encoder1, attn_decoder1, NUM_OF_ITERATIONS,
               train_df, lang, print_every=5000)

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    output = evaluateRandomly(encoder1, attn_decoder1, test_df, lang, n=10, sample=False)
    output.to_csv(RESULT_PATH)