import time
import math
import torch
import unicodedata
from typing import Tuple
import pandas as pd
from collections import defaultdict
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from config import EOS_TOKEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f'Number of records in the dataset: {len(df)}')
    print(f'Checking for NaN values: ', df.isna().sum())
    # Remove NaN values
    df = df.dropna()
    print(
        f'Number of records in the dataset after removing NaN values: {len(df)}')
    return df


class Lang:
    def __init__(self):
        # self.word2index = {"<SOS>": 0, "<EOS>": 1, "<SEP>": 2}
        self.word2index = {}
        self.word2count = defaultdict(int)
        # self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<SEP>"}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence: str):
        for word in word_tokenize(sentence):
            self.addWord(word)

    def addWord(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        self.word2count[word] += 1

    @staticmethod
    def normalizeString(s: str):
        # s = unicodeToAscii(s.lower().strip())
        s = s.lower().strip()
        return s


# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#     )
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in word_tokenize(sentence)]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    # indexes.append(lang.word2index[EOS_token])
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(lang, row):
    input_tensor = tensorFromSentence(lang, row["generated_qn"])
    target_tensor = tensorFromSentence(lang, row["question"])
    return (input_tensor, target_tensor)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def split_dataset(dataset: pd.DataFrame, test_ratio: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        dataset, test_size=test_ratio, random_state=42)
    return train, test
