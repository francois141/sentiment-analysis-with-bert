# Classical imports
import numpy as np
from tqdm import tqdm

# Import deep learning libraries
import torch
from torch.utils.data import Dataset
from keras_preprocessing.sequence import pad_sequences

# Bert model and its tokenizer
from transformers import BertTokenizer

class MovieDataset(Dataset):

    def encode_sentence(self, sentence):
        x = ['[CLS]'] + self.tokenizer.tokenize(sentence)[:510] + ['[SEP]']
        x = self.tokenizer.convert_tokens_to_ids(x)
        return x

    def __init__(self, data):

        # Load the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)

        self.dataset = data

        # Clean the sentences
        self.cleaned_sentences = []

        for i in tqdm(range(self.dataset.shape[0])):
            self.cleaned_sentences.append(
                self.encode_sentence(self.dataset[i, 0]))

        self.cleaned_sentences = np.array(self.cleaned_sentences)
        self.cleaned_sentences = pad_sequences(
            self.cleaned_sentences, maxlen=512, dtype="int", truncating="post", padding="post")

        # Save the length of the dataset
        self.size = self.cleaned_sentences.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor(self.cleaned_sentences[idx]), torch.tensor(self.dataset[idx, 1])
