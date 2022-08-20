# Classical imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pkg_resources import require
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

# Import text specific packages
from bs4 import BeautifulSoup
import re
import contractions
import nltk
from nltk.stem import WordNetLemmatizer 

# Import deep learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from keras_preprocessing.sequence import pad_sequences

# Bert model and its tokenizer
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#####################################################
# Data cleaning and encoding helpers
#####################################################

def clean_dataframe(dataframe,verbose=False):
    dataframe = remove_null_values(dataframe,verbose)
    dataframe = remove_duplicates(dataframe,verbose)
    dataframe = remove_html_tags(dataframe,verbose)
    dataframe = remove_urls(dataframe,verbose)
    dataframe = remove_emoji(dataframe,verbose)
    dataframe = lower_words(dataframe,verbose)
    dataframe = remove_digits(dataframe,verbose)
    dataframe = remove_extra_space(dataframe,verbose)
    dataframe = lemmatize(dataframe,verbose)
    dataframe = remove_contractions(dataframe,verbose)
    return dataframe

def remove_null_values(dataframe,verbose=False):
    if verbose:
        print('Removing all rows with a null entry')
    dataframe.dropna(axis=0,how='any')
    return dataframe

def remove_duplicates(dataframe,verbose=False):
    if verbose:
        print('Removing all duplicates inside of the dataset')
    dataframe.drop_duplicates(inplace = True)
    return dataframe

def remove_html_tags(dataframe,verbose=False):
    if verbose:
        print('Removing all html tags')
    html_cleaner = lambda text : BeautifulSoup(text, "html.parser").get_text()
    dataframe['review'] = dataframe['review'].apply(html_cleaner)
    return dataframe

def remove_urls(dataframe,verbose=False):
    if verbose:
        print('Removing all url')
    url_cleaner = lambda text : re.sub(r'http\S+', '', text)
    dataframe['review'] = dataframe['review'].apply(url_cleaner)
    return dataframe
    
def remove_emoji(dataframe,verbose=False):
    if verbose:
        print('Removing all emojis')
    emoji_clean= re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "]+", flags=re.UNICODE)

    emoji_cleaner = lambda text : emoji_clean.sub(r'',text)
    dataframe['review'] = dataframe['review'].apply(emoji_cleaner)
    return dataframe

def lower_words(dataframe,verbose=False):
    if verbose:
        print('Lowering all words')
    words_lower = lambda text: text.lower()
    dataframe['review'] = dataframe['review'].apply(words_lower)
    return dataframe
    
def remove_digits(dataframe,verbose=False):
    if verbose:
        print('Removing all numbers')
    digit_cleaner = lambda text: ''.join([i for i in text if not i.isdigit()])
    dataframe['review'] = dataframe['review'].apply(digit_cleaner)
    return dataframe

def remove_extra_space(dataframe,verbose=False):
    if verbose:
        print('Removing all extra space')
    #extra_space_cleaner = lambda text: " ".join(text.split())
    #dataframe.apply['review'] = dataframe['review'].apply(extra_space_cleaner)
    return dataframe

def lemmatize(dataframe,verbose=False):
    if verbose:
        print('Lemmatizing the words')
    nltk.download('punkt')
    nltk.download('wordnet')
    lemmatizer = lambda text: ' '.join([WordNetLemmatizer().lemmatize(w) for w in nltk.word_tokenize(text)])
    dataframe['review'] = dataframe['review'].apply(lemmatizer) 
    return dataframe

def remove_contractions(dataframe,verbose=False):
    if verbose:
        print('Removing all contractions')
    contraction_cleaner = lambda text: contractions.fix(text)
    dataframe['review'] = dataframe['review'].apply(contraction_cleaner)
    return dataframe

def encode_sample(sample):
    sample = tokenizer.encode(sample, add_special_tokens=True, max_length=512)
    sample = np.array(sample).reshape(1, -1)
    print(sample.shape)
    sample = pad_sequences(sample, maxlen=512, dtype="long",
                           truncating="post", padding="post")
    return torch.tensor(sample)

#####################################################
# I/0 for model
#####################################################

def save_ckpt(model, optimizer, epochs, ckpt_path):

    # Create a dictionnary
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["epochs"] = epochs

    # Create the checkpoint
    prefix, ext = os.path.splitext(ckpt_path)
    ckpt_path = "{}-{}{}".format(prefix, epochs, ext)
    torch.save(checkpoint, ckpt_path)


def read_ckpt(path, model, optimizer):

    # Load the checkpoint with the given path
    checkpoint = torch.load(path, map_location=device)

    # Load the values
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epochs"]

    # Return them
    return model, optimizer, start_epoch


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = torch.load(path)
    model.eval()
    return model