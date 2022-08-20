
# Classical imports
from sklearn.model_selection import train_test_split


# Import deep learning libraries
import torch
import torch.nn as nn


# Bert model and its tokenizer
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertTokenizer

from utils import *
from model import *
from dataset import *


class SentimentAnalysisModel(nn.Module):

    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()

        # We use BERT as encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.emb_dimensions = self.bert.config.to_dict()['hidden_size']

        # We can treat this problem as a regression one
        self.linear = nn.Linear(self.emb_dimensions, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()

    def forward(self, text):

        # BERT can be in inference mode
        self.bert.requires_grad = False
        with torch.no_grad():
            # Decode with BERT
            decoded_input = self.bert(text)

        # Regress on the weights
        x = self.dropout(decoded_input[0][:, 0, :])
        x = self.linear(x)
        return self.sigmoid(x)

