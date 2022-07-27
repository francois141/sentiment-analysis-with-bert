# Classical imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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

class MovieDataset(Dataset):

    def encode_sentence(self, sentence):
        return self.tokenizer.encode(sentence, add_special_tokens=True, max_length=128)

    def __init__(self):
        
        # Load the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Load the csv data
        self.dataset = pd.read_csv("dataset.csv", sep=',').to_numpy()

        # Clean the sentences
        self.cleaned_sentences = []

        for i in tqdm(range(self.dataset.shape[0])):
            self.cleaned_sentences.append(self.encode_sentence(self.dataset[i,0]))

        self.cleaned_sentences = np.array(self.cleaned_sentences)
        self.cleaned_sentences = pad_sequences(self.cleaned_sentences, maxlen=128, dtype="long", truncating="post", padding="post")

        # Save the length of the dataset
        self.size = self.cleaned_sentences.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor(self.cleaned_sentences[idx]), torch.tensor(self.dataset[idx, 1])


class SentimentAnalysisModel(nn.Module):

    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()

        # We use BERT as encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.emb_dimensions = self.bert.config.to_dict()['hidden_size']

        # We can treat this problem as a regression one
        self.out = nn.Linear(self.emb_dimensions, 1)
        self.sig = nn.Sigmoid()

    def forward(self, text):
        
        # BERT can be in inference mode
        self.bert.requires_grad = False
        with torch.no_grad():
            # Decode with BERT
            decoded_input = self.bert(text)


        # Regress on the weights
        x = self.out(decoded_input[0][:, 0, :])

        # Use sigmoid to clip the values between 0 and 1
        return self.sig(x)


losses = []

# The train loop
def train(model, train_dataloader, optimizer, criterion):

    # Iterate through all the data
    for i, batch in enumerate(train_dataloader):

        # Get text and label
        text, label = batch
        tensor = text.to(device)
        label = label.to(device)

        # Clear the optimizer
        optimizer.zero_grad()

        # Do the predictions
        predictions = model(tensor)

        # Compute the loss 
        loss = criterion(predictions.reshape(-1).float(), label.float())

        # Perform backpropagation
        loss.backward()

        # Optimize the weights
        optimizer.step()

        # Refresh the statistics
        if i % 20 == 0:
            print(loss.item())
            losses.append(loss.item())


# Load dataset and model
print("Loading the dataset.....")
train_dataloader = DataLoader(MovieDataset(), shuffle=True, batch_size=16)
print("Dataset has been loaded with success.....")

print("Loading the model.....")
model = SentimentAnalysisModel()
print("Model has been loaded with success.....")

# Get the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device is : {}".format(device))

# Load optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_function = torch.nn.MSELoss().to(device)

# Send the model to the correct device
model = model.to(device)

# Train it
for i in range(2000):
    train(model, train_dataloader, optimizer, loss_function)

# Output the loss
plt.plot(np.array(losses))
plt.show()
