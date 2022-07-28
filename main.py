# Classical imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pkg_resources import require
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import os

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
            self.cleaned_sentences, maxlen=128, dtype="long", truncating="post", padding="post")

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
        self.out = nn.Linear(self.emb_dimensions, 256)
        self.out2 = nn.Linear(256, 128)
        self.out3 = nn.Linear(128, 1)

        self.leakyRelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self, text):

        # BERT can be in inference mode
        self.bert.requires_grad = False
        with torch.no_grad():
            # Decode with BERT
            decoded_input = self.bert(text)

        # Regress on the weights
        x = self.out(decoded_input[0][:, 0, :])
        x = self.leakyRelu(x)
        x = self.out2(x)
        x = self.leakyRelu(x)
        x = self.out3(x)

        # Use sigmoid to clip the values between 0 and 1
        return self.sig(x)


# The train loop
def train(model, train_dataloader, optimizer, loss_fn, epoch, epochs):

    # Set model in train mode
    model.train()

    train_accuracy = 0
    train_loss = 0

    dataset_size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

    # Iterate through all the data
    for i, batch in tqdm(enumerate(train_dataloader)):

        # Get text and label
        text, label = batch
        tensor = text.to(device)
        label = label.to(device)

        # Clear the optimizer
        optimizer.zero_grad()

        # Do the predictions
        predictions = model(tensor)

        # Compute the loss
        loss = loss_fn(predictions.reshape(-1).float(), label.float())

        # Perform backpropagation
        loss.backward()

        # Optimize the weights
        optimizer.step()

        # Accumulate the train loss
        train_loss += loss.item()

        # Compute the accuracy
        train_accuracy += ((predictions.reshape(-1) > 0.5)
                           == label).float().sum().item()

    if not os.path.isdir('model_checkpoints'):
        os.mkdir('model_checkpoints')

    save_ckpt(model, optimizer, epoch, "model_checkpoints/model_at_epoch".format(epoch))
    print('-' * 100)
    print(f"Train Error at epoch [{epoch}/{epochs}] : Accuracy: {(100*train_accuracy/dataset_size):>0.1f}%, Avg loss: {(train_loss/num_batches):>8f} | Saved model for epoch :  {epoch}")
    print('-' * 100)


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


def read_ckpt(path,model,optimizer):

    # Load the checkpoint with the given path
    checkpoint = torch.load(path, map_location=device) 

    # Load the values
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epochs"]

    # Return them
    return model,optimizer,start_epoch

def save_model(model,path):
    torch.save(model.state_dict(), path)

def load_model(path):
    model = torch.load(path)
    model.eval()
    return model

# The test loop
def test(model, test_dataloader, loss_fn, epoch, epochs):

    # Set model in eval mode
    model.eval()

    test_accuracy = 0
    test_loss = 0

    dataset_size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)

    for i, batch in enumerate(test_dataloader):

        # Get text and label
        text, label = batch
        tensor = text.to(device)
        label = label.to(device)

        # Clear the optimizer
        optimizer.zero_grad()

        # Do the predictions
        predictions = model(tensor)

        # Compute the loss
        test_loss += loss_fn(predictions.reshape(-1).float(),
                             label.float()).item()

        # Compute the accuracy
        test_accuracy += ((predictions.reshape(-1) > 0.5)
                          == label).float().sum().item()

    # Print the results
    print(
        f"Test Error at epoch [{epoch}/{epochs}]  : Accuracy: {(100*test_accuracy/dataset_size):>0.1f}%, Avg loss: {(test_loss/num_batches):>8f}")



def encode_sample(sample):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sample = tokenizer.encode(sample, add_special_tokens=True, max_length=128)
    sample = np.array(sample).reshape(1,-1)
    print(sample.shape)
    sample = pad_sequences(sample, maxlen=128, dtype="long", truncating="post", padding="post")
    return torch.tensor(sample)

def evaluate_sample(model,sample):

    # Turn model in eval mode
    model.eval()

    # Prepare the sentence
    sample = encode_sample(sample).to(device)

    # Predict using the model
    prediction = model(sample)

    # Output predictions
    result = "positive sentiment" if prediction.squeeze(0) > 0.5 else "negative sentiment"
    print(f"Model predicted : {result} with a probability of {prediction[0,0]}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--model_save_dir',type=str, required=False, default='output_model')

    args = parser.parse_args()

    # Get the dataset
    print("Loading the dataset.....")

    dataset = np.array(pd.read_csv("dataset.csv", sep=',').sample(250))
    train_data, test_data = train_test_split(
        dataset, test_size=0.33, shuffle=True, random_state=42)
    train_data = DataLoader(MovieDataset(train_data),
                            shuffle=True, batch_size=64)
    test_data = DataLoader(MovieDataset(test_data),
                           shuffle=False, batch_size=64)

    print("Dataset has been loaded with success.....")

    # Get the model
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
    for epoch in range(args.epochs):
        train(model, train_data, optimizer, loss_function, epoch, args.epochs)
        test(model, test_data, loss_function, epoch, args.epochs)
    
    # Evaluate a single sample to see the results
    evaluate_sample(model,"I am very happy!")

    # Save our model for later inference
    model_save_dir = args.model_save_dir
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
   
    output_path = os.path.join(model_save_dir,"model.pth")
    save_model(model,output_path)