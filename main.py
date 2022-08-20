# Classical imports
from logging.handlers import DatagramHandler
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

from utils import *
from model import *
from dataset import *


# The train loop
def train(model, train_dataloader, optimizer, loss_fn, epoch, epochs):

    # Set model in train mode
    model.train()

    train_accuracy = 0
    train_loss = 0

    dataset_size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

    print('-' * 100)

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

        if i % 1 == 0:
            print(f"Train loss [{i}|{num_batches}]: {(loss.item()):>8f} ")

    if not os.path.isdir('model_checkpoints'):
        os.mkdir('model_checkpoints')

    save_ckpt(model, optimizer, epoch,
              "model_checkpoints/model_at_epoch".format(epoch))
    print('-' * 100)
    print(f"Train Error at epoch [{epoch}/{epochs}] : Accuracy: {(100*train_accuracy/dataset_size):>0.1f}%, Avg loss: {(train_loss/num_batches):>8f} | Saved model for epoch :  {epoch}")
    print('-' * 100)

# The test loop
def test(model, test_dataloader, loss_fn, epoch, epochs):

    # Set model in eval mode    print("Dataset has been loaded with success.....")
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



def evaluate_sample(model, sample):

    # Turn model in eval mode
    model.eval()

    # Prepare the sentence
    sample = encode_sample(sample).to(device)

    # Predict using the model
    prediction = model(sample)

    # Output predictions
    result = "positive sentiment" if prediction.squeeze(
        0) > 0.5 else "negative sentiment"
    print(
        f"Model predicted : {result} with a probability of {prediction[0,0]}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--model_save_dir', type=str,
                        required=False, default='output_model')

    args = parser.parse_args()

    # Get the dataset
    print("Loading the dataset.....")

    dataframe = pd.read_csv("dataset.csv", sep=',')
    dataframe_cleaned = clean_dataframe(dataframe,verbose=True)
    dataset = np.array(dataframe)


    # Load the dataset into a dataloader
    train_data, test_data = train_test_split(
        dataset, test_size=0.33, shuffle=True, random_state=42)
    train_data = DataLoader(MovieDataset(train_data),
                            shuffle=True, batch_size=16)
    test_data = DataLoader(MovieDataset(test_data),
                           shuffle=False, batch_size=16)

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
    loss_function = torch.nn.BCELoss().to(device)

    # Send the model to the correct device
    model = model.to(device)

    # Train it
    for epoch in range(args.epochs):
        train(model, train_data, optimizer, loss_function, epoch, args.epochs)
        test(model, test_data, loss_function, epoch, args.epochs)

    # Evaluate a single sample to see the results
    evaluate_sample(model, "I am very happy!")

    # Save our model for later inference
    model_save_dir = args.model_save_dir
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    output_path = os.path.join(model_save_dir, "model.pth")
    save_model(model, output_path)