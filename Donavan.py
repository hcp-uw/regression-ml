import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the CSV file using Pandas
df = pd.read_csv('Datasets/Reddit_Data.csv')

# Preprocess the data
comments = df['clean_comment'].values.astype(str).tolist()
categories = df['category'].values.astype(str).tolist()

# Tokenize the comments using CountVectorizer
vectorizer = CountVectorizer()
tokenized_comments = vectorizer.fit_transform(comments)

# Encode the categories using LabelEncoder
encoded_categories = LabelEncoder().fit_transform(categories)

print(tokenized_comments)
print(encoded_categories)

# Set the model hyperparameters
input_dim = len(vectorizer.vocabulary_)
hidden_dim = 64

# Initialize the model
model = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True)

# Set the training hyperparameters
learning_rate = 0.001
num_epochs = 10

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create a PyTorch dataset and data loader
batch_size = 16
dataset = torch.utils.data.TensorDataset(tokenized_comments, encoded_categories)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
for epoch in range(num_epochs):
    for batch in data_loader:
        x, y = batch
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Print the training loss after each epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
