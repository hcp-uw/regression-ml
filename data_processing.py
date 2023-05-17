import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the CSV file using Pandas
df = pd.read_csv('../../../PycharmProjects/regression-ml-data/Datasets/Reddit_Data.csv')

# Preprocess the data
comments = df['clean_comment'].values.astype(str).tolist()
categories = df['category'].values.astype(str).tolist()

# Tokenize the comments using CountVectorizer
#  - fit_transform() tokenizes text data by converting a collection of documents into
#    a sparse matrix of token/word counts that only stores non-zero values 
#  - toarray() converts the sparse matrix into a dense numpy array representation
vectorizer = CountVectorizer()
tokenized_comments = vectorizer.fit_transform(comments).toarray()

# Encode the categories using LabelEncoder
#  - fit_transform() returns a new numpy array where each label is replaced 
#    with its corresponding numerical value
encoded_categories = LabelEncoder().fit_transform(categories)

# Define a custom PyTorch dataset class
class MyDataset(Dataset):
    def __init__(self, tokenized_comments, encoded_categories):
        self.tokenized_comments = tokenized_comments
        self.encoded_categories = encoded_categories

    # Returns number of comments/samples 
    def __len__(self):
        return self.tokenized_comments.shape[0]

    # Returns encoded sentiment category for a given input comment
    def __getitem__(self, idx):
        x = self.tokenized_comments[idx]
        y = self.encoded_categories[idx]
        return torch.LongTensor(x), torch.Tensor([y])

# Create a PyTorch data loader to easily input into a neural network model
# batch_size is number of batches to split our dataset into for 
# processing
batch_size = 16
dataset = MyDataset(tokenized_comments, encoded_categories)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)
    
# Set the model hyperparameters
input_dim = len(vectorizer.vocabulary_)
embedding_dim = 100
hidden_dim = 64
output_dim = 3 # number of possible encoded categories

# Initialize the model
model = LSTMClassifier(input_dim, embedding_dim, hidden_dim, output_dim)

# Set the training hyperparameters
learning_rate = 0.001
num_epochs = 10

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for batch in data_loader:
        x, y = batch
        x = torch.transpose(x, 0, 1)
        print(x)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y.squeeze().long())
        loss.backward()
        optimizer.step()

    # Print the training loss after each epoch
    print('Epoch [{}/{}], Loss: {:.4f}'
          .format(epoch+1, num_epochs, loss.item()))