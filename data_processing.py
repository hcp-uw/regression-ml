import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the CSV file using Pandas
df = pd.read_csv('my_file.csv')

# Preprocess the data
comments = df['clean_comment'].values.tolist()
categories = df['category'].values.tolist()

# Tokenize the comments using CountVectorizer
#  - fit_transform() tokenizes text data by converting a colleciton of documents into
#    a sparse matrix of token/word counts that only stores non-zero values 
#  - toarray() converts the sparse matrix into a dense numpy array representation
vectorizer = CountVectorizer()
tokenized_comments = vectorizer.fit_transform(comments).toarray()

# Encode the categories using LabelEncoder
#  - fit_transform() returns a new numpy array where each label is replaced 
#    with its corresponding numerical value
#   - fit() the possible encodings are limited to _____ 
label_encoder = LabelEncoder()
label_encoder.fit([-1, 0, 1])
encoded_categories = label_encoder.fit_transform(categories)

# Define a custom PyTorch dataset class
class MyDataset(Dataset):
    def __init__(self, tokenized_comments, encoded_categories):
        self.tokenized_comments = tokenized_comments
        self.encoded_categories = encoded_categories

    # Returns number of comments/samples 
    def __len__(self):
        return len(self.tokenized_comments)

    # Returns encoded sentiment category for a given input comment
    def __getitem__(self, idx):
        x = self.tokenized_comments[idx]
        y = self.encoded_categories[idx]
        return torch.Tensor(x), torch.Tensor([y])

# Create a PyTorch data loader to easily input into a neural network model
# batch_size is number of batches to split our dataset into for 
# processing
batch_size = 32
dataset = MyDataset(tokenized_comments, encoded_categories)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)