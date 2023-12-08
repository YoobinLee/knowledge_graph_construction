import torch
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
import torch.optim as optim
from torch.nn import CosineSimilarity
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import pandas as pd

##############################################
# Read the files
graph_response_df = pd.read_csv('graph_response_greedy.csv')
graph_wiki_df = pd.read_csv('graph_wiki_greedy.csv')

# Convert node and label names to lowercase
graph_response_df = graph_response_df.applymap(lambda s:s.lower() if type(s) == str else s)
graph_wiki_df = graph_wiki_df.applymap(lambda s:s.lower() if type(s) == str else s)

# Combine the nodes and labels from both dataframes for fitting the LabelEncoder
combined_nodes = pd.concat([graph_response_df['Node1'], graph_response_df['Node2'], graph_wiki_df['Node1'], graph_wiki_df['Node2']])
combined_labels = pd.concat([graph_response_df['Label'], graph_wiki_df['Label']])

# Fit the LabelEncoder on the combined nodes and labels
le_nodes = LabelEncoder().fit(combined_nodes)
le_labels = LabelEncoder().fit(combined_labels)

# Transform the nodes and labels in the individual dataframes
graph_response_df['Node1'] = le_nodes.transform(graph_response_df['Node1'])
graph_response_df['Node2'] = le_nodes.transform(graph_response_df['Node2'])
graph_response_df['Label'] = le_labels.transform(graph_response_df['Label'])

graph_wiki_df['Node1'] = le_nodes.transform(graph_wiki_df['Node1'])
graph_wiki_df['Node2'] = le_nodes.transform(graph_wiki_df['Node2'])
graph_wiki_df['Label'] = le_labels.transform(graph_wiki_df['Label'])

eval_wiki = graph_wiki_df
eval_response = graph_response_df

def build_graph(df):
    G = nx.from_pandas_edgelist(df, 'Node1', 'Node2', edge_attr='Label', create_using=nx.DiGraph())
    data = from_networkx(G)
    # Use the node IDs as node features
    data.x = torch.tensor(list(G.nodes), dtype=torch.float).view(-1, 1)
    # Convert edge attributes to a tensor and store them in data.edge_attr
    edge_attrs = [G.get_edge_data(*e).get('Label', 0) for e in G.edges()]
    data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)
    return data

# Build the evaluation graphs
# Replace with your actual code to build the evaluation graphs
eval_graphs_wiki = eval_wiki.groupby('Claim ID').apply(build_graph)
eval_graphs_response = eval_response.groupby('Claim ID').apply(build_graph)

##############################################
import random
import torch.optim as optim
from torch.nn.functional import cosine_similarity
import pandas as pd
from tqdm import tqdm

import torch.nn.functional as F

# Initialize the GCN model and the attention network
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class Attention(nn.Module):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=0)

# Initialize the GCN model
gcn = GCN(num_features=1, num_classes=128)
attention = Attention(in_features=128)
# Training
optimizer = optim.Adam(list(gcn.parameters()) + list(attention.parameters()), lr=0.001)

# Load the model and optimizer states
checkpoint = torch.load('trained_variables.pt')
gcn.load_state_dict(checkpoint['gcn_state_dict'])
attention.load_state_dict(checkpoint['attention_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Ensure the model is in evaluation mode
gcn.eval()

# Generate the graph embeddings and compare their similarity
similarities = []
for claim_id, graph_wiki in tqdm(eval_graphs_wiki.items()):
    if claim_id in eval_graphs_response:
        # Generate the graph embeddings
        graph_wiki_embedding = gcn(graph_wiki)
        graph_response_embedding = gcn(eval_graphs_response[claim_id])

        # Apply the attention model to the embeddings
        attn1 = attention(graph_wiki_embedding)
        attn2 = attention(graph_response_embedding)

        #mean
        graph_wiki_embedding = torch.mean(attn1*graph_wiki_embedding, dim=0)
        graph_response_embedding = torch.mean(attn2*graph_response_embedding, dim=0)
        # Compute the cosine similarity of the embeddings
        simi = cosine_similarity(graph_wiki_embedding.unsqueeze(0), graph_response_embedding.unsqueeze(0)).mean().item()
        similarities.append(simi)

# Compute the average similarity
average_similarity = sum(similarities) / len(similarities)
print('Average similarity:', average_similarity)
