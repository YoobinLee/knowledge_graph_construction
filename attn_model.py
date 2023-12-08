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
from torch.nn.functional import mse_loss
import numpy as np

##############################################
# Read the files
graph_response_df = pd.read_csv('graph_response_sampled.csv')
graph_wiki_df = pd.read_csv('graph_wiki_sampled.csv')

# Convert node and label names to lowercase
graph_response_df = graph_response_df.applymap(lambda s:s.lower() if type(s) == str else s)
graph_wiki_df = graph_wiki_df.applymap(lambda s:s.lower() if type(s) == str else s)

# Remove or encode symbols in 'Node1', 'Node2', and 'Label'
graph_response_df = graph_response_df.replace({'<': 'symbol1', '%': 'symbol2', '[': 'symbol3'})
graph_wiki_df = graph_wiki_df.replace({'<': 'symbol1', '%': 'symbol2', '[': 'symbol3'})

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

##############################################
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
    
# class Attention(nn.Module):
#     def __init__(self, in_features):
#         super(Attention, self).__init__()
#         self.linear = nn.Linear(in_features, 1)

#     def forward(self, x):
#         return F.softmax(self.linear(x), dim=0)

class Attention(nn.Module):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.in_features = in_features
        # rest of your code

    def forward(self, x):
        # Compute uniform attention scores
        attn = torch.ones_like(x) / x.size(0)
        return attn

# Masking function
def mask_graph(data, mask_rate=0.15):
    # Ensure data.x is defined and is a tensor
    if data.x is None:
        data.x = torch.ones((data.num_nodes, 1))

    # Create a mask for nodes
    node_mask = torch.rand(data.num_nodes) < mask_rate
    data.x[node_mask] = 0

    # Create a mask for edges
    edge_mask = torch.rand(data.edge_index.shape[1]) < mask_rate

    # Apply the mask to the edge_index by removing the masked edges
    data.edge_index = data.edge_index[:, ~edge_mask]

# Initialize the GCN model
gcn = GCN(num_features=1, num_classes=128)
attention = Attention(in_features=128)
# Training
optimizer = optim.Adam(list(gcn.parameters()) + list(attention.parameters()), lr=0.001)

# Load the model and optimizer states
checkpoint = torch.load('trained_variables.pt')
#print("Model parameters before loading checkpoint:")
# for param in gcn.parameters():
#     print(param.data)
gcn.load_state_dict(checkpoint['gcn_state_dict'])
attention.load_state_dict(checkpoint['attention_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#print("Model parameters after loading checkpoint:")
# for param in gcn.parameters():
#     print(param.data)

# Ensure the model is in evaluation mode
gcn.eval()
print("Model is in training mode:", gcn.training)

# Generate the graph embeddings and compare their similarity
distances = []
for claim_id, graph_wiki in tqdm(eval_graphs_wiki.items()):
    if claim_id in eval_graphs_response:
        # Generate the graph embeddings
        #print("Input to GCN model:", graph_wiki,'claim id',claim_id)
        graph_wiki_embedding = gcn(graph_wiki)
        #print("Output of GCN model:", graph_wiki_embedding,'claim id',claim_id)
        graph_response_embedding = gcn(eval_graphs_response[claim_id])

        # Apply the attention model to the embeddings
        #print("Input to attention network attn1:", graph_wiki_embedding)
        attn1 = attention(graph_wiki_embedding)
        #print("Output of attention network attn1:", attn1)
        #print("Input to attention network attn2:", graph_response_embedding)
        attn2 = attention(graph_response_embedding)
        #print("Output of attention network attn2:", attn2)

        #Compute the weighted average of the embeddings
        print("Input to weighted average:", graph_wiki_embedding, graph_response_embedding)
        graph_wiki_embedding = torch.sum(attn1*graph_wiki_embedding, dim=0)
        graph_response_embedding = torch.sum(attn2*graph_response_embedding, dim=0)
        print("Output of weighted average:", graph_wiki_embedding, graph_response_embedding)

        # # Sample 3 random graphs from the array
        # negative_samples = np.random.choice(eval_graphs_response, 3, replace=False)
        # max_distance = 0
        # max_distance_graph_emb = None
        # for negative_graph in negative_samples:
        #     if torch.equal(negative_graph.x, graph_wiki.x):
        #         continue
        #     negative_emb = gcn(negative_graph)
        #     negative_attn = attention(negative_emb)
        #     negative_graph_emb = torch.mean(negative_attn * negative_emb, dim=0)
        #     distance = torch.dist(graph_wiki_embedding, negative_graph_emb)
        #     if distance > max_distance:
        #         max_distance = distance
        #         max_distance_graph_emb = negative_graph_emb

        # print("graph_wiki_embedding in hexadecimal:", graph_wiki_embedding)
        # print("graph_response_embedding in hexadecimal:", graph_response_embedding)

        # Compute the Euclidean distance of the embeddings
        dist = torch.dist(graph_wiki_embedding, graph_response_embedding).item()
        dist = cosine_similarity(graph_wiki_embedding.unsqueeze(0), graph_response_embedding.unsqueeze(0)).item()+1
        # semi = cosine_similarity(graph_wiki_embedding.unsqueeze(0), max_distance_graph_emb.unsqueeze(0)) + 1
        # #print('dist',dist)
        # print('semi',semi)
        distances.append(dist)

# Compute the average distance
average_distance = sum(distances) / len(distances)
print('Average distance:', average_distance)