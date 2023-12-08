import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch.nn as nn

# Read the files
graph_response_df = pd.read_csv('graph_response.csv')
graph_wiki_df = pd.read_csv('graph_wiki.csv')

# Convert node and label names to lowercase
graph_response_df = graph_response_df.apply(lambda s:s.lower() if type(s) == str else s)
graph_wiki_df = graph_wiki_df.apply(lambda s:s.lower() if type(s) == str else s)

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

# print(graph_response_df[20:30])
# print(graph_wiki_df[0:20])

# Split the data into a training set and an evaluation set
train_wiki, eval_wiki = train_test_split(graph_wiki_df, test_size=0.2)
train_response, eval_response = train_test_split(graph_response_df, test_size=0.2)

from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx

# Function to build a graph from a DataFrame
def build_graph(df):
    G = nx.from_pandas_edgelist(df, 'Node1', 'Node2', edge_attr='Label', create_using=nx.DiGraph())
    data = from_networkx(G)
    return data

# Build the graphs
train_graphs_wiki = train_wiki.groupby('Claim ID').apply(build_graph)
train_graphs_response = train_response.groupby('Claim ID').apply(build_graph)

# Initialize GCN model
num_features_per_node = 1  # replace with actual number if nodes have features
gcn = GCNConv(num_features_per_node, 128)

# Define a simple feed-forward network for computing attention scores
class Attention(nn.Module):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize the attention network
attention = Attention(128).to('cuda')

import torch.optim as optim
from torch.nn import CosineSimilarity
import random

# Initialize optimizer
optimizer = optim.Adam(gcn.parameters(), lr=0.00001)

# Initialize cosine similarity function
cos_sim = CosineSimilarity(dim=1)

# Initialize progress bar
pbar = tqdm(total=len(train_graphs_wiki))

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the GPU if available
gcn = gcn.to(device)
attention = attention.to(device)

# Import the necessary module
from torch.nn.functional import pairwise_distance

# Define the contrastive loss function
def contrastive_loss(output1, output2, target, margin=1.0):
    distance = pairwise_distance(output1, output2)
    #loss = 0.5 * target * torch.pow(distance, 2) + 0.5 * (1 - target) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    loss =(1 - target) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

# Initialize a variable to store the previous graph embedding
prev_graph_embedding = None
# Define the weight factor
weight_factor = 0.9



# Train the model
for idx, (claim_id1, graph) in enumerate(train_graphs_wiki.items()):
    if claim_id1 not in train_graphs_response:
        continue

    optimizer.zero_grad()
    
    # Check if node features exist, if not create them
    node_features_wiki = graph.x if graph.x is not None else torch.ones((graph.num_nodes, 1))
    node_features_response = train_graphs_response[claim_id1].x if train_graphs_response[claim_id1].x is not None else torch.ones((train_graphs_response[claim_id1].num_nodes, 1))
    
    # Move data to the GPU if available
    node_features_wiki = node_features_wiki.to(device)
    node_features_response = node_features_response.to(device)
    graph.edge_index = graph.edge_index.to(device)
    train_graphs_response[claim_id1].edge_index = train_graphs_response[claim_id1].edge_index.to(device)
    
    # Generate node embeddings
    node_embeddings_wiki = gcn(node_features_wiki, graph.edge_index)
    node_embeddings_response = gcn(node_features_response, train_graphs_response[claim_id1].edge_index)
    
    # Compute attention scores
    attention_scores_wiki = torch.nn.functional.softmax(attention(node_embeddings_wiki), dim=0)
    attention_scores_response = torch.nn.functional.softmax(attention(node_embeddings_response), dim=0)
    
    # Aggregate node embeddings
    graph_embeddings_wiki = torch.mean(attention_scores_wiki * node_embeddings_wiki, dim=0)
    graph_embeddings_response1 = torch.mean(attention_scores_response * node_embeddings_response, dim=0)

    # Calculate loss for positive pair
    target = torch.tensor([1.0]).to(device)  # Positive pair
    loss = 1 - cos_sim(graph_embeddings_wiki.unsqueeze(0), graph_embeddings_response1.unsqueeze(0))

    # Sample a series of graphs
    negative_samples = random.sample(list(train_graphs_response.keys()), 10)

    # Calculate Euclidean distance with target embedding and select the most distant one
    max_distance = 0
    max_distance_graph_embedding = None
    for claim_id2 in negative_samples:
        if claim_id2 == claim_id1:
            continue

        graph_response2 = train_graphs_response[claim_id2]

        # Generate node embeddings for graph_response2
        node_features_response2 = graph_response2.x.to(device) if graph_response2.x is not None else torch.ones((graph_response2.num_nodes, 1)).to(device)
        node_embeddings_response2 = gcn(node_features_response2, graph_response2.edge_index.to(device))
        attention_scores_response2 = torch.nn.functional.softmax(attention(node_embeddings_response2), dim=0)
        graph_embeddings_response2 = torch.mean(attention_scores_response2 * node_embeddings_response2, dim=0)

        # Calculate Euclidean distance
        distance = torch.dist(graph_embeddings_wiki, graph_embeddings_response2)

        # Update max_distance and max_distance_graph_embedding if necessary
        if distance > max_distance:
            max_distance = distance
            max_distance_graph_embedding = graph_embeddings_response2

    # Calculate loss for negative pair
    target = torch.tensor([0.0]).to(device)  # Negative pair
    contra= weight_factor * contrastive_loss(graph_embeddings_wiki.unsqueeze(0), max_distance_graph_embedding.unsqueeze(0), target)
    loss = (1-weight_factor)*loss + contra
    #print('######contra#####',contra)
    if claim_id1%32 == 0:
        print(f'Epoch {idx+1}, Loss: {loss.item()}')
    loss.backward()
    optimizer.step()

    # Update progress bar
    pbar.update(1)

# Close progress bar
pbar.close()

# Save the trained variables
torch.save({
    'model_state_dict': gcn.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'attention_state_dict': attention.state_dict(),
}, 'trained_variables.pt')

# Build the evaluation graphs
eval_graphs_wiki = eval_wiki.groupby('Claim ID').apply(build_graph)
eval_graphs_response = eval_response.groupby('Claim ID').apply(build_graph)

# Initialize progress bar for evaluation
pbar_eval = tqdm(total=len(eval_graphs_wiki))

# Initialize counters for correct and total predictions
sim_predictions = 0
total_predictions = 0

# Evaluation
gcn.eval()
attention.eval()

# Evaluate the model
gcn.eval()
attention.eval()
with torch.no_grad():
    for idx, (claim_id, graph) in enumerate(eval_graphs_wiki.items()):
        if claim_id not in eval_graphs_response:
            continue

        # Check if node features exist, if not create them
        node_features_wiki = graph.x if graph.x is not None else torch.ones((graph.num_nodes, 1))
        node_features_response = eval_graphs_response[claim_id].x if eval_graphs_response[claim_id].x is not None else torch.ones((eval_graphs_response[claim_id].num_nodes, 1))
        
        # Move data to the GPU if available
        node_features_wiki = node_features_wiki.to(device)
        node_features_response = node_features_response.to(device)
        graph.edge_index = graph.edge_index.to(device)
        eval_graphs_response[claim_id].edge_index = eval_graphs_response[claim_id].edge_index.to(device)
        
        # Generate node embeddings
        node_embeddings_wiki = gcn(node_features_wiki, graph.edge_index)
        node_embeddings_response = gcn(node_features_response, eval_graphs_response[claim_id].edge_index)
        
        # Compute attention scores
        attention_scores_wiki = torch.nn.functional.softmax(attention(node_embeddings_wiki), dim=0)
        attention_scores_response = torch.nn.functional.softmax(attention(node_embeddings_response), dim=0)
        
        # Aggregate node embeddings
        graph_embeddings_wiki = torch.mean(attention_scores_wiki * node_embeddings_wiki, dim=0)
        graph_embeddings_response = torch.mean(attention_scores_response * node_embeddings_response, dim=0)
        
        # Calculate cosine similarity
        similarity = cos_sim(graph_embeddings_wiki.unsqueeze(0), graph_embeddings_response.unsqueeze(0))
        
        # Update counters
        total_predictions += 1
        if claim_id % 32 == 0:
            print(f'Claim {claim_id}, Similarity: {similarity.item()}')
        sim_predictions += similarity.item()

    # Close progress bar
    pbar_eval.close()

    print(sim_predictions)
    # Calculate accuracy
    accuracy = sim_predictions / total_predictions
    print(f'Similarity Ave: {accuracy}')