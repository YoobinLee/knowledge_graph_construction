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

# Function to build a graph from a DataFrame
def build_graph(df):
    G = nx.from_pandas_edgelist(df, 'Node1', 'Node2', edge_attr='Label', create_using=nx.DiGraph())
    data = from_networkx(G)
    return data


# Build the evaluation graphs
# Replace with your actual code to build the evaluation graphs
eval_graphs_wiki = eval_wiki.groupby('Claim ID').apply(build_graph)
eval_graphs_response = eval_response.groupby('Claim ID').apply(build_graph)

##############################################
# Initialize cosine similarity function
cos_sim = CosineSimilarity(dim=1)
# Define a simple feed-forward network for computing attention scores
class Attention(nn.Module):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x)
# Load the trained variables
checkpoint = torch.load('trained_variables.pt')

# Initialize the model and optimizer
num_features_per_node = 1  # replace with actual number if nodes have features
gcn = GCNConv(num_features_per_node, 128).to('cuda')
attention = Attention(128).to('cuda').to('cuda')
optimizer = optim.Adam(gcn.parameters(), lr=0.01)

# Load the state dicts into the model and optimizer
gcn.load_state_dict(checkpoint['model_state_dict'])
attention.load_state_dict(checkpoint['attention_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

##############################################
# Set the model to evaluation mode
gcn.eval()
attention.eval()

# Initialize progress bar for evaluation
pbar_eval = tqdm(total=len(eval_graphs_wiki))

# Initialize counters for correct and total predictions
sim_predictions = 0
total_predictions = 0

# Evaluate the model
with torch.no_grad():
    for idx, (claim_id, graph) in enumerate(eval_graphs_wiki.items()):
        if claim_id not in eval_graphs_response:
            continue

        # Check if node features exist, if not create them
        node_features_wiki = graph.x.to('cuda') if graph.x is not None else torch.ones((graph.num_nodes, 1)).to('cuda')
        node_features_response = eval_graphs_response[claim_id].x.to('cuda') if eval_graphs_response[claim_id].x is not None else torch.ones((eval_graphs_response[claim_id].num_nodes, 1)).to('cuda')
        
        # Generate node embeddings
        node_embeddings_wiki = gcn(node_features_wiki, graph.edge_index.to('cuda'))
        node_embeddings_response = gcn(node_features_response, eval_graphs_response[claim_id].edge_index.to('cuda'))
        
        # Compute attention scores
        attention_scores_wiki = torch.nn.functional.softmax(attention(node_embeddings_wiki), dim=0)
        attention_scores_response = torch.nn.functional.softmax(attention(node_embeddings_response), dim=0)
        
        # Aggregate node embeddings
        graph_embeddings_wiki = torch.mean(attention_scores_wiki * node_embeddings_wiki, dim=0)
        graph_embeddings_response = torch.mean(attention_scores_response * node_embeddings_response, dim=0)
        
        # Calculate cosine similarity
        similarity = cos_sim(graph_embeddings_wiki.unsqueeze(0), graph_embeddings_response.unsqueeze(0))
        
        #print('###############wiki:',graph_embeddings_wiki,'###############response:',graph_embeddings_response)

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