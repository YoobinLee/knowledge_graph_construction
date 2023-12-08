import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx

# Load the data
graph_wiki_real_df = pd.read_csv('graph_wiki_real.csv')

# Preprocess the data
graph_wiki_real_df = graph_wiki_real_df.apply(lambda s:s.lower() if type(s) == str else s)
le_nodes = LabelEncoder().fit(pd.concat([graph_wiki_real_df['Node1'], graph_wiki_real_df['Node2']]))
le_labels = LabelEncoder().fit(graph_wiki_real_df['Label'])
graph_wiki_real_df['Node1'] = le_nodes.transform(graph_wiki_real_df['Node1'])
graph_wiki_real_df['Node2'] = le_nodes.transform(graph_wiki_real_df['Node2'])
graph_wiki_real_df['Label'] = le_labels.transform(graph_wiki_real_df['Label'])

# Split the data
train_wiki_real, eval_wiki_real = train_test_split(graph_wiki_real_df, test_size=0.2)

def build_graph(df):
    G = nx.from_pandas_edgelist(df, 'Node1', 'Node2', edge_attr='Label', create_using=nx.DiGraph())
    data = from_networkx(G)
    # Use the node IDs as node features
    data.x = torch.tensor(list(G.nodes), dtype=torch.float).view(-1, 1)
    # Convert edge attributes to a tensor and store them in data.edge_attr
    edge_attrs = [G.get_edge_data(*e).get('Label', 0) for e in G.edges()]
    data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)
    return data

train_graphs_wiki_real = train_wiki_real.groupby('Claim ID').apply(build_graph)
eval_graphs_wiki_real = eval_wiki_real.groupby('Claim ID').apply(build_graph)


##############################################
import random
import torch.optim as optim
from torch.nn.functional import cosine_similarity

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
# Masking function
def mask_graph(data, mask_rate=0.15):
    # Ensure data.x is defined and is a tensor
    if data.x is None:
        data.x = torch.ones((data.num_nodes, 1))

    # Create a mask for nodes
    node_mask = torch.rand(data.num_nodes) < mask_rate
    data.x[node_mask] = 0

    # Create a mask for edge attributes
    edge_mask = torch.rand(data.edge_index.shape[1]) < mask_rate
    data.edge_attr[edge_mask] = 0
    return data
# Training
optimizer = optim.Adam(list(gcn.parameters()) + list(attention.parameters()), lr=0.001)
for epoch in tqdm(range(100)):
    for claim_id, graph in train_graphs_wiki_real.items():
        optimizer.zero_grad()
        graph1 = mask_graph(graph)
        graph2 = mask_graph(graph)
        emb1 = gcn(graph1)
        emb2 = gcn(graph2)
        attn1 = attention(emb1)
        attn2 = attention(emb2)
        graph_emb1 = torch.mean(attn1 * emb1, dim=0)
        graph_emb2 = torch.mean(attn2 * emb2, dim=0)
        pos_loss = -cosine_similarity(graph_emb1.unsqueeze(0), graph_emb2.unsqueeze(0))

        negative_samples = random.sample(list(train_graphs_wiki_real), 10)
        max_distance = 0
        max_distance_graph_emb = None
        for negative_graph in negative_samples:
            if negative_graph == graph:
                continue
            negative_emb = gcn(mask_graph(negative_graph))
            negative_attn = attention(negative_emb)
            negative_graph_emb = torch.mean(negative_attn * negative_emb, dim=0)
            distance = torch.dist(graph_emb1, negative_graph_emb)
            if distance > max_distance:
                max_distance = distance
                max_distance_graph_emb = negative_graph_emb

        neg_loss = cosine_similarity(graph_emb1.unsqueeze(0), max_distance_graph_emb.unsqueeze(0))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

# Save the model and optimizer states
torch.save({
    'gcn_state_dict': gcn.state_dict(),
    'attention_state_dict': attention.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'trained_variables.pt')

# Evaluation
with torch.no_grad():
    similarities = []
    for claim_id, graph in eval_graphs_wiki_real.items():
        emb1 = gcn(mask_graph(graph))
        emb2 = gcn(mask_graph(graph))
        attn1 = attention(emb1)
        attn2 = attention(emb2)
        graph_emb1 = torch.mean(attn1 * emb1, dim=0)
        graph_emb2 = torch.mean(attn2 * emb2, dim=0)
        similarities.append(cosine_similarity(graph_emb1.unsqueeze(0), graph_emb2.unsqueeze(0)).mean().item())
    print(f'Average similarity: {sum(similarities) / len(similarities)}')