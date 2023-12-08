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
from torch.nn.functional import mse_loss

# Load the data
graph_wiki_real_df = pd.read_csv('graph_wiki.csv')

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
from sklearn.model_selection import KFold

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
#     def __init__(self, in_features, num_heads=8):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.linear = nn.ModuleList([nn.Linear(in_features, 1) for _ in range(num_heads)])
#         self.leakyrelu = nn.LeakyReLU()

#     def forward(self, x):
#         # Normalize the inputs to the attention network
#         x = F.normalize(x, p=2, dim=1)
#         attns = [F.softmax(self.leakyrelu(linear(x)), dim=0) for linear in self.linear]
#         attn = torch.mean(torch.stack(attns), dim=0)
#         return attn

class Attention(nn.Module):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.in_features = in_features
        # rest of your code

    def forward(self, x):
        # Compute uniform attention scores
        attn = torch.ones_like(x) / x.size(0)
        return attn

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

    # Create a mask for edges
    edge_mask = torch.rand(data.edge_index.shape[1]) < mask_rate

    # Apply the mask to the edge_index by removing the masked edges
    data.edge_index = data.edge_index[:, ~edge_mask]

    return data
# Training
optimizer = optim.Adam(list(gcn.parameters()) + list(attention.parameters()), lr=0.001)

# Convert pandas object to list
train_graphs_wiki_real_list = list(train_graphs_wiki_real.values)

# Cross-validation
kf = KFold(n_splits=5)
stop_flag = 0
for train_index, val_index in kf.split(train_graphs_wiki_real_list):
    train_graphs = [train_graphs_wiki_real_list[i] for i in train_index]
    val_graphs = [train_graphs_wiki_real_list[i] for i in val_index]

    # Early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in tqdm(range(20)):
        for graph in train_graphs:
            optimizer.zero_grad()
            graph1 = mask_graph(graph)
            graph2 = mask_graph(graph)
            emb1 = gcn(graph1)
            emb2 = gcn(graph2)
            attn1 = attention(emb1)
            attn2 = attention(emb2)
            graph_emb1 = torch.sum(attn1 * emb1, dim=0)
            graph_emb2 = torch.sum(attn2 * emb2, dim=0)
            pos_loss = mse_loss(graph_emb1.unsqueeze(0), graph_emb2.unsqueeze(0))

            negative_samples = random.sample(train_graphs, 3)
            max_distance = 0
            max_distance_graph_emb = None
            for negative_graph in negative_samples:
                if negative_graph == graph:
                    continue
                negative_emb = gcn(mask_graph(negative_graph))
                negative_attn = attention(negative_emb)
                negative_graph_emb = torch.sum(negative_attn * negative_emb, dim=0)
                distance = torch.dist(graph_emb1, negative_graph_emb)
                if distance > max_distance:
                    max_distance = distance
                    max_distance_graph_emb = negative_graph_emb

            neg_loss = cosine_similarity(graph_emb1.unsqueeze(0), max_distance_graph_emb.unsqueeze(0))+1

            #Calculate the loss and backpropagate
            loss = torch.clamp(pos_loss + neg_loss, min=0)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} loss: {loss.item()} poss_loss: {pos_loss.item()} neg_loss: {neg_loss.item()}')

        # Validation
        val_loss = 0
        for graph in val_graphs:
            emb1 = gcn(mask_graph(graph))
            emb2 = gcn(mask_graph(graph))
            attn1 = attention(emb1)
            attn2 = attention(emb2)
            graph_emb1 = torch.sum(attn1 * emb1, dim=0)
            graph_emb2 = torch.sum(attn2 * emb2, dim=0)
            pos_loss = mse_loss(graph_emb1.unsqueeze(0), graph_emb2.unsqueeze(0))

            negative_samples = random.sample(val_graphs, 3)
            max_distance = 0
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
                else:
                    max_distance_graph_emb = torch.zeros_like(graph_emb1)

            neg_loss = cosine_similarity(graph_emb1.unsqueeze(0), max_distance_graph_emb.unsqueeze(0)) +1
            loss = torch.clamp(pos_loss + neg_loss, min=0)
            val_loss += loss.item()
        val_loss /= len(val_graphs)

        print(f'Epoch {epoch} loss: {loss.item()}, val_loss: {val_loss}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print('patience_counter', patience_counter)
        else:
            patience_counter += 1
            print('patience_counter', patience_counter)
            if patience_counter >= patience:
                print('Early stopping')
                stop_flag = 1
                break
    if stop_flag == 1:
        break

# Save the model and optimizer states
torch.save({
    'gcn_state_dict': gcn.state_dict(),
    'attention_state_dict': attention.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'trained_variables.pt')

# Evaluation
with torch.no_grad():
    similarities = []
    count = 0
    for claim_id, graph in eval_graphs_wiki_real.items():
        emb1 = gcn(mask_graph(graph))
        emb2 = gcn(mask_graph(graph))
        attn1 = attention(emb1)
        attn2 = attention(emb2)
        graph_emb1 = torch.sum(attn1 * emb1, dim=0)
        graph_emb2 = torch.sum(attn2 * emb2, dim=0)

        emb3 = gcn(graph)
        attn3 = attention(emb3)
        graph_emb3 = torch.sum(attn3 * emb3, dim=0)

        negative_samples = random.sample(list(eval_graphs_wiki_real), 3)
        max_distance = 0
        max_distance_graph_emb = None
        for negative_graph in negative_samples:
            if negative_graph == graph:
                continue
            negative_emb = gcn(negative_graph)
            negative_attn = attention(negative_emb)
            negative_graph_emb = torch.sum(negative_attn * negative_emb, dim=0)
            distance = torch.dist(graph_emb1, negative_graph_emb)
            if distance > max_distance:
                max_distance = distance
                max_distance_graph_emb = negative_graph_emb
            else:
                max_distance_graph_emb = torch.zeros_like(graph_emb1)
        #print('distance:', distance)
        
        simi = cosine_similarity(graph_emb1.unsqueeze(0), graph_emb2.unsqueeze(0)) + 1
        semi = cosine_similarity(graph_emb3.unsqueeze(0), max_distance_graph_emb.unsqueeze(0)) + 1

        if simi > semi:
            count += 1
        
        print('simi:', simi, 'semi:', semi)

        similarities.append(cosine_similarity(graph_emb1.unsqueeze(0), graph_emb2.unsqueeze(0)).mean().item())
    print(f'Average similarity: {sum(similarities) / len(similarities)}')
    print('acc:', count/len(similarities))