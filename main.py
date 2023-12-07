import json
import spacy
import csv
from collections import defaultdict

# Load SpaCy's English NLP model
nlp = spacy.load('en_core_web_sm')

# The files to load
files = ['AI502-Llama-2-13b-chat-hf-Greedy-300703.jsonl', 'AI502-Llama-2-13b-chat-hf-Sampled-300704.jsonl']

# A dictionary to hold the nodes and edges
graph = defaultdict(list)

from tqdm import tqdm

# A dictionary to hold the nodes and edges
graph = defaultdict(lambda: defaultdict(list))

# A dictionary to map each claim to a unique ID
claim_ids = {}
next_claim_id = 1

# Process each file
for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Processing {file}"):
            data = json.loads(line)
            claim = data['claim']

            # Assign a unique ID to each claim
            if claim not in claim_ids:
                claim_ids[claim] = next_claim_id
                next_claim_id += 1
            claim_id = claim_ids[claim]

            # Process each sentence in the file
            for sentence_type in ['wiki', 'response']:
                # Use SpaCy to parse the text and split it into sentences
                doc = nlp(data[sentence_type])
                sentences = list(doc.sents)

                for sentence in sentences:
                    # Extract the entities and relationships
                    for entity in sentence.ents:
                        for token in entity.subtree:
                            # Include more dependency types
                            if token.dep_ in ('attr', 'dobj', 'pobj', 'nsubj', 'prep', 'ccomp', 'xcomp', 'acomp', 'advcl', 'relcl'):
                                subject = [w for w in token.head.lefts if w.dep_ in ('nsubj', 'pobj')]
                                if subject:
                                    subject = subject[0]
                                    # If the subject is 'which' or 'that', use the head of the subject as the node
                                    if subject.text.lower() in ('which', 'that', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
                                                                 'Who', 'Which', 'Whom', 'Whose', 'Where', 'When', 'Why', 'How','He','She','It','They','he','she','it','they'):
                                        subject = subject.head
                                    graph[claim_id][sentence_type].append((subject.text, entity.text, token.head.text))

# Write the nodes and edges to two separate CSV files
for sentence_type in ['wiki', 'response']:
    with open(f'graph_{sentence_type}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Claim ID', 'Claim', 'Source', 'Node', 'Edge', 'Label'])  # Write the header
        for claim, claim_id in claim_ids.items():
            for edge in graph[claim_id][sentence_type]:
                writer.writerow([claim_id, claim, sentence_type, edge[0], edge[1], edge[2]])


from karateclub import Graph2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from networkx import from_pandas_edgelist
import pandas as pd

# Load the knowledge graphs
graph_response_df = pd.read_csv('graph_response.csv')
graph_wiki_df = pd.read_csv('graph_wiki.csv')

# Convert the dataframes into a list of graphs
graphs_response = [from_pandas_edgelist(df, 'Node', 'Label', 'Edge') for _, df in graph_response_df.groupby('Claim ID')]
graphs_wiki = [from_pandas_edgelist(df, 'Node', 'Label', 'Edge') for _, df in graph_wiki_df.groupby('Claim ID')]

# Combine the response and wiki graphs and split them into training, validation, and test sets
#train eval split
graphs_train_response, graphs_eval_response = train_test_split(graphs_response, test_size=0.2, random_state=42)
graphs_train_wiki, graphs_eval_wiki = train_test_split(graphs_wiki, test_size=0.2, random_state=42)

graph_train = graphs_train_response + graphs_train_wiki
graph_eval = graphs_eval_response + graphs_eval_wiki

# Train the Graph2Vec model on the training set
model = Graph2Vec(dimensions=128)
model.fit(graph_train)

# Get the embeddings of the graphs in the training set
embeddings_train = model.get_embedding()

# Get the embeddings of the graphs in the validation and test sets
embeddings_val = model.transform(graph_eval)

# Calculate the cosine similarity between the embeddings for the same claim ID
similarities = cosine_similarity(embeddings_train[:len(graphs_train_response)], embeddings_train[len(graphs_train_response):])
similarities_eval = cosine_similarity(embeddings_train[:len(graphs_eval_response)], embeddings_train[len(graphs_eval_response):])

print('Similarity: %.3f' % similarities.mean())
print('Eval Similarity: %.3f' % similarities_eval.mean())
