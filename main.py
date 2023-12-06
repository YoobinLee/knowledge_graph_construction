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
                # Use SpaCy to parse the sentence
                doc = nlp(data[sentence_type])

                # Extract the entities and relationships
                for entity in doc.ents:
                    for token in entity.subtree:
                        if token.dep_ in ('attr', 'dobj'):
                            subject = [w for w in token.head.lefts if w.dep_ == 'nsubj']
                            if subject:
                                subject = subject[0]
                                graph[claim_id][sentence_type].append((subject.text, entity.text, token.head.text))

# Write the nodes and edges to two separate CSV files
for sentence_type in ['wiki', 'response']:
    with open(f'graph_{sentence_type}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Claim ID', 'Claim', 'Source', 'Node', 'Edge', 'Label'])  # Write the header
        for claim, claim_id in claim_ids.items():
            for edge in graph[claim_id][sentence_type]:
                writer.writerow([claim_id, claim, sentence_type, edge[0], edge[1], edge[2]])