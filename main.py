import json
import spacy
from spacy import displacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import networkx as nx
import csv
from collections import defaultdict
import argparse
import en_core_web_sm

import re
import bs4
import requests


##input parser##
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "input",
    metavar="INPUT",
    type=str,
    default="AI502-Llama-2-13b-chat-hf-Greedy-300703.jsonl",
    help="File name of input json file",
)
args = parser.parse_args()


##generate knowledge graph##
# Load SpaCy's English NLP model
nlp = en_core_web_sm.load()

from tqdm import tqdm

# nodes, edge, label
graph = defaultdict(lambda: defaultdict(list))

# unique claim id
claim_ids = {}
next_claim_id = 1


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""

  #############################################################
  
    for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
            # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " "+ tok.text
      
      # check: token is a modifier or not
        if tok.dep_.endswith("mod") == True:
            modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
            if prv_tok_dep == "compound":
                modifier = prv_tok_text + " "+ tok.text
      
      ## chunk 3
        if tok.dep_.find("subj") == True:
            ent1 = modifier +" "+ prefix + " "+ tok.text
            prefix = ""
            modifier = ""
            prv_tok_dep = ""
            prv_tok_text = ""      

      ## chunk 4
        if tok.dep_.find("obj") == True:
            ent2 = modifier +" "+ prefix +" "+ tok.text
        
        ## chunk 5  
        # update variables
        prv_tok_dep = tok.dep_
        prv_tok_text = tok.text
  #############################################################

    return [ent1.strip(), ent2.strip()]

def get_relation(sent):

    doc = nlp(sent)

    # Matcher class object 
    matcher = Matcher(nlp.vocab)

    #define the pattern 
    pattern = [{'DEP':'ROOT'}, 
                {'DEP':'prep','OP':"?"}, 
                {'DEP':'agent','OP':"?"},  
                {'POS':'ADJ','OP':"?"}] 

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    
    k = len(matches) - 1

    for k in range(len(matches)):
        if k < len(matches):
            span = doc[matches[k][1]:matches[k][2]]
            return span.text
        else:
            print(f"No match found at index {k}")
            return None

with open(args.input, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines, desc=f"Processing {args.input}"):
        #read data
        data = json.loads(line)
        claim = data['claim']

        # assign unique id
        if claim not in claim_ids:
            claim_ids[claim] = next_claim_id
            next_claim_id += 1
        claim_id = claim_ids[claim]

        # process text
        for sentence_type in ['wiki', 'response']:
            # Use SpaCy to parse the entire text
            text = data[sentence_type].replace('\"', '')  # Remove all \" symbols
            text = data[sentence_type].replace('-RSB-"', '')  # Remove all \" symbols
            text = data[sentence_type].replace('-PRB-"', '')  # Remove all \" symbols
            text = data[sentence_type].replace('-RRB-"', '')  # Remove all \" symbols
            text = data[sentence_type].replace('-LSB-"', '')  # Remove all \" symbols
            doc = nlp(text)

            for sent in doc.sents:  # iterate over sentences
                entities1, entities2 = get_entities(sent.text)
                if entities1 == '' or entities2 == '':
                    continue
                relation = get_relation(sent.text)
                graph[claim_id][sentence_type].append((entities1, entities2, relation))


# Write the nodes and edges to two separate CSV files
for sentence_type in ['wiki', 'response']:
    with open(f'graph_{sentence_type}_{args.input}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Claim ID', 'Claim', 'Source', 'Node1', 'Node2', 'Label'])  # Write the header
        for claim, claim_id in claim_ids.items():
            for edge in graph[claim_id][sentence_type]:
                writer.writerow([claim_id, claim, sentence_type, edge[0], edge[1], edge[2]])

