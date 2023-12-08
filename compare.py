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
import pandas as pd

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

print('graph_response_'+args.input+'.csv')

# Load the knowledge graphs
graph_response_df = pd.read_csv('graph_response_'+args.input+'.csv')
graph_wiki_df = pd.read_csv('graph_wiki_'+args.input+'.csv')

if graph_response_df.empty:
    print('No graph response found')
    exit()
else:
    print('Graph response found')