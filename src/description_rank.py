import json
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List
from collections import defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize



def read_descriptions(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_query(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    query_idx = []
    for line in lines:
        parts = line.split(' ', 1)
        if len(parts) == 2:
            query_idx.append(int(parts[0]))
            data.append(parts[1].strip())  
    return query_idx, data

def read_ground_truth(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    ground_truth = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:
            query_id, _, table_id, relevance = parts
            relevance = int(relevance)
            if query_id not in ground_truth:
                ground_truth[query_id] = {}
            ground_truth[query_id][table_id] = relevance
    return ground_truth

def extract_text(table):
    return re.sub(r'[^\w\s]', '', table)

def rank_tables(query: str, corpus, keys) -> List[int]:
    """
    Ranks tables based on the BM25 algorithm given a query and a JSON string of tables.
    
    :param query: The search query as a string.
    :param tables_json: JSON string representing the tables.
    :return: List of indices representing the tables in descending order of relevance.
    """
    # Extract text from each table

    # Tokenize the corpus and query
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    tokenized_query = query.split(" ")

    # Create a BM25 object and get scores
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Rank the tables based on scores
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    ranked_tables = [keys[i] for i in ranked]
    print(ranked_tables[:10])

    return ranked_tables
        

# Function to calculate DCG at a given cutoff
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

# Function to calculate NDCG at a given cutoff
def ndcg_at_k(r, k, ground_truth):
    dcg_max = dcg_at_k(sorted(ground_truth, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

# Function to calculate Average Precision (AP)
def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)

# Function to calculate Precision at K
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

# def average_precision(r):
#     n_relevant = sum((rel > 0) for rel in r)
#     if n_relevant == 0:
#         return 0.0
#     cumulative_precision = 0.0
#     num_hits = 0.0
#     for i, rel in enumerate(r):
#         if rel > 0:
#             num_hits += 1
#             cumulative_precision += num_hits / (i + 1)
#     return cumulative_precision / n_relevant


# Calculating NDCG and MAP
def calculate_metrics(query_rankings, ground_truth, cutoffs=[5, 10, 15, 20]):
    ndcg_scores = defaultdict(list)
    average_precisions = []

    for query, ranked_tables in query_rankings.items():
        ground_truth_relevance = [ground_truth[query].get(table_id, 0) for table_id in ranked_tables]

        for k in cutoffs:
            ndcg_score = ndcg_at_k(ground_truth_relevance, k, list(ground_truth[query].values()))
            ndcg_scores[k].append(ndcg_score)

        ap_score = average_precision(ground_truth_relevance)
        average_precisions.append(ap_score)

    # Calculate mean scores
    mean_ndcg_scores = {k: np.mean(v) for k, v in ndcg_scores.items()}
    mean_ap = np.mean(average_precisions)

    return mean_ndcg_scores, mean_ap


def query(description_path, query_path, qtrels_path):
    tables = read_descriptions(description_path)
    query_idx, queries = read_query(query_path)
    ground_truth = read_ground_truth(qtrels_path)
    
    corpus = [extract_text(value) for key, value in tables.items()]
    query_rankings = {}
    for idx in range(len(queries)):
        ranked_tables = rank_tables(queries[idx], corpus, list(tables.keys()))
        print(query_idx[idx])
        query_rankings[str(query_idx[idx])] = ranked_tables
    mean_ndcg_scores, mean_ap = calculate_metrics(query_rankings, ground_truth)
    return mean_ndcg_scores, mean_ap


def main(args=None):
    description_path = './wikitables/descriptions.json'
    query_path = './wikitables/queries.txt'
    qtrels_path = './wikitables/qtrels.txt'
    mean_ndcg_scores, mean_ap = query(description_path, query_path, qtrels_path)
    print(mean_ndcg_scores)
    print(mean_ap)
    

if __name__ == '__main__':
    main()