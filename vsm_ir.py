import json
import math
import sys
import inverted_index
import xml.etree.ElementTree as ET

inverted_index_file_name = "vsm_inverted_index.json"
ranked_query_file_name = "ranked_query_docs.txt"

def main():
    args = sys.argv
    if len(args) == 1:
        print("ERROR - len(args) == 0")
        return
    cmd = args[1]
    if cmd == 'create_index':
        if len(args) < 3:
            print("ERROR - can't create index - corpus path is empty.")
            return
        create_index(args[2])
    elif cmd == 'query':
        if len(args) < 5:
            print("ERROR - can't create index - corpus path is empty.")
            return
        if args[2] not in ["bm25", "tfidf"]:
            print("ERROR - unknown ranking")
            return
        k = 1.5
        b = 0.75
        return query(args[2], args[3], args[4], k, b)
    else:
        print("ERROR - invalid command.")
        return


def create_index(corpus_path):
    H, vector_lengths, document_lengths = inverted_index.build(corpus_path)
    Obj = {"inverted_index": H, "vector_lengths": vector_lengths, "document_lengths": document_lengths}
    with open(inverted_index_file_name, 'w') as f:
        json.dump(Obj, f)


def load_inverted_index(inverted_index_file):
    with open(inverted_index_file, 'r') as f:
        data = json.load(f)
    return data["inverted_index"], data["vector_lengths"], data["document_lengths"]


def query(ranking, index_path, question, k, b):
    H, vector_lengths, document_lengths = load_inverted_index(index_path)
    if ranking == "tfidf":
        results = get_query_results_tfidf(H, vector_lengths, question)
        return save_results(results)
    elif ranking == "bm25":
        results = get_query_results_bm25(H, document_lengths, question, k, b)
        return save_results(results)


def save_results(results):
    with open(ranked_query_file_name,"w"):
        1 == 1
    res = []
    with open(ranked_query_file_name,"a") as f:
        for result in results:
            res.append(int(result))
            f.write(f"{int(result)}\n")
    return res


def get_query_results_bm25(H, document_lengths, question, k, b):
    Q = {}
    query_tokens = inverted_index.get_tokens(question)
    inverted_index.add_tokens_to_vector(Q, query_tokens)
    R = {}
    N = len(document_lengths)
    avg_length = sum(document_lengths.values()) / len(document_lengths)
    for T in Q:
        if T not in H:
            continue
        n_T = H[T]["df"]
        idf_T = math.log2(1 + (N - n_T + 0.5)/(n_T+0.5))
        for O in H[T]["documents"]:
            id = O["document"]
            if id not in R:
                R[id] = 0
            f_T_D = O["tf"]
            R[id] += idf_T * (f_T_D * (k+1)) / (f_T_D + k* (1 - b + b * (document_lengths[id]/ avg_length)))
    R = dict(sorted(R.items(), key=lambda item: item[1], reverse=True))
    return R


def get_query_results_tfidf(H, vector_lengths, question):
    Q = {} #1
    query_tokens = inverted_index.get_tokens(question)
    inverted_index.add_tokens_to_vector(Q, query_tokens)
    R = {} #2
    for T in Q: #3
        if T not in H:
            continue
        I = H[T]["idf"]
        K = Q[T]
        W = K * I #5
        for O in H[T]["documents"]:
            id = O["document"]
            if id not in R:
                R[id] = 0
            C = O["tf"]
            R[id] += W*I*C #11
    L = get_query_vector_length(Q)
    for document in R:
        S = R[document]
        Y = vector_lengths[document]
        R[document] = S / (L * Y)
    R = dict(sorted(R.items(), key=lambda item: item[1], reverse=True))
    return R


def get_query_vector_length(query):
    s = 0
    for token in query:
        s += math.pow(query[token], 2)
    return math.sqrt(s)


if __name__ == "__main__":
    main()