import math
import os
import re
import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer
import stop_words


def build(corpus_path):
    H = {}
    total_number_of_documents = 0
    document_lengths = {}
    corpus_files_names = get_corpus_files_names(corpus_path)
    for corpus_files in corpus_files_names:
        xml_documents = get_xml_documents(corpus_path, corpus_files)
        for xml_document in xml_documents:
            total_number_of_documents += 1
            record_num, vector, length = get_document_vector(xml_document)
            insert_document_vector_to_hash_map(H, record_num, vector)
            document_lengths[record_num] = length
    compute_idf(H, total_number_of_documents)
    vector_lengths = compute_vector_length(H)
    return H, vector_lengths, document_lengths


def compute_vector_length(H):
    vector_lengths = {}
    for token in H:
        i = H[token]["idf"]
        for token_occurrence in H[token]["documents"]:
            document = token_occurrence["document"]
            if document not in vector_lengths:
                vector_lengths[document] = 0
            c = token_occurrence["tf"]
            vector_lengths[document] += math.pow(i * c, 2)
    for document in vector_lengths:
        vector_lengths[document] = math.sqrt(vector_lengths[document])
    return vector_lengths


def compute_idf(H, total_number_of_documents):
    for token in H:
        idf = math.log(total_number_of_documents/H[token]["df"])
        H[token]["idf"] = idf


def insert_document_vector_to_hash_map(H, ID, V):
    for token in V:
        if token not in H:
            H[token] = {"df": 0, 'documents': []}
        H[token]["df"] += 1
        H[token]["documents"].append({"document": ID, "tf": V[token]})


def get_document_vector(xml_document):
    v = {}
    length = 0
    record_num_element = xml_document.find("./RECORDNUM")
    if record_num_element is None:
        return
    record_num = record_num_element.text
    for element_name in ["TITLE", "EXTRACT", "ABSTRACT"]:
        element = xml_document.find(f"./{element_name}")
        if element is None:
            continue
        tokens = get_tokens(element.text)
        length += len(tokens)
        add_tokens_to_vector(v, tokens)
    return record_num, v, length


def add_tokens_to_vector(v, tokens):
    for token in tokens:
        if token not in v:
            v[token] = 0
        v[token] += 1


def get_tokens(sentence):
    ps = PorterStemmer()
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(r'[\d]', '', sentence)
    word_tokens = sentence.split()
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words.STOP_WORDS]
    filtered_sentence = [ps.stem(w) for w in filtered_sentence]
    return filtered_sentence


def get_xml_documents(dir_name, file_name):
    file = "{dir}/{file}".format(dir=dir_name, file=file_name)
    tree = ET.parse(file)
    root = tree.getroot()
    xml_documents = root.findall("./RECORD")
    return xml_documents


def get_corpus_files_names(dir_name):
    return os.listdir(path=dir_name)
