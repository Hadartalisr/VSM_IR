import math
import xml.etree.ElementTree as ET
import vsm_ir

queries_file = "other-xmls/cfquery.xml"
corpus_dir = "./cfc-xml"


def find_best_k_and_b():
    """
    This function was written in order to choose the best k & b parameters in bm25 algorithm
    We found that the best NDCG result is given with k = 2, b = 0.9
    """
    # ks = [1.2 + 0.1 * x for x in range(8)]
    # bs = [0 + 0.1 * x for x in range(10)]
    ks = [1.8 + 0.05 * x for x in range(5)]
    bs = [0.8 + 0.05 * x for x in range(5)]
    for k in ks:
        for b in bs:
            sum_ndcg = 0
            queries = get_sorted_queries()
            for query in queries:
                bm25_result = vsm_ir.query("bm25", vsm_ir.inverted_index_file_name, query["text"], k, b, 10, 11)
                ndcg = calc_NDCG(10, query["records"], bm25_result)
                sum_ndcg += ndcg
            print("k", k, "b", b, "ndgc", sum_ndcg / len(queries))


def find_bm25_best_min_score_to_return():
    """
    This function was written in order to choose the best min score of a bm25 result to be returned.
    We found that the best F score result
    (f_score 0.002886002886002885 precision 0.31857773505013853 recall 0.30357809125797464 avg_num_of_results 36.80)
    is given with min_score = 11
    """
    queries = get_sorted_queries()
    k = 2
    b = 0.9
    scores = [7 + x for x in range(20)]
    for score in scores:
        sum_precision = 0
        sum_recall = 0
        sum_f_scores = 0
        num_of_results = 0
        for query in queries:
            bm25_result = vsm_ir.query("bm25", vsm_ir.inverted_index_file_name, query["text"], k, b, 10, score)
            recall, precision = calc_recall_precision(query["records"], bm25_result)
            f_score = calc_f_score(recall, precision)
            num_of_results += len(bm25_result)
            sum_precision += precision
            sum_recall += recall
            sum_f_scores += f_score
        print("min_score", score, "f_score", f_score / len(queries),
              "precision", sum_precision / len(queries), "recall", sum_recall/ len(queries),
              "avg_num_of_results", num_of_results / len(queries))


def find_tfidf_best_min_score_to_return():
    """
    This function was written in order to choose the best min score of a tfidf result to be returned.
    We found that the best F score result
    (f_score 0.002886002886002885 precision 0.3685466774915622 recall 0.2616427961255298 avg_num_of_results 23.0)
    is given with min_score = 0.3
    """
    queries = get_sorted_queries()
    scores = [0 + 0.1*x for x in range(10)]
    for score in scores:
        sum_precision = 0
        sum_recall = 0
        sum_f_scores = 0
        num_of_results = 0
        for query in queries:
            tfidf_result = vsm_ir.query("tfidf", vsm_ir.inverted_index_file_name, query["text"], 0, 0, 10, score)
            recall, precision = calc_recall_precision(query["records"], tfidf_result)
            f_score = calc_f_score(recall, precision)
            num_of_results += len(tfidf_result)
            sum_precision += precision
            sum_recall += recall
            sum_f_scores += f_score
        print("min_score", score, "f_score", f_score / len(queries),
              "precision", sum_precision / len(queries), "recall", sum_recall / len(queries),
              "avg_num_of_results", num_of_results / len(queries))


def calc_f_score(recall, precision):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calc_recall_precision(records, real_results):
    relevant_document_retrieved = 0
    raw_records = []
    for record in records:
        raw_records.append(record["record"])
    for i, real_result in enumerate(real_results):
        if real_result in raw_records:
            relevant_document_retrieved += 1
    recall = relevant_document_retrieved / len(records)
    precision = relevant_document_retrieved / len(real_results)
    return recall, precision


def calc_NDCG(n, records, real_results):
    idcg = calc_idcg(n, records)
    dcg = calc_dcg(n, records, real_results)
    return dcg[n - 1] / idcg[n - 1]


def calc_dcg(n, records, real_results):
    h = {}
    for record in records:
        h[record["record"]] = record["score"]
    res = []
    if real_results[0] in h:
        res.append(h[real_results[0]])
    else:
        res.append(0)
    for i in range(1, n):
        if len(real_results) <= i:
            dcg = res[i - 1]
        else:
            if real_results[i] in h:
                dcg = res[i - 1] + h[real_results[i]] / math.log2(i + 1)
            else:
                dcg = res[i - 1]
        res.append(dcg)
    return res


def calc_idcg(n, records):
    res = [records[0]["score"]]
    for i in range(1, n):
        if len(records) <= i:
            idcg = res[i - 1]
        else:
            idcg = res[i - 1] + records[i]["score"] / math.log2(i + 1)
        res.append(idcg)
    return res


def get_sorted_queries():
    queries = get_queries()
    for query in queries:
        query["records"].sort(key=lambda record: record["score"], reverse=True)
    return queries


def get_queries():
    """
    :return: the example queries formatted as dictionaries
    """
    queries = []
    xml_queries = get_xml_queries()
    for xml_query in xml_queries:
        query = {}
        query["number"] = xml_query.findtext("QueryNumber")
        query["text"] = xml_query.findtext("QueryText")
        query["results"] = int(xml_query.findtext("Results"))
        query["records"] = []
        records = xml_query.find("Records")
        for record in records:
            item = record.text
            score = 0
            for i in range(0, len(record.attrib["score"])):
                score += int(record.attrib["score"][i])
            query["records"].append({"record": int(item), "score": score / 8})
        queries.append(query)
    return queries


def get_xml_queries():
    """
    :return: the example xml queries that were given in the moodle
    """
    tree = ET.parse(queries_file)
    root = tree.getroot()
    xml_queries = root.findall("./QUERY")
    return xml_queries


# find_bm25_best_min_score_to_return()
# find_tfidf_best_min_score_to_return()
# find_best_k_and_b()
