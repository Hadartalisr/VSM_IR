import math
import xml.etree.ElementTree as ET

import inverted_index
import vsm_ir

queries_file = "other-xmls/cfquery.xml"
corpus_dir = "./cfc-xml"


def test():
    ks = [1.2 + 0.1 * x for x in range(9)]
    bs = [0.5 + 0.1 * x for x in range(6)]
    for k in ks:
        for b in bs:
            # results = []
            sum_recall = 0
            sum_precision = 0
            sum_ndcg = 0
            queries = get_sorted_queries()
            for query in queries:
                tfidf_result = vsm_ir.query("tfidf", vsm_ir.inverted_index_file_name, query["text"], k , b)
                recall, precision = calc_recall_precision(query["records"], tfidf_result)
                # print(recall, precision)
                # print(calc_NDCG(10, query["records"], tfidf_result))
                bm25_result = vsm_ir.query("bm25", vsm_ir.inverted_index_file_name, query["text"], k ,b)
                recall, precision = calc_recall_precision(query["records"], bm25_result)
                ndcg = calc_NDCG(10, query["records"], bm25_result)
                sum_ndcg += ndcg
                sum_precision += precision
                sum_recall += recall
            print("k", k, "b", b, "recall ", sum_recall/len(queries), "precision", sum_precision/len(queries), "ndgc", sum_ndcg/len(queries))


def calc_recall_precision(records, real_results):
    relevant_document_retrieved = 0
    raw_records = []
    for record in records:
        raw_records.append(record["record"])
    for real_result in real_results:
        if real_result in raw_records:
            relevant_document_retrieved += 1
    recall = relevant_document_retrieved / len(records)
    precision = relevant_document_retrieved / len(real_results)
    return recall, precision


def calc_NDCG(n, records, real_results):
    idcg = calc_idcg(n, records)
    dcg = calc_dcg(n, records, real_results)
    return dcg[n-1]/idcg[n-1]


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
            dcg = res[i-1]
        else:
            if real_results[i] in h:
                dcg = res[i-1] + h[real_results[i]]/math.log2(i+1)
            else:
                dcg = res[i-1]
        res.append(dcg)
    return res


def calc_idcg(n, records):
    res = [records[0]["score"]]
    for i in range(1, n):
        if len(records) <= i:
            idcg = res[i-1]
        else:
            idcg = res[i-1] + records[i]["score"]/math.log2(i+1)
        res.append(idcg)
    return res


def get_sorted_queries():
    queries = get_queries()
    for query in queries:
        query["records"].sort(key=lambda record: record["score"], reverse=True)
    return queries


def get_queries():
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
            query["records"].append({"record": int(item), "score": score/8})
        queries.append(query)
    return queries


def get_xml_queries():
    tree = ET.parse(queries_file)
    root = tree.getroot()
    xml_queries = root.findall("./QUERY")
    return xml_queries

test()