import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections import defaultdict
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from datasets import load_dataset

'''
***************************************** PART 1 ***********************************************
'''

# global variables
# we use defaultdict instead of initializing a dict every time the key does not exist
inverted_index = defaultdict(dict)
documents_word_counts = {}
total_num_documents = 0
doc_lens = defaultdict(lambda: 0.0)
doc_num_of_words = {}
idfs = {}


# calculate number of words for each document
def compute_doc_num_of_words():
    for doc_num, document in documents_word_counts.items():
        doc_num_of_words[doc_num] = sum(document.values())  # save the amount to doc_num_of_words (doc_num : num_words)


# sum of (tf*idf)^2 and sqrt of them
# doc_lens will hold doc_num : doc_length mapping
def compute_vec_length():
    for word, occurrences_list in inverted_index.items():
        idf = idfs[word]  # use the pre-calculate IDF
        for doc_num, tf in occurrences_list.items():
            doc_lens[doc_num] += ((tf[0] / tf[1]) * idf) ** 2  # sum of squares
    for doc_num, doc_len in doc_lens.items():
        doc_lens[doc_num] = math.sqrt(doc_len)  # iterate them again and perform sqrt


# compute IDFs and save them into the idfs map (word to its idf)
def compute_idf():
    for word, occurrences_list in inverted_index.items():
        N_T = len(occurrences_list.keys())
        idfs[word] = math.log2(total_num_documents / N_T)


# actual algorithm to create the inverted index
def create_inverted_index():
    for doc_num, document in documents_word_counts.items():
        max_occ = max(document.values())  # document values are word counts
        for word, word_count in document.items():
            (inverted_index[word])[doc_num] = (word_count, max_occ)  # save word count and max occurrences for tfidf
    compute_idf()
    compute_vec_length()
    compute_doc_num_of_words()


# tokenize, remove stopwords and stem the document text
def tokenize_and_stem_document(document_text, tokenizer, stop_words, stemmer):
    document_tokens = tokenizer.tokenize(document_text)  # tokenize the document text
    filtered_document = [word for word in document_tokens if not word.lower() in stop_words]  # filter out stop words
    return [stemmer.stem(word) for word in filtered_document]  # stem the words


# tokenize, stem and create word count
def get_word_counts_histogram(document_text):
    try:
        stop_words = set(stopwords.words("english"))  # in case we already downloaded the stopwords
    except:
        nltk.download('stopwords')  # in case we didn't download the stopwords
        stop_words = set(stopwords.words("english"))

    stemmer = PorterStemmer()  # initialize a stemmer
    tokenizer = RegexpTokenizer(r'\w+')  # initialize a tokenizer

    stemmed_document = tokenize_and_stem_document(document_text, tokenizer, stop_words, stemmer)  # tokenize and stem
    return dict(Counter(stemmed_document))  # use Counter class to create a word count for the document


# add word count to document (documents_word_counts is a dictionary contains doc_num : word_count_histogram)
def add_to_documents_word_counts(record_num, document_text):
    documents_word_counts[record_num] = get_word_counts_histogram(document_text)


# create word count for every document
def process_document(doc):
    record_num = int(doc['id'])
    document_text = (doc['title'] + ' ') * 4 + doc['text']
    add_to_documents_word_counts(record_num, document_text)


# main function to create the inverted index
def create_index():
    wiki = load_dataset("wikipedia", "20220301.en")['train']
    for doc in wiki:
        process_document(doc)
    global total_num_documents
    total_num_documents = len(documents_word_counts)  # save total number of documents to compute idf
    create_inverted_index()  # create inverted index, doc_lens, doc_num_of_words and IDFs dictionaries

    index = {  # save them all to the file
        'inverted_index': inverted_index,
        'doc_lens': doc_lens,
        'doc_num_of_words': doc_num_of_words,
        'idfs': idfs
    }

    index_file = open('vsm_inverted_index.json', "w")
    json.dump(index, index_file, indent=2)
    index_file.close()


'''
***************************************** PART 2 ***********************************************
'''


# process the query by tokenize, stem and create a word count
def process_query(question, inverted_idx):
    question = get_word_counts_histogram(question)  # get a dictionary of word to word_count
    return {word: word_count for word, word_count in question.items() if
            word in inverted_idx}  # filter only items that "exist" in our world - exist in the inverted index


# calculate question vector length - The length of a document vector is the square-root of sum of the
# squares of the weights of its tokens
def compute_question_length(question, max_word_occ, doc_to_idf):
    question_len = 0.0
    for word, word_count in question.items():
        idf = doc_to_idf[word]
        tf = word_count / max_word_occ
        question_len += ((tf * idf) ** 2)  # sum of squares of the weights
    return math.sqrt(question_len)  # sqrt of that


# rank the documents per tfidf ranking system
def rank_tfidf(question, inverted_idx, doc_lengths, doc_to_idf, results_dict):
    max_word_occ = max(question.values())  # needed to calc tf for every term in the question
    for word, word_count in question.items():  # for each word in the question
        occ_list = inverted_idx[word]  # get the occurrences list from the inverted index
        idf = doc_to_idf[word]  # use the idf mapping saved before
        tf_question = word_count / max_word_occ
        question_word_weight = tf_question * idf  # question word tfidf
        for doc_num, tf_doc in occ_list.items():
            results_dict[doc_num] += (question_word_weight * idf * (tf_doc[0] / tf_doc[1]))

    question_len = compute_question_length(question, max_word_occ, doc_to_idf)  # calculate question vector length (L)

    for doc_num, score in results_dict.items():  # score is S which is the dot-product of D and Q
        doc_len = doc_lengths[str(doc_num)]  # Y in our algorithm
        results_dict[doc_num] = score / (doc_len * question_len)  # S / (L * Y)


# average document length (number of words)
def calc_avg_doc_len(doc_number_of_words, total_num_of_documents):
    return sum(doc_number_of_words.values()) / total_num_of_documents


# calculate BM25's special IDF per the formula
def calc_bm25_idf(N, n_qi):
    return math.log(((N - n_qi + 0.5) / n_qi + 0.5) + 1)  # per the formula


# rank the documents per bm25 ranking system
def rank_bm25(question, inverted_idx, total_num_of_documents, doc_number_of_words, results_dict):
    b = 0.5  # best empirical b and k1 params for bm25
    k1 = 2.2

    avgdl = calc_avg_doc_len(doc_number_of_words, total_num_of_documents)  # average document length (number of words)
    for q_i, word_count in question.items():  # for each word in the question
        occ_list = inverted_idx[q_i]  # get the occurrences list from the inverted index
        n_qi = len(occ_list)
        idf = calc_bm25_idf(total_num_of_documents, n_qi)
        for doc_num, tf in occ_list.items():  # calculate the whole bm25 rank and save it inside results_dict
            f_qi_d = tf[0]  # we need just the number of times this term appears in BM25, we don't need the whole tf
            document_len_norm = doc_number_of_words[doc_num] / avgdl
            results_dict[doc_num] += word_count * (
                    idf * ((f_qi_d * (k1 + 1)) / (f_qi_d + k1 * (1 - b + b * document_len_norm))))


# run the query on the question using the inverted index
def query(ranking, index_path, question):
    f = open(index_path)
    index = json.load(f)  # open the index saved by create_index function

    # unpack the inverted index, doc lengths, idfs and number of words per document
    inverted_idx = index['inverted_index']
    doc_lengths = index['doc_lens']
    doc_number_of_words = index['doc_num_of_words']
    doc_to_idf = index['idfs']

    results = defaultdict(lambda: 0.0)
    question = process_query(question, inverted_idx)  # process the query by tokenize, stem and create a word count
    if ranking == 'tfidf':  # rank based on the input param "ranking"
        rank_tfidf(question, inverted_idx, doc_lengths, doc_to_idf, results)
    else:
        rank_bm25(question, inverted_idx, len(doc_lengths), doc_number_of_words, results)

    results = sorted(results.items(), key=lambda res: res[1], reverse=True)  # sort the results in a decreasing order

    f = open('ranked_query_docs.txt', "w")  # write results to a file
    threshold = 0.075 if ranking == 'tfidf' else 7.9
    for doc_num, doc_score in results:
        if doc_score >= threshold:  # save only results with score bigger then threshold
            f.write(str(doc_num) + "\n")
    f.close()


if __name__ == '__main__':
    create_index()
    query('bm25', './vsm_inverted_index.json', 'Joel Embiid')
