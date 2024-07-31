"""
Python implementation of WAGRank 1.
"""
import os
import numpy as np
from numpy.linalg import norm
from gensim.models.keyedvectors import KeyedVectors
import math
import networkx as nx
import nltk
from configparser import ConfigParser
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates
import nltk.data
from nltk.stem import PorterStemmer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
porter_stemmer = PorterStemmer()


def setup_environment():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print('Completed resource downloads.')


def filter_for_tags(tagged, tags=['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']):
    return [item for item in tagged if item[1] in tags]


gensim_model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=300000)
embedding_dimension = gensim_model.vector_size
total_vocab = gensim_model.index_to_key


def get_vocab(lexicon):
    vocab = {}
    size = len(lexicon)
    for index in range(size):
        vocab[lexicon[index]] = index
    return vocab


def word_num(lexicon):
    vocab = get_vocab(lexicon)
    dic = {}
    for word in lexicon:
        dic[vocab[word]] = word
    return dic


def cosine_similar_matrix(matrix):
    vocab_num = matrix.shape[0]
    features1 = matrix
    features2 = matrix
    norm1 = norm(features1, axis=-1).reshape(vocab_num, 1)
    norm2 = norm(features2, axis=-1).reshape(1, vocab_num)
    end_norm = np.dot(norm1, norm2)
    cos = np.dot(features1, features2.T) / end_norm
    return cos


def embedding_matrix(lexicon):
    new_add_vocab = list(set(lexicon) - set(total_vocab))
    model = gensim_model
    lex_size = len(lexicon)
    vocab = get_vocab(lexicon)
    embed_matrix = np.zeros((lex_size, embedding_dimension))
    for word in lexicon:
        if word in new_add_vocab:
            if porter_stemmer.stem(word) in total_vocab:
                embed_matrix[vocab[word]] = model[porter_stemmer.stem(word)]
            else:
                embed_matrix[vocab[word]] = np.random.random(embedding_dimension)
        else:
            embed_matrix[vocab[word]] = model[word]
    return embed_matrix


def top_similar(lexicon, topn, matrix):
    topn_dic = {}
    vocab = get_vocab(lexicon)
    dic = word_num(lexicon)
    cosine_matrix = cosine_similar_matrix(matrix)
    for word in lexicon:
        sort_index = np.argsort(-cosine_matrix[vocab[word]])
        topn_words = [dic[index] for index in sort_index[:topn + 1]]
        topn_dic[word] = topn_words[1:]
    return topn_dic


def word_embedding(lexicon, matrix):
    embedding_dic = {}
    vocab = get_vocab(lexicon)
    for word in lexicon:
        embedding_dic[word] = matrix[vocab[word]]
    return embedding_dic


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)


def sent_self_attention(sent_matrix):
    inner_matrix = np.matmul(sent_matrix, sent_matrix.T) / math.sqrt(embedding_dimension)
    softmax_matrix = softmax(inner_matrix)
    return softmax_matrix


def attention_score(all_sent, lexicon, topn):
    attention_score_dic = {}
    matrix = embedding_matrix(lexicon)
    embedding_dic = word_embedding(lexicon, matrix)
    topn_dic = top_similar(lexicon, topn, matrix)
    for word in lexicon:
        attention_score_dic[word] = 0.0
    for s in all_sent:
        lex_size = len(s)
        sent_embedding_matrix = np.zeros((lex_size, embedding_dimension))
        for m in range(lex_size):
            sent_embedding_matrix[m] = embedding_dic[s[m]]
        weight_matrix = sent_self_attention(sent_embedding_matrix)
        diag = weight_matrix.diagonal()
        diag_weight = diag / diag.sum()
        if lex_size > 0:
            for j in range(lex_size):
                attention_score_dic[s[j]] += diag_weight[j]
    return topn_dic, attention_score_dic


def create_graph(lexicon, textlist_filtered_write, topn):
    vocab = get_vocab(lexicon)
    vocab_size = len(vocab)
    board = np.zeros((vocab_size, vocab_size), dtype='float')
    two_dics = attention_score(textlist_filtered_write, lexicon, topn)
    topn_dic = two_dics[0]
    attention_score_dic = two_dics[1]
    for word in lexicon:
        topsim = topn_dic[word]
        for w in topsim:
            board[vocab[w]][vocab[word]] = attention_score_dic[w] + attention_score_dic[word]
            board[vocab[word]][vocab[w]] = attention_score_dic[w] + attention_score_dic[word]
    return board


def load_local_corenlp_pos_tagger():
    config_parser = ConfigParser()
    config_parser.read('config.ini')
    host = config_parser.get('STANFORDCORENLPTAGGER', 'host')
    port = config_parser.get('STANFORDCORENLPTAGGER', 'port')
    return PosTaggingCoreNLP(host, port)


ptagger = load_local_corenlp_pos_tagger()


def no_redundancy_candidate(phrase_list, top_num):
    no_redundancy_list = []
    stemmed_no_redundancy_list = []
    for phrase in phrase_list:
        stemmed_phrase = word_stemmed(phrase)
        if stemmed_phrase not in stemmed_no_redundancy_list:
            stemmed_no_redundancy_list.append(stemmed_phrase)
            no_redundancy_list.append(phrase)
    if len(no_redundancy_list) < top_num:
        candidate_list = no_redundancy_list
    else:
        candidate_list = no_redundancy_list[:top_num]
    return candidate_list


def extract_key_phrases(text, topn):
    text = text.replace('      ', '. ').replace('     ', '. ').replace('   ', ' ') \
        .replace('..', '.').replace(',.', ',').replace(':.', ':').replace('?.', ':')
    text = text.replace('Fig.', 'Figure').replace('Fig .', 'Figure') \
        .replace('FIG.', 'Figure').replace('FIG .', 'Figure').replace('et al.', '').replace('e.g.', '')
    text = text.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
    sentences = tokenizer.tokenize(text)
    lower_lines = []
    for l, sentence in enumerate(sentences):
        if sentence[0].islower():
            if sentences[l - 1][-1] == '.':
                sentences[l - 1] = sentences[l - 1][:-1]
            sentences[l - 1] += ' ' + sentence
            lower_lines.append(l)
    final_output = []
    for l, sentence in enumerate(sentences):
        if l not in lower_lines and any(
                c.isalpha() for c in sentence):
            final_output.append(sentence)
    sent_tokens = final_output
    textlist_filtered = []
    textlist_filtered_write = []
    all_candidates = []
    for s in sent_tokens:
        tagged = ptagger.pos_tag_raw_text(s)
        text_obj = InputTextObj(tagged, 'en')
        candidates = extract_candidates(text_obj)
        all_candidates += candidates
        sent_tagged = [x for item in tagged for x in item]
        sent_tagged = filter_for_tags(sent_tagged)
        sent_textlist_filtered = [x[0].lower() for x in sent_tagged]
        textlist_filtered += sent_textlist_filtered
        textlist_filtered_write.append(sent_textlist_filtered)
    word_set_list = list(set(textlist_filtered))
    graph = create_graph(word_set_list, textlist_filtered_write, topn)
    dic = word_num(word_set_list)
    nx_graph = nx.from_numpy_array(graph)
    calculated_page_rank = nx.pagerank(nx_graph, max_iter=600)
    word_score = {}
    for index, score in calculated_page_rank.items():
        word_score[dic[index]] = score
    modified_key_phrases = set(all_candidates)
    phrase_score = {}
    for phrase in list(modified_key_phrases):
        phrase_score[phrase] = 0
        for word in phrase.split():
            if word in word_score:
                phrase_score[phrase] += word_score[word]
            else:
                phrase_score[phrase] = 0
                break
    sorted_phrases = sorted(phrase_score, key=phrase_score.get, reverse=True)
    # final_key_phrases = sorted_phrases[:5]  # Top 5 ranked phrases
    final_key_phrases = sorted_phrases[:10]  # Top 10 ranked phrases
    # final_key_phrases = no_redundancy_candidate(sorted_phrases, 5)  # Top 5 ranked phrases with redundancy filter
    # final_key_phrases = no_redundancy_candidate(sorted_phrases, 10)  # Top 10 ranked phrases with redundancy filter
    return final_key_phrases


def word_stemmed(phrase):
    stemmed_phrase = phrase
    for word in phrase.split():
        stemmed_phrase = stemmed_phrase.replace(word, porter_stemmer.stem(word))
    return stemmed_phrase


def count_number(textlist, textstr, text_key_list, topn):
    TT = 0
    TAP = 0
    TE = 0
    TP = 0
    precision_list = []
    recall_list = []
    F1_list = []
    print(len(textlist))
    for i in range(len(textlist)):
        text_i = textlist[i]
        textstr_i = textstr[i]
        assigned_phrases_i = text_key_list[i]
        extracted_phrases_i = extract_key_phrases(textstr_i, topn)
        assigned_phrases_i = [word_stemmed(w).lower() for w in assigned_phrases_i]
        extracted_phrases_i = [word_stemmed(w).lower() for w in extracted_phrases_i]
        per_TAP = len(assigned_phrases_i)
        per_TE = len(extracted_phrases_i)
        per_TP = 0
        for j in extracted_phrases_i:
            if j in assigned_phrases_i:
                per_TP += 1
        TT += len(text_i)
        TAP += per_TAP
        TE += per_TE
        TP += per_TP
        per_precision = per_TP / per_TE
        per_recall = per_TP / per_TAP
        if per_precision + per_recall != 0:
            per_F1 = (2 * per_precision * per_recall) / (per_precision + per_recall)
        else:
            per_F1 = 0
        precision_list.append(per_precision)
        recall_list.append(per_recall)
        F1_list.append(per_F1)
    macro_precision = sum(precision_list) / len(precision_list)
    macro_recall = sum(recall_list) / len(recall_list)
    macro_F1 = sum(F1_list) / len(F1_list)
    return macro_precision, macro_recall, macro_F1


setup_environment()

if __name__ == '__main__':
    # SemEval2017: path1 = "./Data./SemEval2017./docsutf8", path2 = "./Data./SemEval2017./keys"
    # Inspec: path1 = "./Data./Inspec./docsutf8", path2 = "./Data./Inspec./keys"
    path1 = "./Data./SemEval2017./docsutf8"
    dirs1 = os.listdir(path1)
    path2 = "./Data./SemEval2017./keys"
    dirs2 = os.listdir(path2)
    textlist = []
    textstr = []
    text_key_list = []
    for a in dirs1:
        f1 = open("./Data./SemEval2017./docsutf8./" + a, 'r', encoding='utf-8')
        texti = f1.read()
        textstr.append(str(texti))
        textlist.append(texti.split())
        f1.close()
    for b in dirs2:
        f2 = open("./Data./SemEval2017./keys./" + b, 'r', encoding='utf-8')
        text_key_list_i = []
        for key in f2.readlines():
            text_key_list_i.append(key.strip())
        text_key_list.append(text_key_list_i)
        f2.close()
    all_macro_precision = []
    all_macro_recall = []
    all_macro_F1 = []
    for run in range(10):
        a = count_number(textlist, textstr, text_key_list, 20)  # SemEval2017
        # a = count_number(textlist, textstr, text_key_list, 30)  # Inspec
        all_macro_precision.append(a[0])
        all_macro_recall.append(a[1])
        all_macro_F1.append(a[2])
    print(len(all_macro_recall))
    print('all_macro_precision', all_macro_precision)
    print('all_macro_recall', all_macro_recall)
    print('all_macro_F1', all_macro_F1)
    print('precision', (np.mean(all_macro_precision), np.std(all_macro_precision, ddof=1)))
    print('recall', (np.mean(all_macro_recall), np.std(all_macro_recall, ddof=1)))
    print('F1', (np.mean(all_macro_F1), np.std(all_macro_F1, ddof=1)))
