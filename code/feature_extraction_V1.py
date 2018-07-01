import prepare_data
import pandas as pd
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import matplotlib.pyplot as plt
import pyphen
dic = pyphen.Pyphen(lang='en')


#######------------------- Helper Functions


def tok_pos_tagging(text):
    
    tok_text = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]
    tok_pos_text = list(zip(*[list(zip(*nltk.pos_tag(sent))) for sent in tok_text]))
    
    return tok_pos_text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatize_text(tok_text,pos_text):
    lemmatizer = WordNetLemmatizer()
    lem_text = [[lemmatizer.lemmatize(word,get_wordnet_pos(tag)) for word,tag in zip(*[sent,tags])]
                for sent,tags in zip(*[tok_text,pos_text])]
    return lem_text

def count_sent_word_length(tok_text):
    n_sent = len(tok_text)
    sent_len = [len(words) for words in tok_text]
    n_words = sum(sent_len)
    word_len = [len(word) for words in tok_text for word in words]
    
    mean_word_len = np.mean(word_len)
    mean_sent_len = np.mean(sent_len)
    return mean_word_len,mean_sent_len
    
def load_basic_eng(basic_eng_path="data/basic_english.txt"):

    with open(basic_eng_path,"r") as f:
        data = f.read()
        words = data.split(" , ")
        basic_df = pd.DataFrame(data=words,columns=["word"])
        basic_df["index"] = range(0,len(basic_df))
    return basic_df

def load_eng_words(eng_words_path="data/20k_words.txt"):

    with open(eng_words_path,"r") as f:
        data = f.read()
        words = data.split("\n")
        english_df = pd.DataFrame(data=words,columns=["word"])
        english_df["index"] = range(0,len(english_df))
    return english_df

def calc_basic_eng_ratio(text,basic_df):
    text_set = set(np.concatenate(text))
    return len(text_set.intersection(
        basic_df["word"].values))/len(text_set)


def calc_syllables_count(tok_text):
    return len(
        np.concatenate(
            [dic.inserted(word).split("-") for sent in tok_text for word in sent]
        )
    )/len(tok_text)

def TTR(tok_text): 
    words = {} 
    num_words = 0 
    for line in tok_text: 
        for word in line: 
            num_words += 1 
            if word in words: 
                words[word] += 1 
            else: 
                words[word] = 1 
    return len(words) / num_words 

def TTR2(tok_text):
    word_list = np.concatenate(tok_text)
    return (len(set(word_list)))/(len(word_list))

def pos_token_count(pos_text,tag_symb):
    #Token Count for first letter of Tokens
    
    return np.mean([len([tag for tag in sent if tag.startswith(tag_symb)]) for sent in pos_text])



#######------------------- Ensemble Feature Function

def get_text_features_V1(text):
    #Returns the the following features for the given text:
    #Mean Word Length
    #Mean Sentence Length
    #Proportion of Basic English Words in the text
    #Number of Syllables per Sentence
    #TTR Ratio
    #Token Counts per Sentence and Category:
    #   - Nouns
    #   - Verbs
    #   - Adjectives
    #   - Adverb
    #   - Pronoun
    
    
    #tokenize, pos tagging, lemmatize
    tok_text,pos_text = tok_pos_tagging(text)
    lem_text = lemmatize_text(tok_text,pos_text)
    
    #Mean Word Len, Mean Sent Len
    mean_word_len,mean_sent_len = count_sent_word_length(tok_text)
    
    #Load English Vocabulary
    basic_df = load_basic_eng()
    english_df = load_eng_words()
    non_basic_df = english_df.loc[
        ~english_df["word"].isin(basic_df["word"])]
    
    #Basic English Ratio
    basic_eng_ratio = calc_basic_eng_ratio(lem_text,basic_df)
    
    #Syllables per Sentence Ratio
    syll_sent_ratio = calc_syllables_count(tok_text)
    
    #TTR
    ttr_ratio = TTR2(tok_text)
    
    #POS Token per Sentence
    #Noun, Verb, Adjective, Adverb, Pronoun
    token_count = []
    for tag in ["N","VB","J","RB","PRP",","]:
        token_count += [pos_token_count(pos_text,tag)]
    
    #tok_text,pos_text,lem_text
    return np.concatenate([[mean_word_len,mean_sent_len,basic_eng_ratio,syll_sent_ratio,ttr_ratio],token_count])
    