import prepare_data
import pandas as pd
import nltk
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import matplotlib.pyplot as plt
import pyphen
import re
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat', 'tagger'])
dic = pyphen.Pyphen(lang='en')

data_path = "/project/ts_cost_function/TS_LossFunction/code/data/"

#######------------------- Description of this file

# The following functions and classes are used to calculate the different features of a text
# We seperated these processes from the notebooks, because the TextFeatureCreator (wrapper for all features)
# is used in several notebooks

#######-------------------


#######------------------- Helper Functions Paul

def tok_pos_lem_tagging(text,nlp):
    #Return Tokens,POS Tags and lemmatized words of the text
    tok_pos_lem_text = [(token.text,token.tag_,token.lemma_) for token in nlp(text)]

    last_dot = 0
    tok_pos_lem_sents = []
    for ind,(tok,pos,lem) in enumerate(tok_pos_lem_text):
        if lem == "-PRON-":
            tok_pos_lem_text[ind] = (tok,pos,tok)
        if tok == '.':
            tok_pos_lem_sents += [list(zip(*tok_pos_lem_text[last_dot:ind]))]
            last_dot = ind + 1
    
    return list(zip(*[sent for sent in tok_pos_lem_sents if len(sent) == 3]))

def count_sent_word_length(tok_text):
    #mean sentence and word length
    n_sent = len(tok_text)
    sent_len = [len(words) for words in tok_text]
    n_words = sum(sent_len)
    word_len = [len(word) for words in tok_text for word in words]
    
    mean_word_len = np.mean(word_len)
    mean_sent_len = np.mean(sent_len)
    return mean_word_len,mean_sent_len
    
def load_basic_eng(basic_eng_path=data_path+"basic_english.txt"):
    #returns dataframe with the most common english words
    with open(basic_eng_path,"r") as f:
        data = f.read()
        words = data.split(" , ")
        basic_df = pd.DataFrame(data=words,columns=["word"])
        basic_df["index"] = range(0,len(basic_df))
    return basic_df

def load_eng_words(eng_words_path=data_path+"20k_words.txt"):
    #returns dataframe with 20000 english words
    with open(eng_words_path,"r") as f:
        data = f.read()
        words = data.split("\n")
        english_df = pd.DataFrame(data=words,columns=["word"])
        english_df["index"] = range(0,len(english_df))
    return english_df

def calc_basic_eng_ratio(text,basic_df):
    #how many of the used words are basic english words
    text_set = set(np.concatenate(text))
    return len(text_set.intersection(
        basic_df["word"].values))/len(text_set)


def calc_syllables_count(tok_text):
    #calculates the mean syllables rate
    return len(
        np.concatenate(
            [dic.inserted(word).split("-") for sent in tok_text for word in sent]
        )
    )/len(tok_text)


def TTR2(lem_text):
    #2. Version of Type Token Ratio
    word_list = np.concatenate(lem_text)
    return (len(set(word_list)))/(len(word_list))

def pos_token_count(pos_text,tag_symb):
    #Token Count for first letter of Tokens
    return np.mean([len([tag for tag in sent if tag.startswith(tag_symb)]) for sent in pos_text])

#######------------------- Helper Functions Rebekah

def parse(textlist):
    # takes a LIST of strings, each string being one passage/text
    # returns list of dependency tags, one for each string
    deps = []
    for doc in nlp.pipe(textlist, batch_size=50, n_threads=3):
        if doc.is_parsed:
            deps.append(tuple([n.dep_ for n in doc]))
        else:
            deps.append(None)
    return deps

#######------------------- Extract Feature Function Paul

def get_text_features_V1(textlist):
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
    
    nlp = spacy.load('en_core_web_sm')
    
    #Load English Vocabulary
    basic_df = load_basic_eng()
    english_df = load_eng_words()
    non_basic_df = english_df.loc[
        ~english_df["word"].isin(basic_df["word"])]

    text_features = []
    
    for text in textlist:
    
        #tokenize, pos tagging, lemmatize
        #tok_text,pos_text = tok_pos_tagging(text)
        #lem_text = lemmatize_text(tok_text,pos_text)
        tok_text,pos_text,lem_text = tok_pos_lem_tagging(text,nlp)

        #Mean Word Len, Mean Sent Len
        mean_word_len,mean_sent_len = count_sent_word_length(tok_text)

        #Basic English Ratio
        basic_eng_ratio = calc_basic_eng_ratio(lem_text,basic_df)

        #Syllables per Sentence Ratio
        syll_sent_ratio = calc_syllables_count(tok_text)

        #TTR
        ttr_ratio = TTR2(lem_text)

        #POS Token per Sentence
        #Noun, Verb, Adjective, Adverb, Pronoun
        token_count = []
        
        for tag in ["N","VB","J","RB","PRP"]:
                token_count += [pos_token_count(pos_text,tag)]

        #tok_text,pos_text,lem_text
        text_features += [np.concatenate([[mean_word_len,mean_sent_len,basic_eng_ratio,syll_sent_ratio,ttr_ratio],token_count])]
        
    return text_features

#######------------------- Extract Feature Function Rebekah

def get_text_features_V2(textlist):
    # takes list of texts
    # returns feature vectors (one for each text)
    #features = ["Subordination", "Complements", "Coordination", "Apposition", "Passive verbs", "Parataxis", "Auxiliary Verbs", "Negation", "Prepositional Phrases", "Modifiers"]

    deps = parse(textlist)
    features = []
    
    for idx in range(len(textlist)):
        counts = []
        n = deps[idx].count('ROOT')
        # 01: Clauses / Subordination
        counts.append(deps[idx].count('acl') + deps[idx].count('advcl') + deps[idx].count('relcl'))
        # 02: Complements
        counts.append(deps[idx].count('ccomp') + deps[idx].count('xcomp'))
        # 03: Coordination
        counts.append(deps[idx].count('cc'))
        # 04: Apposition
        counts.append(deps[idx].count('appos'))
        # 05: Passive Verbs
        counts.append(deps[idx].count('nsubjpass') + deps[idx].count('csubjpass'))
        # 06: Parataxis
        counts.append(deps[idx].count('parataxis'))
        # 07: Auxiliary Verbs
        counts.append(deps[idx].count('aux') + deps[idx].count('auxpass'))
        # 08: Negation
        counts.append(deps[idx].count('neg'))
        # 09: Prepositional Phrases
        counts.append(deps[idx].count('prep'))
        # 10: Modifiers
        counts.append(deps[idx].count('advmod') + deps[idx].count('amod') + deps[idx].count('nummod') + deps[idx].count('nmod'))
        
        features.append(np.array(counts)/n)
    
    return features

#######------------------- Extract Feature Function Ensemble

class TextFeatureCreator:    
    lsat_texts = None
    re_features_max = None
    pa_features_max = None
    
    def __init__(self,lsat_path):
        
        #load the maximum text: LSAT
        with open(lsat_path, 'r', encoding="utf8") as lsat_file:
            self.lsat_texts = lsat_file.read()
            self.lsat_texts = self.lsat_texts.split("\n\n")
            for text in self.lsat_texts:
                if text[0] == '#':
                    self.lsat_texts.remove(text)
        
        self.feature_max = np.mean(self.get_text_features(self.lsat_texts), axis=0)
        self.feature_names = [
            "Mean word length", 
            "Mean sentence length", 
            "Basic english ratio", 
            "Syllables per sentence", 
            "Type token ratio", 
            "#nouns", 
            "#verbs", 
            "#adjectives", 
            "#adverbs", 
            "#pronouns"]+[
            "Subordination", 
            "Complements", 
            "Coordination", 
            "Apposition", 
            "Passive verbs", 
            "Parataxis", 
            "Auxiliary Verbs", 
            "Negation", 
            "Prepositional Phrases", 
            "Modifiers"]
        
        
    def get_text_features(self,texts):
        replace = re.compile(r"(\r|\n|##)")
        texts = [replace.sub("",text) for text in texts]
    
        features_v1 = get_text_features_V1(texts)
        features_v2 = get_text_features_V2(texts)
        
        return np.concatenate([features_v1,features_v2],axis=1)

    def norm_features(self,features):
        #Normalize features by the maximum value of the LSAT text
        return np.divide(features,self.feature_max)
