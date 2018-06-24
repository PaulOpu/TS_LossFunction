# needs console command: python -m spacy download en
import spacy
import nltk
import numpy as np
nlp = spacy.load('en', disable=['ner', 'textcat', 'tagger'])

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

def counts(textlist):
    # takes list of texts
    # returns feature vectors (one for each text)
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
