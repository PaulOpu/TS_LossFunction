import pandas as pd
import ast

def get_aligned_sentences(simple_path,normal_path,twins=True):
    sdata = read_file(simple_path)
    ndata = read_file(normal_path)
    aligned_data = zip(*[sdata,ndata])
    if twins:
        aligned_data = del_twins(aligned_data)
    return aligned_data


def read_file(path):
    with open(path,"r") as f:
        data = f.read()
        data = data.lower().split("\n")
        data = [line.split("\t") for line in data]
        return data


def del_twins(aligned_data):
    return [(ss,ns) for ss,ns in aligned_data if ss[-1] != ns[-1]]


def create_df(aligned_data):
    data = [(st,sn,ss,nt,nn,ns) for (st,sn,ss),(nt,nn,ns) in aligned_data]
    df = pd.DataFrame(data=list(data),columns=["simple_topic","simple_numb","simple_sentence","normal_topic","normal_numb","normal_sentence"])
    return df

def load_df(path,list_cols):
    #list_cols: which columns have lists inside and have to be reparced
    df = pd.read_csv(path,sep="|")
    df[list_cols] = df[list_cols].applymap(
    lambda x: ast.literal_eval(x))
    return df

def save_df(path,df):
    df.to_csv(path,sep="|",index=False)