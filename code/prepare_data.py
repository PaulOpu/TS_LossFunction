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
        data = data.split("\n")
        data = [line.split("\t") for line in data]
        return data


def del_twins(aligned_data):
    return [ss for ss,ns in aligned_data if ss[-1] != ns[-1]]