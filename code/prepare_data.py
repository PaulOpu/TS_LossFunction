def get_aligned_sentences(simple_path,normal_path):
    sdata = read_file(simple_path)
    ndata = read_file(normal_path)
    return zip(*[sdata,ndata])


def read_file(path):
    with open(path,"r") as f:
        data = f.read()
        data = data.split("\n")
        data = [line.split("\t") for line in data]
        return data