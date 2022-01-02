import os
import re
from tqdm import tqdm
from collections import defaultdict


class SBWC(object):
    def __init__(self):
        self.path = os.path.join(os.getcwd(),"../../../other_datasets/spanish_billion_words")
        self.filenames = sorted(os.listdir(self.path))
        self.pattern = re.compile(r"[^\s]+")

    def __iter__(self):
        filenames = self.filenames
        pattern = self.pattern
        path = self.path
        for filename in tqdm(filenames):
            with open(os.path.join(path,filename),"r") as f:
                for line in f.readlines():
                    line = pattern.findall(line.lower())
                    yield line


class Melisa(object):

    accents = [
        ('[óòöøôõ]','ó'), ('[áàäåâã]','á'), ('[íìïî]','í'), ('[éèëê]','é'), ('[úùû]','ú'), ('[ç¢]','c'), 
        ('[ÓÒÖØÔÕ]','Ó'), ('[ÁÀÄÅÂÃ]','Á'), ('[ÍÌÏÎ]','Í'), ('[ÉÈËÊ]','É'), ('[ÚÙÛ]','Ù'), ('Ç','C'),
        ('[ý¥]','y'), ('š','s'), ('ß','b'), ('\x08','')
    ]

    def __init__(self):
        self.path = os.path.join(os.getcwd(),"../../../other_datasets/melisa")
        self.filenames = sorted(os.listdir(self.path))
        self.pattern = re.compile(r"(\w+|[\.,!\(\)\"\-:\?/%;¡\$'¿\\]|\d+)")

    def __iter__(self):
        filenames = self.filenames
        pattern = self.pattern
        path = self.path
        accents = self.accents
        for filename in tqdm(filenames):
            with open(os.path.join(path,filename),"r") as f:
                for line in f.readlines():
                    for rep, rep_with in accents:
                        line = re.sub(rep,rep_with,line)
                    line = pattern.findall(line.lower())
                    yield line
        

class MelisaSBWC(object):

    accents = [
        ('[óòöøôõ]','ó'), ('[áàäåâã]','á'), ('[íìïî]','í'), ('[éèëê]','é'), ('[úùû]','ú'), ('[ç¢]','c'), 
        ('[ÓÒÖØÔÕ]','Ó'), ('[ÁÀÄÅÂÃ]','Á'), ('[ÍÌÏÎ]','Í'), ('[ÉÈËÊ]','É'), ('[ÚÙÛ]','Ù'), ('Ç','C'),
        ('[ý¥]','y'), ('š','s'), ('ß','b'), ('\x08','')
    ]

    def __init__(self):
        self.melisa_path = os.path.join(os.getcwd(),"../../../other_datasets/melisa")
        self.sbwc_path = os.path.join(os.getcwd(),"../../../other_datasets/spanish_billion_words")
        self.melisa_filenames = sorted(os.listdir(self.melisa_path))
        self.sbwc_filenames = sorted(os.listdir(self.sbwc_path))
        self.melisa_pattern = re.compile(r"(\w+|[\.,!\(\)\"\-:\?/%;¡\$'¿\\]|\d+)")
        self.sbwc_pattern = re.compile(r"[^\s]+")

    def __iter__(self):

        filenames, pattern, path = self.melisa_filenames,self.melisa_pattern,self.melisa_path
        accents = self.accents
        for filename in tqdm(filenames):
            with open(os.path.join(path,filename),"r") as f:
                for line in f.readlines():
                    for rep, rep_with in accents:
                        line = re.sub(rep,rep_with,line)
                    line = pattern.findall(line.lower())
                    yield line

        filenames, pattern, path = self.sbwc_filenames,self.sbwc_pattern,self.sbwc_path      
        for filename in tqdm(filenames):
            with open(os.path.join(path,filename),"r") as f:
                for line in f.readlines():
                    line = pattern.findall(line.lower())
                    yield line
        
        
def inspect_SBWC():
    corpus = SBWC()
    typecounts = defaultdict(lambda: 0)
    sentencecounts = 0
    for sentence in corpus:
        for word in sentence:
            typecounts[word] += 1
        sentencecounts += 1
    print("Cantidad total de tokens:",sum(typecounts.values()))
    print("Cantidad total de tipos:",len(typecounts))
    print("Cantidad total de oraciones:",sentencecounts)
    print()
    freq_cutoff = 5
    frequent_types_dict = {tk: counts for tk, counts in typecounts.items() if counts >= freq_cutoff}
    most_frequent_types = sorted(frequent_types_dict.keys(),key=lambda key: frequent_types_dict[key],reverse=True)[:273]
    counts_of_most_frequent_types = sum([frequent_types_dict[tk] for tk in most_frequent_types])
    print("Suma de las frecuencia de las 273 palabras más comunes:",counts_of_most_frequent_types)
    print("Cantidad total de tokens con frecuencia mayor a {}:".format(freq_cutoff),sum(frequent_types_dict.values()))
    print("Cantidad total de tipos con frecuencia mayor a {}:".format(freq_cutoff),len(frequent_types_dict))
    print()
    




def main():
    pass



if __name__ == "__main__":
    # main()
    inspect_SBWC()