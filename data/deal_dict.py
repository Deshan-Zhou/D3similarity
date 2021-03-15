#对序列型符号进行分词，我们采用是n-gram的方法，将n个字符作为一个分词单位，用错开一个字符的方法构造出n个子序列
#而某个序列的高维向量表示就是这个子序列的对应向量的平均值。
#对于fasta序列，3个氨基酸可能有特殊的含义，所以一般使用3-gram
#对于smiles序列，这里将单个字符作为一个词
#在这里我们构造smiles序列和fasta序列的词汇表

import pandas as pd
import pickle

smile_gram = 1
fasta_gram = 3
smi_dict_set = set()
fas_dict_set = set()
one_fas_dict_set = set()

smi_max_len = 0
fas_max_len = 0

smile = pd.read_csv("drug_smiles.csv",header=None,index_col=None)
fasta = pd.read_csv("protein_fasta.csv",header=None,index_col=None)

for item in smile.values:
    if len(item[1]) < smile_gram:
        while len(item[1]) < smile_gram:
            item[1] = item[1] + "_"
        smi_dict_set.add(item[1])
    else:
        for i in range(len(item[1]) - smile_gram + 1):
            smi_dict_set.add(item[1][i:i+smile_gram])

    if len(item[1]) > smi_max_len:
        smi_max_len = len(item[1])

for item in fasta.values:
    if len(item[1]) < fasta_gram:
        while len(item[1]) < smile_gram:
            item[1] = item[1] + "_"
        fas_dict_set.add(item[1])
    else:
        for i in range(len(item[1]) - fasta_gram +1):
            fas_dict_set.add(item[1][i:i+fasta_gram])

    if len(item[1]) > fas_max_len:
        fas_max_len = len(item[1])

for item in fas_dict_set:
    one_fas_dict_set.add(item[0])
    one_fas_dict_set.add(item[1])
    one_fas_dict_set.add(item[2])

print(len(smi_dict_set))
print(smi_dict_set)
print(len(fas_dict_set))
print(fas_dict_set)
print(len(one_fas_dict_set))
print(one_fas_dict_set)
print("smiles最大序列长度：",smi_max_len)
print("fasta最大序列长度：",fas_max_len)

smi_dict = {}
fas_dict = {}
count = 1
for item in smi_dict_set:
    smi_dict[item] = count
    count = count + 1
count = 1
for item in fas_dict_set:
    fas_dict[item] = count
    count = count + 1
print(smi_dict)
print(fas_dict)
with open("smi_dict.pickle","wb") as f:
    pickle.dump(smi_dict,f)
with open("fas_dict.pickle","wb") as f:
    pickle.dump(fas_dict,f)
"""
读取smi_dict的方式
with open("smi_dict.pickle","rb") as f:
    test1 = pickle.load(f)
with open("fas_dict.pickle","rb") as f:
    test2 = pickle.load(f)
print(test1)
print(test2)
"""