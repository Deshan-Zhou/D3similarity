import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split,StratifiedKFold
import scipy.io as scio

from config import Config

class HetrDataset():
    def __init__(self):
        config = Config()
        self.repeat_nums = config.repeat_nums
        self.fold_nums = config.fold_nums
        self.neg_samp_ratio = config.neg_samp_ratio

        self.dg_smi_path = config.dg_smiles_path
        self.pt_fas_path = config.pt_fasta_path
        self.smi_dict_path = config.smi_dict_path
        self.fas_dict_path = config.fas_dict_path
        self.smi_ngram = config.smi_n_gram
        self.fas_ngram = config.fas_n_gram
        self.smi_max_len = config.smiles_max_len
        self.fas_max_len = config.fasta_max_len

        self.dg_pt_path = config.dg_pt_path

        self.read_data()
        self.pre_process()

    def read_data(self):
        #这是序列数据，还没转变为数值特征
        self.drug_smi = pd.read_csv(self.dg_smi_path,header=None,index_col=None).values
        self.protein_fas = pd.read_csv(self.pt_fas_path,header=None,index_col=None).values
        #导入字符与数值之间映射的字典
        with open(self.smi_dict_path, "rb") as f:
            self.smi_dict = pickle.load(f)
        with open(self.fas_dict_path, "rb") as f:
            self.fas_dict = pickle.load(f)

        self.dg_pt = pd.read_csv(self.dg_pt_path, header=0, index_col=0).values


    def pre_process(self):
        """
        :return:all_data_set:list    repeat_nums*fold_nums*3
        """
        self.all_data_set = []      #[]:repeat_nums*fold_nums*3
        whole_positive_index = []
        whole_negetive_index = []
        for i in range(self.dg_pt.shape[0]):
            for j in range(self.dg_pt.shape[1]):
                if int(self.dg_pt[i, j]) == 1:
                    whole_positive_index.append([i, j])
                elif int(self.dg_pt[i, j]) == 0:
                    whole_negetive_index.append([i, j])

        for x in range(self.repeat_nums):

            #对负2例进行self.neg_samp_ratio倍正例的下采样
            negative_sample_index = np.random.choice(np.arange(len(whole_negetive_index)),
                                                     size=self.neg_samp_ratio * len(whole_positive_index),replace=False)
            data_set = np.zeros((self.neg_samp_ratio*len(whole_positive_index) + len(negative_sample_index),3), dtype=int)

            count = 0
            for item in whole_positive_index:
                #对正例进行self.neg_samp_ratio倍过采样
                for i in range(self.neg_samp_ratio):
                    data_set[count][0] = item[0]
                    data_set[count][1] = item[1]
                    data_set[count][2] = 1
                    count = count + 1
            for i in negative_sample_index:
                data_set[count][0] = whole_negetive_index[i][0]
                data_set[count][1] = whole_negetive_index[i][1]
                data_set[count][2] = 0
                count = count + 1

            all_fold_dataset = []
            rs = np.random.randint(0,1000,1)[0]
            kf = StratifiedKFold(n_splits=self.fold_nums, shuffle=True, random_state=rs)
            for train_index, test_index in kf.split(data_set[:,0:2], data_set[:, 2]):
                train_data, test_data = data_set[train_index], data_set[test_index]
                one_fold_dataset = []
                one_fold_dataset.append(train_data)
                one_fold_dataset.append(test_data)
                all_fold_dataset.append(one_fold_dataset)

            self.all_data_set.append(all_fold_dataset)

        #将序列转为数值特征表示
        self.smi_input = np.zeros((len(self.drug_smi),self.smi_max_len),dtype=int)
        self.fas_input = np.zeros((len(self.protein_fas),self.fas_max_len),dtype=int)

        for i in range(len(self.drug_smi)):
            for j in range(len(self.drug_smi[i,1]) - self.smi_ngram +1):
                key = self.drug_smi[i,1][j:j + self.smi_ngram]
                self.smi_input[i,j] = self.smi_dict[key]

        for i in range(len(self.protein_fas)):
            for j in range(len(self.protein_fas[i,1]) - self.fas_ngram +1):
                key = self.protein_fas[i,1][j:j + self.fas_ngram]
                self.fas_input[i,j] = self.fas_dict[key]

    def get_train_batch(self,repeat_nums,flod_nums,batch_size):

        train_drugs = []
        train_proteins = []
        train_affinity = []
        train_data = self.all_data_set[repeat_nums][flod_nums][0]
        #print(type(train_data))     ndarray类型,没打乱，tag先1后0
        for index,(i,j,tag) in enumerate(train_data):
            train_drugs.append(self.smi_input[i])
            train_proteins.append(self.fas_input[j])
            train_affinity.append(tag)

        train_drugs = np.array(train_drugs)
        train_proteins = np.array(train_proteins)
        train_affinity = np.array(train_affinity)

        #打乱训练数据和标签，通过打乱索引从而打乱数据，数据量很大时能节约内存，并且每次生成的数据都不一样
        data_index = np.arange(len(train_drugs))        #生成下标
        np.random.shuffle(data_index)
        train_drugs = train_drugs[data_index]
        train_proteins = train_proteins[data_index]
        train_affinity = train_affinity[data_index]

        #迭代返回
        sindex = 0
        eindex = batch_size
        while eindex < len(train_drugs):
            tra_dg_batch = train_drugs[sindex:eindex,:]
            tra_pt_batch = train_proteins[sindex:eindex,:]
            tra_tag_batch = train_affinity[sindex:eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield tra_dg_batch,tra_pt_batch,tra_tag_batch

        if eindex >= len(train_drugs):
            tra_dg_batch = train_drugs[sindex:,:]
            tra_pt_batch = train_proteins[sindex:,:]
            tra_tag_batch = train_affinity[sindex:]
            yield tra_dg_batch,tra_pt_batch,tra_tag_batch

    def get_test_batch(self,repeat_nums,flod_nums,batch_size):
        #测试可以一次性输入吧，如果内存充足的话
        train_drugs = []
        train_proteins = []
        train_affinity = []
        train_data = self.all_data_set[repeat_nums][flod_nums][1]
        #print(type(train_data))     ndarray类型,没打乱，tag先1后0
        for index,(i,j,tag) in enumerate(train_data):
            train_drugs.append(self.smi_input[i])
            train_proteins.append(self.fas_input[j])
            train_affinity.append(tag)

        train_drugs = np.array(train_drugs)
        train_proteins = np.array(train_proteins)
        train_affinity = np.array(train_affinity)

        #打乱训练数据和标签，通过打乱索引从而打乱数据，数据量很大时能节约内存，并且每次生成的数据都不一样
        data_index = np.arange(len(train_drugs))        #生成下标
        np.random.shuffle(data_index)
        train_drugs = train_drugs[data_index]
        train_proteins = train_proteins[data_index]
        train_affinity = train_affinity[data_index]

        #迭代返回
        sindex = 0
        eindex = batch_size
        while eindex < len(train_drugs):
            tra_dg_batch = train_drugs[sindex:eindex,:]
            tra_pt_batch = train_proteins[sindex:eindex,:]
            tra_tag_batch = train_affinity[sindex:eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield tra_dg_batch,tra_pt_batch,tra_tag_batch

        if eindex >= len(train_drugs):
            tra_dg_batch = train_drugs[sindex:,:]
            tra_pt_batch = train_proteins[sindex:,:]
            tra_tag_batch = train_affinity[sindex:]
            yield tra_dg_batch,tra_pt_batch,tra_tag_batch

#ht = HetrDataset()





