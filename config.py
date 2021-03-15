class Config():
    def __init__(self):
        # model parmeters
        self.use_gpu = True

        self.repeat_nums = 1       #
        self.fold_nums = 10         #The numbers of crossflod-validation
        self.neg_samp_ratio = 10   #NO. negative : NO. positive = nag_samp_ratio:1

        self.num_epochs = 30      #The number of epochs to train
        self.common_epochs = 1
        self.predict_epochs = 1
        self.batch_size = 128     #余下的batch可能比较小，会导致auc报错

        #和神经网络相关
        self.embedding_size = 64          # The embedding size for every word
        self.num_filters= 128               #The number of filter for convolutional layers  #[32,64]
        self.common_size = 32
        self.smi_ConvWin_size=12           #The size of filters for convolutional layers of smiles encoder  #[4,6,8]
        self.fas_ConvWin_size=20           #The size of filters for convolutional layers of protein encoder #[8,12]
        self.common_learn_rate = 0.00001          #Learning Rate  #0.001 [1,0.1,0.01,0.001,0.0001]
        self.pre_learn_rate = 0.00001

        #和序列数据相关，下面的参数都不需要改
        self.smi_n_gram = 1
        self.fas_n_gram = 3
        self.smi_dict_len=38              #The length of dictionary
        self.fas_dict_len=6658            #The length of dictionary
        self.fasta_max_len = 2300        #The max sequense length of protein
        self.smiles_max_len = 400        #The max sequense length of smiles

        # The path of data
        self.dg_pt_path = 'data/drug_protein.csv'

        self.smi_dict_path = 'data/smi_dict.pickle'
        self.fas_dict_path = 'data/fas_dict.pickle'

        self.dg_smiles_path = 'data/drug_smiles.csv'
        self.pt_fasta_path = 'data/protein_fasta.csv'


