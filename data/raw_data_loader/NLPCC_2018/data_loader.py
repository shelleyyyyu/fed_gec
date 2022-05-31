import os
import random
import h5py
import numpy as np
import json
from data.raw_data_loader.base.base_raw_data_loader import Seq2SeqRawDataLoader


class RawDataLoader(Seq2SeqRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_file_name = ["nlpcc_insert_data.txt", "nlpcc_delete_data.txt", "nlpcc_substitution_data.txt", "nlpcc_paraphrase_data.txt"]
        self.test_file_name = ["nlpcc_insert_test_data.txt", "nlpcc_delete_test_data.txt", "nlpcc_substitution_test_data.txt", "nlpcc_paraphrase_test_data.txt"]
        self.attributes["train_index_list"] = []
        self.attributes["test_index_list"] = []
        self.attributes['label_index_list'] = []

    def  load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0:
            train_size = 0
            test_size = 0
            for idx, label in enumerate(['insert', 'delete', 'substitution', 'paraphrase']):
                train_size += self.process_data_file(os.path.join(self.data_path, self.train_file_name[idx]), label)

            for idx, label in enumerate(['insert', 'delete', 'substitution', 'paraphrase']):
                test_size += self.process_data_file(os.path.join(self.data_path, self.test_file_name[idx]), label)


            #for train_dataset in self.train_file_name:
            #    label = train_dataset.split("_")[1]
            #    train_size += self.process_data_file(os.path.join(self.data_path, train_dataset), label)
            #for test_dataset in [self.test_file_name[0]]:
            #    label = test_dataset.split("_")[1]
            #    test_size += self.process_data_file(os.path.join(self.data_path, test_dataset), label)

            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [i for i in range(train_size, train_size + test_size)]
            self.attributes["index_list"] = self.attributes["train_index_list"] + self.attributes["test_index_list"]
            assert len(self.attributes['index_list']) == len(self.attributes['label_index_list'])


    def process_data_file(self, file_path, label):
        cnt = 0
        print(label)
        with open(os.path.join(file_path), 'r', encoding='utf-8',errors="ignore") as f:
            for line in f:
                assert len(self.X) == len(self.Y)
                line = line.strip()
                if line:
                    temp = line.split("\t")
                    idx = len(self.X)
                    self.X[idx] = ''.join(temp[0].strip().split(' '))
                    self.Y[idx] = ''.join(temp[1].strip().split(' '))
                    self.attributes["label_index_list"].append(label)
                cnt += 1
        return cnt


    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.X.keys():
            f["X/" + str(key)] = self.X[key]
            f["Y/" + str(key)] = self.Y[key]
        f.close()
