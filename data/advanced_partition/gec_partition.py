from typing import Counter
import h5py
import argparse
import numpy as np
import json
import math
from decimal import *
import random
import math
from collections import Counter
from numpy.lib.shape_base import split

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_file",
        type=str,
        default="/Users/shelly/Desktop/laiye/project/cgec/code/FedNLP/data/raw_data_loader/test/nlpcc_%s_data.h5",
        metavar="DF",
        help="data pickle file path",
    )

    parser.add_argument(
        "--partition_file",
        type=str,
        default="/Users/shelly/Desktop/laiye/project/cgec/code/FedNLP/data/raw_data_loader/test/nlpcc_%s_data_partition.h5",
        metavar="PF",
        help="partition pickle file path",
    )

    args = parser.parse_args()
    print("start reading data")
    client_num = 4

    print("retrieve data")
    insert_data = h5py.File(args.data_file%('insert'), "r")
    insert_attributes = json.loads(insert_data["attributes"][()])
    print("insert_attributes", insert_attributes.keys())

    delete_data = h5py.File(args.data_file%('delete'), "r")
    delete_attributes = json.loads(delete_data["attributes"][()])
    print("delete_attributes", delete_attributes.keys())

    substitution_data = h5py.File(args.data_file%('substitution'), "r")
    substitution_attributes = json.loads(substitution_data["attributes"][()])
    print("substitution_attributes", substitution_attributes.keys())

    paraphrase_data = h5py.File(args.data_file%('paraphrase'), "r")
    paraphrase_attributes = json.loads(paraphrase_data["attributes"][()])
    print("paraphrase_attributes", paraphrase_attributes.keys())

    test_data = h5py.File(args.data_file % ('test'), "r")
    # insert
    partition_result_train = [insert_data, delete_data, substitution_data, delete_data]
    partition_result_test = [test_data]

    print("store data in h5 data")
    partition = h5py.File(args.partition_file, "a")
    partition["/natural" + "_clients=%d" %(client_num) + "/n_clients"] = client_num
    partition["/natural" + "_clients=%d" %(client_num) + "/natural_factors"] = 4
    for partition_id in range(client_num):
        train = partition_result_train[partition_id]
        test = partition_result_test[0]
        train_path = ("/natural" + "_clients=%d" %
                    (client_num) + "/partition_data/" +
                    str(partition_id) + "/train/")
        test_path = ("/natural" + "_clients=%d" %
                    (client_num) + "/partition_data/" +
                    str(partition_id) + "/test/")
        print(train)
        partition[train_path] = train
        print(train)
        print(partition_id, ': partition_id')
        partition[test_path] = test
        print(partition_id, ': partition_id')

    partition.close()
    insert_data.close()
    delete_data.close()
    substitution_data.close()
    paraphrase_data.close()
    test_data.close()


main()
