from .utils import transform_tensor_to_list
import logging
from da.augment_by_rule import DataAugmentationByRule
from data_preprocessing.base.base_preprocessor import BasePreprocessor
from data_preprocessing.base.base_data_loader import BaseDataLoader
import numpy as np

class FedOptTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device,
                 args, model_trainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num
        # self.train_local = self.train_data_local_dict[client_index]
        # self.local_sample_number = self.train_data_local_num_dict[client_index]
        
        self.dataAugmentationByRule = DataAugmentationByRule()
        self.device = device
        self.args = args
        # self.trainer.preprocessor

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index, train_data_loss_list, train_data_list, if_augment='0', augment_percentage=0.1):
        self.client_index = client_index
        # Before data augmentation
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        logging.info('Before Augmentation: %s; len of examples: %d' %(str(client_index), len(self.train_local.examples)))
        # After data augmentation
        augment_train_local, augment_train_data_local_num  = self.data_augmentation(self.train_local, client_index, train_data_loss_list, train_data_list, if_augment, augment_percentage)
        self.train_local = augment_train_local
        self.train_data_local_dict[client_index] = self.train_local
        self.train_data_local_num_dict[client_index] = augment_train_data_local_num
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        logging.info('After Augmentation: %s; len of examples: %d' %(str(client_index), len(self.train_local.examples)))
    
    def data_augmentation(self, train_local, client_index, train_data_loss_list, train_data_list, if_augment='0', augment_percentage=0.1):
        if client_index == 0:
            augment_type = 'delete'
        elif client_index == 1:
            augment_type = 'insert'
        elif client_index == 2:
            augment_type = 'paraphrase'
        elif client_index == 3:
            augment_type = 'substitution'

        if if_augment == '1' and len(train_data_loss_list) > 0 and len(train_data_list) > 0:
            logging.info("Client %s : augmenting local %s-type data (augment_percentage: %.2f)" %(str(client_index), str(augment_type), augment_percentage))
            
            aug_train_data = {}
            aug_train_data['X'] = []
            aug_train_data['y'] = []
            
            sort_train_indices = np.argsort(train_data_loss_list)[::-1]
            to_augment_threadshold = int(len(sort_train_indices) * augment_percentage)
            
            for idx, i in enumerate(sort_train_indices):
                origin_input_data = train_local.examples[i].input_text
                output_data = train_local.examples[i].target_text
                aug_train_data['X'].append(origin_input_data)
                aug_train_data['y'].append(output_data)
                
                if idx < to_augment_threadshold:
                    #logging.info('index %s have augmented' %(str(i)))
                    rule_augment_data = self.dataAugmentationByRule.augment_by_type(output_data, augment_type=augment_type)
                    if rule_augment_data:
                        aug_train_data['X'].append(rule_augment_data)
                        aug_train_data['y'].append(output_data)
                #else:
                    #logging.info('index %s not augmented' %(str(idx)))

            aug_train_examples, aug_train_features, aug_train_dataset = \
            self.trainer.preprocessor.transform(aug_train_data['X'], aug_train_data['y'])
            aug_train_loader = BaseDataLoader(aug_train_examples, aug_train_features, aug_train_dataset,
                                   batch_size=self.args.train_batch_size,
                                   num_workers=0,
                                   pin_memory=True,
                                   drop_last=False)
            return aug_train_loader, len(aug_train_loader)
        else:
            logging.info("Client %s will not augment local data" %(str(client_index)))
            return train_local, len(train_local) 
        

    def train(self, round_idx = None):
        self.args.round_idx = round_idx
        logging.info('Train round %s: len of examples: %d' %(str(round_idx), len(self.train_local.examples)))
        train_data_loss_list, train_data_list, loss = self.trainer.train(self.train_local, self.device, self.args)
        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number, train_data_loss_list, train_data_list, loss#, f0_5, recall, precision
