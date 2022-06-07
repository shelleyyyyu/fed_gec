import logging
import os
import sys
import random
from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
try:
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager


class FedOptServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        #train_data_loss_list = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_DATA_LOSS_LIST)
        #train_data_list = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_DATA_LIST)
        
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            # aggregate the params 
            global_model_params = self.aggregator.aggregate()
            # Test on server: test_on_the_server()
            self.aggregator.test_on_server_for_all_clients(self.round_idx)
            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                post_complete_message_to_sweep_process(self.args)
                self.finish()
                return

            logging.info(self.is_preprocessed)
            # sampling clients
            if self.is_preprocessed:
                logging.info(self.preprocessed_client_lists)
                logging.info(self.round_idx)
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)
             
            
            logging.info(client_indexes)
            
            print("size = %d" % self.size)
            if self.args.is_mobile == 1:
                print("transform_tensor_to_list")
                global_model_params = transform_tensor_to_list(global_model_params)
                
            #TODO: -HERE NEED TO DECIDE THE NEXT AUGMENT TYPE(CLIENT)
            
            next_augment_client = random.choice(list([int(i) for i in range(1, self.size)]))
            next_augment_percentage = random.choice(list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

            logging.info('next_augment_client: %s'%(str(next_augment_client)))

            for receiver_id in range(1, self.size):
                if str(receiver_id) == str(next_augment_client):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params, client_indexes[receiver_id - 1],
                                                           if_augment='1', next_augment_percentage=next_augment_percentage)
                else:
                    self.send_message_sync_model_to_client(receiver_id, global_model_params, client_indexes[receiver_id - 1], 
                                                           if_augment='0', next_augment_percentage=next_augment_percentage)

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, if_augment='0', next_augment_percentage=0.1):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_IF_AUGMENT, str(if_augment))
        message.add_params(MyMessage.MSG_ARG_KEY_AUGMENT_PERCENTAGE, str(next_augment_percentage))
        self.send_message(message)
