import logging
import os
import sys
import random
import torch
from .message_define import MyMessage
from collections import OrderedDict
from collections import namedtuple
import torch.optim as optim
import numpy as np

from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .ActorCriticAlgorithm import PolicyNet, CriticNet
from .ActorCriticAlgorithm import BuildState

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
try:
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager

class FedOptServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None, validate_round_num=3, gamma=0.99):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        
        # For RL
        self.build_state = BuildState()
        self.policy = PolicyNet(4, 4, hidden_sizes=(128, 64))
        self.critic = CriticNet(4, 1, hidden_sizes=(128, 64))
        
        self.running_reward = 0.0
        self.ep_reward = 0.0
        
        self.validate_round_num = validate_round_num
        
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value']) 
        self.policyOptimizer = optim.Adam(self.policy.parameters(), lr=3e-2)
        self.criticOptimizer = optim.Adam(self.critic.parameters(), lr=3e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = gamma
        torch.autograd.set_detect_anomaly(True)

        
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
        train_loss_result = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_LOSS)
        #train_f0_5_result = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_F0_5)
        #train_recall_result = msg_params.get(MyMessage.MSG_ARG_KEY_PRECISION)
        #train_precision_result = msg_params.get(MyMessage.MSG_ARG_KEY_RECALL)
        
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        self.aggregator.add_local_observations(sender_id - 1, train_loss_result)
        #, train_f0_5_result, train_precision_result, train_recall_result)
        b_all_observed = self.aggregator.check_whether_all_observed()
        logging.info("b_all_observed = " + str(b_all_observed))
        logging.info('sender_id: %s'%(str(sender_id)))
        
        if b_all_received and b_all_observed:
            # aggregate the params 
            global_model_params = self.aggregator.aggregate()
            # Test on server: test_on_the_server()
            result, _ = self.aggregator.test_on_server_for_all_clients(self.round_idx) 
            # result = {'eval_loss': 15.566793060302734, 'f0_5_score': 0.03124, 'recall_score': 0.01666, 'precision_score': 0.24}
            
            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                #post_complete_message_to_sweep_process(self.args)
                logging.info("training is finished! \n%s\n" % (str(args)))
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
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)
            
            logging.info("client_indexes = %s" %str(client_indexes))
            
            print("size = %d" % self.size)
            if self.args.is_mobile == 1:
                print("transform_tensor_to_list")
                global_model_params = transform_tensor_to_list(global_model_params)
                
            #TODO: HERE NEED TO DECIDE THE NEXT AUGMENT TYPE(CLIENT)
            
            loss_dict = self.aggregator.get_local_observations()
            #loss_dict['valid'] = result['eval_loss']
            #f0_5_dict['valid'] = result['f0_5_score']
            #precision_dict['valid'] = result['precision_score']
            #recall_dict['valid'] = result['recall_score']
            cur_learning_state = self.build_state.build(loss_dict)
            
            # PolicyNet
            action_prob = self.policy(cur_learning_state) #, action_percentages
            sample_from = Categorical(action_prob)
            action = sample_from.sample()
            # CriticNet
            state_value = self.critic(cur_learning_state)
            
            action_log_probs = sample_from.log_prob(action)
            self.policy.saved_log_probs.append(action_log_probs)
            self.critic.saved_values.append(state_value)

            
            # State the next augment client and percentages
            next_augment_client = action
            next_augment_percentage = action_prob[next_augment_client]
            next_augment_client += 1
            
            logging.info('next_augment_client: %s' %(str(next_augment_client)))
            logging.info('next_augment_percentage: %s' %(str(next_augment_percentage)))
            
            # next_augment_client = random.choice(list([int(i) for i in range(1, self.size)]))
            # next_augment_percentage = random.choice(list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
            
            for receiver_id in range(1, self.size):
                if str(receiver_id) == str(next_augment_client.item()):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params, client_indexes[receiver_id - 1],
                                                           if_augment='1', next_augment_percentage=next_augment_percentage)
                else:
                    self.send_message_sync_model_to_client(receiver_id, global_model_params, client_indexes[receiver_id - 1], 
                                                           if_augment='0', next_augment_percentage=next_augment_percentage)
            
            # Optimize the policy network and critic network
            #loss_dict['valid'] = result['eval_loss']
            #f0_5_dict['valid'] = result['f0_5_score']
            #precision_dict['valid'] = result['precision_score']
            #recall_dict['valid'] = result['recall_score']
            rewards = torch.sum(torch.Tensor([result['eval_loss'], result['f0_5_score'], result['precision_score'], result['recall_score']])) # torch.sum(cur_learning_state)
            self.policy.rewards.append(rewards)

            # Validation every $validate_round_num$ round 
            if self.round_idx != 0 and self.round_idx % self.validate_round_num == 0:
                logging.info('Round: %d; Rewards: %.4f' %(self.round_idx, torch.sum(torch.stack(self.policy.rewards), dim=-1)))

                R = 0
                saved_log_probs = self.policy.saved_log_probs
                saved_values = self.critic.saved_values
                policy_losses, value_losses, returns = [], [], [] 
                # calculate the true value using rewards returned from the environment
                for r in self.policy.rewards[::-1]:
                    # calculate the discounted value
                    R = r + self.gamma * R
                    returns.insert(0, R)

                returns = torch.tensor(returns)
                
                if len(returns) > 1:
                    # Do Normalization
                    returns = (returns - returns.mean()) / (returns.std() + self.eps)

                for (_log_prob, _value, _return) in zip(saved_log_probs, saved_values, returns):
                    _advantage = _return - _value.item()

                    # calculate actor (policy) loss 
                    policy_losses.append(-(_log_prob * _advantage))

                    # calculate critic (value) loss using L1 smooth loss
                    value_losses.append(F.smooth_l1_loss(_value, torch.tensor([_return])))
                
                # sum up all the values of policy_losses and value_losses
                policy_loss = torch.stack(policy_losses).sum()
                critic_loss = torch.stack(value_losses).sum()
                
                # reset gradients
                self.policyOptimizer.zero_grad()
                self.criticOptimizer.zero_grad()

                critic_loss.backward(retain_graph=True)
                policy_loss.backward(retain_graph=True)
                
                self.criticOptimizer.step()
                self.policyOptimizer.step()

                # reset rewards and action buffer
                del self.policy.rewards[:]
                del self.policy.saved_log_probs[:]
                del self.critic.saved_values[:]
                
            

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
        message.add_params(MyMessage.MSG_ARG_KEY_AUGMENT_PERCENTAGE, next_augment_percentage)
        self.send_message(message)
