import logging
import os
import sys
import random
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_sizes=(512,),
                 activation="Tanh", bias=True, dropout=0.1):
        super(FeedForward, self).__init__()
        self.activation = getattr(nn, activation)()

        n_inputs = [input_dim] + list(hidden_sizes)
        n_outputs = list(hidden_sizes) + [out_dim]
        self.linears = nn.ModuleList([nn.Linear(n_in, n_out, bias=bias)
                                      for n_in, n_out in zip(n_inputs, n_outputs)])
        self.num_layer = len(self.linears)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_):
        x = input_
        i = 0
        for linear in self.linears:
            x = linear(x)
            if i < self.num_layer - 1:
                x = self.dropout_layer(x)
            x = self.activation(x)
            i += 1
        return x
        
        
class BuildState():
    def build(self, loss_dict, f0_5_dict, precision_dict, recall_dict):
        state = []
        for (key, loss_value), (key, f0_5_value), (key, precision_value), (key, recall_value) in zip(loss_dict.item(), f0_5_dict.item(), precision_dict.item(), recall_dict.item()):
            state.extend([loss_value, f0_5_value, precision_value, recall_value])
        return state