import torch
import torch.nn as nn


class VerticalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=20):
        super(VerticalAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, input):
        hidden = self.linear1(input)
        hidden = self.tanh(hidden)
        attention_weights = self.linear2(hidden)
        attention_weights = self.softmax(attention_weights)
        output = torch.sum(attention_weights * input, dim=1)
        return output, attention_weights