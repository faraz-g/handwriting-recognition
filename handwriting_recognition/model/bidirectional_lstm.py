import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """From: https://github.com/dmitrijsk/AttentionHTR/blob/main/model/modules/sequence_modeling.py"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size

        return output
