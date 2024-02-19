from torch import nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class SimpleLSTM(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, num_layers, dropout_p, bidirectional):
        super(SimpleLSTM, self).__init__()
        self.build(x_dim, y_dim, hidden_size, num_layers, dropout_p, bidirectional)

    def build(self, x_dim, y_dim, hidden_size, num_layers, dropout_p, bidirectional):
        self.lstm = nn.LSTM(
            x_dim,
            hidden_size,
            num_layers,
            dropout=dropout_p,
            bidirectional=bidirectional,
            batch_first=True,
        )
        d = 2 if bidirectional else 1
        self.output_layer = nn.Linear(d * hidden_size, y_dim)

    def forward(self, seqs, seqs_len=None):
        if seqs_len != None:
            seqs = pack_padded_sequence(seqs, seqs_len, batch_first=True)
        lstm_output, _ = self.lstm(seqs)
        if seqs_len != None:
            lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)
        seqs_pred = self.output_layer(lstm_output)
        return seqs_pred

class LSTM_FF(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, num_layers,
                 dropout_p, bidirectional):
        super(LSTM_FF, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.build()

    def build(self):
        # Build LSTM
        self.lstm = nn.LSTM(
            self.x_dim,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout_p,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        d = 2 if self.bidirectional else 1
        self.output_layer = nn.Linear(d * self.hidden_size, self.y_dim)
        self.activation = nn.Sigmoid()

    def forward(self, seqs, seqs_len=None):
        with torch.backends.cudnn.flags(enabled=False):
            batch_size = seqs.shape[0]
            if seqs_len != None:
                seqs = pack_padded_sequence(seqs, seqs_len, batch_first=True)

            # We get the last hidden state to run the classif
            seqs, (hidden_state,cell_state) = self.lstm(seqs)

            if seqs_len != None:
                seqs, _ = nn.utils.rnn.pad_packed_sequence(seqs, batch_first=True)

            # Extract last hidden state
            d = 2 if self.bidirectional else 1
            # Concatenating the final forward and backward hidden states
            #out = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim=1)
            #out = hidden_state[0].view(self.num_layers_lstm, d, batch_size, self.hidden_size_lstm)[-1]
            hidden_state = hidden_state.view(self.num_layers, d, batch_size, self.hidden_size)[-1]
            # Handle directions
            if d == 1:
                out = hidden_state.squeeze(0)
            elif d == 2:
                h1, h2 = hidden_state[0], hidden_state[1]
                out = torch.cat((h1, h2), 1)
            out = self.activation(self.output_layer(out))
            return out.squeeze(-1)
