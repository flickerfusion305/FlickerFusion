import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_input_dim)
        if args.dropout: self.dropout1 = nn.Dropout(p = args.dropout_p)
        # add layer 128
        # add layer 128
        
        if args.rnn_layer_dim > 1:
            self.rnn = nn.GRU(args.rnn_input_dim, args.rnn_hidden_dim, args.rnn_layer_dim, batch_first=True)
        else:
            self.rnn = nn.GRUCell(args.rnn_input_dim, args.rnn_hidden_dim)
        if args.dropout: self.dropout2 = nn.Dropout(p = args.dropout_p)

        # add layer 128
        # add layer 128
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        if args.self_loc:
            if args.env == "sc2":
                if args.env_args["map_name"] == "6h_vs_8z":
                    loc_shape = 5
                elif args.env_args["map_name"] == "3s5z_vs_3s6z":
                    loc_shape = 8
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            self.fc_sl1 = nn.Linear(loc_shape, args.rnn_hidden_dim)
            self.fc_sl2 = nn.Linear(args.rnn_hidden_dim*2, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        bs, ts, na, os = inputs.shape

        x = F.relu(self.fc1(inputs))
        if self.args.self_loc:
            if self.args.env_args["map_name"] == "6h_vs_8z":
                self_f = th.cat([inputs[:,:,:,:4], inputs[:,:,:,77:78]], dim=-1)
            elif self.args.env_args["map_name"] == "3s5z_vs_3s6z":
                self_f = th.cat([inputs[:,:,:,:4], inputs[:,:,:,132:136]], dim=-1)
            x_loc = F.relu(self.fc_sl1(self_f))
            x = F.relu(self.fc_sl2(th.cat([x, x_loc], dim=-1)))
        
        if self.args.dropout: x = self.dropout1(x)

        if self.args.rnn_layer_dim == 1: #Single GRU Cell
            h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            hs = []
            for t in range(ts):
                curr_x = x[:, t].reshape(-1, self.args.rnn_input_dim)
                h = self.rnn(curr_x, h)
                hs.append(h.view(bs, na, self.args.rnn_hidden_dim))
            hs = th.stack(hs, dim=1)  # Concat over time
            if self.args.dropout: hs = self.dropout2(hs)
            q = self.fc2(hs)
        
        else: #Multilayer GRU
            h = hidden_state.reshape(self.args.rnn_layer_dim, -1, self.args.rnn_hidden_dim)
            hs = []
            x = x.reshape(-1, ts, self.args.rnn_input_dim)
            x, hs = self.rnn(x, h.contiguous()) # x: (bs * na, ts, h) h: (num_layers, bs * na, h)

            x = x.reshape(bs, ts, na, self.args.rnn_hidden_dim)
            hs = hs.reshape(bs, na, self.args.rnn_layer_dim, self.args.rnn_hidden_dim)

            if self.args.dropout: x = self.dropout2(x)
            q = self.fc2(x)

        return q, hs
