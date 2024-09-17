import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNSkipAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNSkipAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_input_dim)
        if args.dropout: self.dropout1 = nn.Dropout(p = args.dropout_p)

        if args.aggregate_hidden_layer_num == 0:
            self.aggregate_hidden = nn.Linear(args.rnn_hidden_dim + args.rnn_input_dim * args.skip_size, args.rnn_hidden_dim)
        else:
            self.aggregate_hiddens = []
            hidden_layer_sizes = [args.rnn_hidden_dim + args.rnn_input_dim * args.skip_size] + args.aggregate_hidden_layer_size + [args.rnn_hidden_dim]
            for i in range(args.aggregate_hidden_layer_num + 1):
                self.aggregate_hiddens.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]).to("cuda")) # fix if cpu

        self.rnn = nn.GRUCell(args.rnn_input_dim, args.rnn_hidden_dim)
        if args.dropout: self.dropout2 = nn.Dropout(p = args.dropout_p)

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # if args.self_loc:
        #     if args.env == "sc2":
        #         if args.env_args["map_name"] == "6h_vs_8z":
        #             loc_shape = 5
        #         elif args.env_args["map_name"] == "3s5z_vs_3s6z":
        #             loc_shape = 8
        #         else:
        #             raise NotImplementedError
        #     else:
        #         raise NotImplementedError
        #     self.fc_sl1 = nn.Linear(loc_shape, args.rnn_hidden_dim)
        #     self.fc_sl2 = nn.Linear(args.rnn_hidden_dim*2, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim + self.args.rnn_input_dim * self.args.skip_size).zero_()

    def forward(self, inputs, pre_h):
        bs, ts, na, os = inputs.shape

        x = F.relu(self.fc1(inputs))

        # if self.args.self_loc:
        #     if self.args.env_args["map_name"] == "6h_vs_8z":
        #         self_f = th.cat([inputs[:,:,:,:4], inputs[:,:,:,77:78]], dim=-1)
        #     elif self.args.env_args["map_name"] == "3s5z_vs_3s6z":
        #         self_f = th.cat([inputs[:,:,:,:4], inputs[:,:,:,132:136]], dim=-1)
        #     x_loc = F.relu(self.fc_sl1(self_f))
        #     x = F.relu(self.fc_sl2(th.cat([x, x_loc], dim=-1)))
        
        if self.args.dropout: x = self.dropout1(x)

        pre_h = pre_h.reshape(-1, self.args.rnn_hidden_dim + self.args.rnn_input_dim * self.args.skip_size)
        hs = []

        for t in range(ts):
            # Aggregate pre_h -> h for GRU
            if self.args.aggregate_hidden_layer_num == 0:
                h = F.relu(self.aggregate_hidden(pre_h))
            else:
                for i in range(self.args.aggregate_hidden_layer_num + 1):
                    if i == 0:
                        h = F.relu(self.aggregate_hiddens[i](pre_h))
                    else:
                        h = F.relu(self.aggregate_hiddens[i](h))

            curr_x = x[:, t].reshape(-1, self.args.rnn_input_dim) # bs*na, rnn_input
            curr_h = self.rnn(curr_x, h) # bs*na, rnn_hidden
            hs.append(curr_h.view(bs, na, self.args.rnn_hidden_dim))
            pre_h = th.cat((curr_h, pre_h[:,-self.args.rnn_input_dim*(self.args.skip_size-1):], curr_x), axis = 1)
        hs = th.stack(hs, dim=1)  # Concat over time

        if self.args.dropout: hs = self.dropout2(hs)
        q = self.fc2(hs)

        return q, pre_h # only need last concat_hidden state for sequential inference (ts = 1)
