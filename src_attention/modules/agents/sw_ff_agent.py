import torch as th
import torch.nn as nn
import torch.nn.functional as F

class SlidingFFAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SlidingFFAgent, self).__init__()
        self.args = args

        # Easiest to reuse rnn_hidden_dim variable
        self.fc1 = nn.Linear(input_shape, args.sw_hidden_dim)
        self.fc2 = nn.Linear(args.sw_hidden_dim, args.sw_input_dim)
        self.fc3 = nn.Linear(args.sw_input_dim * args.sw_window_size, args.sw_output_dim)
        self.fc4 = nn.Linear(args.sw_output_dim, args.sw_decoder_hidden_layer)
        self.fc5 = nn.Linear(args.sw_decoder_hidden_layer, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, (self.args.sw_window_size - 1) * self.args.sw_input_dim).zero_() # Use hidden state as a window

    def forward(self, inputs, hidden_state):
        bs, ts, na, os = inputs.shape

        x = F.relu(self.fc1(inputs)) # bs, ts, na, sw_input
        x = F.relu(self.fc2(x))
        xs = []
        window = hidden_state.reshape(bs, na, -1) # bs, na, (sw_window-1)*sw_input

        for t in range(ts):
            curr_x = th.cat((window, x[:,t]), axis = 2) # bs, na, (sw_window)*sw_input
            x_out = F.relu(self.fc3(curr_x.view(bs, na, -1))) # flatten for sliding window input
            xs.append(x_out.view(bs, na, -1))
            window = curr_x[:,:,-(self.args.sw_window_size - 1) * self.args.sw_input_dim:] # update window

        xs = th.stack(xs, dim=1)
        xs = F.relu(self.fc4(xs))
        q = self.fc5(xs)
        return q, window
                
        # x = F.relu(self.fc1(inputs))
        # # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # h = F.relu(self.fc2(x))
        # q = self.fc3(h)
        # return q, h
