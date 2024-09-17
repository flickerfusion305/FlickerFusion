import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer

class EntityAttentionSWAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EntityAttentionSWAgent, self).__init__()
        self.args = args

        # input_shape: dimension of the raw input features for each token (or entity)
        # args.attn_embed_dim: dimension of the token

        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        # if args.pooling_type is None:

        if args.dropout: self.dropout1 = nn.Dropout(p = args.dropout_p)
        self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim*(1+args.double_attn), args)
        # else:
        #     self.attn = EntityPoolingLayer(args.attn_embed_dim,
        #                                    args.attn_embed_dim,
        #                                    args.attn_embed_dim*(1+args.double_attn),
        #                                    args.pooling_type,
        #                                    args)
        if args.dropout: self.dropout2 = nn.Dropout(p = args.dropout_p)

        self.fc2 = nn.Linear(args.attn_embed_dim, args.sw_input_dim)
        if args.dropout: self.dropout3 = nn.Dropout(p = args.dropout_p)

        self.fc3 = nn.Linear(args.sw_input_dim * args.sw_window_size, args.sw_output_dim)
        if args.dropout: self.dropout4 = nn.Dropout(p = args.dropout_p)

        self.fc4 = nn.Linear(args.sw_output_dim, args.sw_decoder_hidden_layer)
        if args.dropout: self.dropout5 = nn.Dropout(p = args.dropout_p)

        self.fc5 = nn.Linear(args.sw_decoder_hidden_layer, args.n_actions)
        self.attn_weights=None

    # initialize the hidden states of the agents before the start of a new episode or sequence.
    def init_hidden(self):
        # make hidden states on same device as model
        self.attn_weights=None
        return self.fc1.weight.new(1, (self.args.sw_window_size - 1) * self.args.sw_input_dim).zero_() # Use hidden state as a window

    def forward(self, inputs, hidden_state, ret_attn_logits=None, msg=None, ret_attn_weights=False):
        entities, obs_mask, entity_mask = inputs
        # bs: batch_size, ts: time_steps, ne: num_entities, ed: entity_dim
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        # entities: Entity representations
        # shape: batch size, # of entities, embedding dimension
        x1 = F.relu(self.fc1(entities))
        if self.args.dropout: x1 = self.dropout1(x1)
        attn_outs = self.attn(x1, pre_mask=obs_mask,
                              post_mask=agent_mask,
                              ret_attn_logits=ret_attn_logits,
                              ret_attn_weights=ret_attn_weights)
        if ret_attn_logits is not None:
            x2, attn_logits = attn_outs
        elif ret_attn_weights:
            x2, attn_weights = attn_outs
            if self.attn_weights is None:
                self.attn_weights = attn_weights
            else:
                self.attn_weights=th.cat([self.attn_weights, attn_weights], dim=0)
        else:
            x2 = attn_outs
        for i in range(self.args.repeat_attn):
            attn_outs = self.attn(x2, pre_mask=obs_mask,
                              post_mask=agent_mask,
                              ret_attn_logits=ret_attn_logits)
            if ret_attn_logits is not None:
                x2, attn_logits = attn_outs
            else:
                x2 = attn_outs
        # if self.args.self_loc:
        #     loc = self.self_fc(entities[:, :self.args.n_agents])
        #     x2 = th.cat([x2, loc], dim=2)
        if self.args.dropout: x2 = self.dropout2(x2) # bs, na, attn_embed

        x3 = F.relu(self.fc2(x2))
        if self.args.dropout: x3 = self.dropout3(x3)

        x3 = x3.reshape(bs, ts, self.args.n_agents, -1) # bs, ts, na, sw_input
        xs = []
        window = hidden_state.reshape(bs, self.args.n_agents, -1) # bs, na, (sw_window-1)*sw_input
        for t in range(ts):
            curr_x = th.cat((window, x3[:,t]), axis = 2) # bs, na, (sw_window)*sw_input
            x_out = F.relu(self.fc3(curr_x.view(bs, self.args.n_agents, -1))) # flatten for sliding window input
            xs.append(x_out.view(bs, self.args.n_agents, -1))
            window = curr_x[:,:,-(self.args.sw_window_size - 1) * self.args.sw_input_dim:] # update window

        xs = th.stack(xs, dim=1)
        xs = F.relu(self.fc4(xs))
        q = self.fc5(xs)

        # zero out output for inactive agents
        q = q.reshape(bs, ts, self.args.n_agents, -1)
        q = q.masked_fill(agent_mask.reshape(bs, ts, self.args.n_agents, 1).bool(), 0)

        return q, window
