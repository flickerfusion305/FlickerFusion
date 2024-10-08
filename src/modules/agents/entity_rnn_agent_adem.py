import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer


class EntityAttentionRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EntityAttentionRNNAgent, self).__init__()
        self.args = args

        # input_shape: dimension of the raw input features for each token (or entity)
        # args.attn_embed_dim: dimension of the token
        # FC = FF = MLP
        # each MLP의 width (hidden layer size e.g. 64, 128) and depth (e.g. 1, 2, 3)
        # lambda embedding size: (1) same size as lambda vector, (2) 수동으로 정해줌,
        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        # if args.pooling_type is None:
        self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim*(1+args.double_attn), args)
        # else:
        #     self.attn = EntityPoolingLayer(args.attn_embed_dim,
        #                                    args.attn_embed_dim,
        #                                    args.attn_embed_dim*(1+args.double_attn),
        #                                    args.pooling_type,
        #                                    args)
        # INSERT DOMAIN PROXY CONCAT
        #  Return shape: batch size, # of agents, (embedding dimension + domain_proxy_size)
        self.fc2 = nn.Linear((args.attn_embed_dim)*(1+args.self_loc+args.double_attn), args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # IGNORE msg, this is for COPA
        msg_d = self.args.msg_dim if self.args.use_msg else 0
        self.fc3 = nn.Linear(args.rnn_hidden_dim+msg_d, args.n_actions)
        if args.self_loc:
            self.self_fc = nn.Linear(input_shape, args.attn_embed_dim)
        self.attn_weights=None

    # initialize the hidden states of the agents before the start of a new episode or sequence.
    def init_hidden(self):
        # make hidden states on same device as model
        self.attn_weights=None
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

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
        if self.args.self_loc:
            loc = self.self_fc(entities[:, :self.args.n_agents])
            x2 = th.cat([x2, loc], dim=2)
        x3 = F.relu(self.fc2(x2))
        x3 = x3.reshape(bs, ts, self.args.n_agents, -1)

        h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hs = []
        for t in range(ts):
            curr_x3 = x3[:, t].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(curr_x3, h)
            hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
        hs = th.stack(hs, dim=1)  # Concat over time
        if msg is not None:
            hs = th.cat([hs, msg], dim=3)
        q = self.fc3(hs)
        # zero out output for inactive agents
        q = q.reshape(bs, ts, self.args.n_agents, -1)
        q = q.masked_fill(agent_mask.reshape(bs, ts, self.args.n_agents, 1).bool(), 0)
        # q = q.reshape(bs * self.args.n_agents, -1)

        # Q-values (q): The output q is a tensor of shape (batch_size, time_steps,
        # num_agents, n_actions). This tensor represents the Q-values for all agents
        # at each time step in the batch.

        if ret_attn_logits is not None:
            return q, h, attn_logits.reshape(bs, ts, self.args.n_agents, ne)
        return q, hs


class ImagineEntityAttentionRNNAgent(EntityAttentionRNNAgent):
    def __init__(self, *args, **kwargs):
        super(ImagineEntityAttentionRNNAgent, self).__init__(*args, **kwargs)

    def logical_not(self, inp):
        return 1 - inp

    def logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out

    def entitymask2attnmask(self, entity_mask):
        bs, ts, ne = entity_mask.shape
        # agent_mask = entity_mask[:, :, :self.args.n_agents]
        in1 = (1 - entity_mask.to(th.float)).reshape(bs * ts, ne, 1)
        in2 = (1 - entity_mask.to(th.float)).reshape(bs * ts, 1, ne)
        attn_mask = 1 - th.bmm(in1, in2)
        return attn_mask.reshape(bs, ts, ne, ne).to(th.uint8)

    def forward(self, inputs, hidden_state, imagine=False, **kwargs):
        if not imagine:
            if kwargs.get('ret_attn_weights', False):
                return super(ImagineEntityAttentionRNNAgent, self).forward(inputs, hidden_state, ret_attn_weights=True)
            else:
                return super(ImagineEntityAttentionRNNAgent, self).forward(inputs, hidden_state)
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape

        # create random split of entities (once per episode)
        groupA_probs = th.rand(bs, 1, 1, device=entities.device).repeat(1, 1, ne)

        groupA = th.bernoulli(groupA_probs).to(th.uint8)
        groupB = self.logical_not(groupA)
        # mask out entities not present in env
        groupA = self.logical_or(groupA, entity_mask[:, [0]])
        groupB = self.logical_or(groupB, entity_mask[:, [0]])

        # convert entity mask to attention mask
        groupAattnmask = self.entitymask2attnmask(groupA)
        groupBattnmask = self.entitymask2attnmask(groupB)
        # create attention mask for interactions between groups
        interactattnmask = self.logical_or(self.logical_not(groupAattnmask),
                                           self.logical_not(groupBattnmask))
        # get within group attention mask
        withinattnmask = self.logical_not(interactattnmask)

        activeattnmask = self.entitymask2attnmask(entity_mask[:, [0]])
        # get masks to use for mixer (no obs_mask but mask out unused entities)
        Wattnmask_noobs = self.logical_or(withinattnmask, activeattnmask)
        Iattnmask_noobs = self.logical_or(interactattnmask, activeattnmask)
        # mask out agents that aren't observable (also expands time dim due to shape of obs_mask)
        withinattnmask = self.logical_or(withinattnmask, obs_mask)
        interactattnmask = self.logical_or(interactattnmask, obs_mask)

        entities = entities.repeat(3, 1, 1, 1)
        obs_mask = th.cat([obs_mask, withinattnmask, interactattnmask], dim=0)
        entity_mask = entity_mask.repeat(3, 1, 1)

        inputs = (entities, obs_mask, entity_mask)
        hidden_state = hidden_state.repeat(3, 1, 1)
        q, h = super(ImagineEntityAttentionRNNAgent, self).forward(inputs, hidden_state)
        return q, h, (Wattnmask_noobs.repeat(1, ts, 1, 1), Iattnmask_noobs.repeat(1, ts, 1, 1))
