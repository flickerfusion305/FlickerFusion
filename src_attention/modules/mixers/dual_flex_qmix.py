import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer

class DuelingAttentionHyperNet(nn.Module):
    """
    A flexible hypernetwork using attention mechanisms, enhanced with a value-advantage
    decomposition (dueling structure) for improved expressiveness.
    """
    def __init__(self, args, extra_dims=0, mode='matrix'):
        super(DuelingAttentionHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.extra_dims = extra_dims
        self.entity_dim = args.entity_shape
        if self.args.entity_last_action:
            self.entity_dim += args.n_actions
        if extra_dims > 0:
            self.entity_dim += extra_dims

        hypernet_embed = args.hypernet_embed
        self.fc1 = nn.Linear(self.entity_dim, hypernet_embed)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(hypernet_embed,
                                             hypernet_embed,
                                             hypernet_embed, args)
        else:
            self.attn = EntityPoolingLayer(hypernet_embed,
                                           hypernet_embed,
                                           hypernet_embed,
                                           args.pooling_type,
                                           args)
        self.value_fc = nn.Linear(hypernet_embed, 1)
        self.advantage_fc = nn.Linear(hypernet_embed, args.mixing_embed_dim)

    def forward(self, entities, entity_mask, attn_mask=None):
        x1 = F.relu(self.fc1(entities))
        agent_mask = entity_mask[:, :self.args.n_agents]
        if attn_mask is None:
            # create attn_mask from entity mask
            attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                                   (1 - entity_mask.to(th.float)).unsqueeze(1))
        x2 = self.attn(x1, pre_mask=attn_mask.to(th.uint8),
                       post_mask=agent_mask)
        # Value and advantage streams
        value = self.value_fc(x2)
        advantage = self.advantage_fc(x2)
        advantage = advantage.masked_fill(agent_mask.unsqueeze(2).bool(), 0) #[bs, na, edim]

        if self.mode == 'vector':
            return value.mean(dim=1), advantage.mean(dim=1)
        elif self.mode == 'alt_vector':
            return value.mean(dim=2), advantage.mean(dim=2)
        elif self.mode == 'scalar':
            return value.mean(dim=(1, 2)), advantage.mean(dim=(1, 2))
        return value, advantage

class DualFlexQMixer(nn.Module):
    def __init__(self, args):
        super(DualFlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim

        # Using the dueling attention hypernet for value and advantage
        self.hyper_v = DuelingAttentionHyperNet(args, mode='scalar')
        self.hyper_adv_1 = DuelingAttentionHyperNet(args, mode='matrix')
        self.hyper_adv_final = DuelingAttentionHyperNet(args, mode='vector')

        self.non_lin = F.elu
        if getattr(self.args, "mixer_non_lin", "elu") == "tanh":
            self.non_lin = F.tanh

    def forward(self, agent_qs, inputs, imagine_groups=None):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.reshape(-1, 1, self.n_agents * 2)
            Wmask, Imask = imagine_groups
            value_W, adv_W = self.hyper_adv_1(entities, entity_mask, attn_mask=Wmask.reshape(bs * max_t, ne, ne))
            value_I, adv_I = self.hyper_adv_1(entities, entity_mask, attn_mask=Imask.reshape(bs * max_t, ne, ne))
            adv = th.cat([adv_W, adv_I], dim=1)
        else:
            agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
            value, adv = self.hyper_adv_1(entities, entity_mask)
        
        adv = adv.view(bs * max_t, -1, self.embed_dim)
        if self.args.softmax_mixing_weights:
            adv = F.softmax(adv, dim=-1)
        else:
            adv = th.abs(adv)

        # First layer processing with dueling structure
        hidden = self.non_lin(th.bmm(agent_qs, adv))

        # Second layer with final mixing
        if self.args.softmax_mixing_weights:
            adv_final = F.softmax(self.hyper_adv_final(entities, entity_mask)[1], dim=-1)
        else:
            adv_final = th.abs(self.hyper_adv_final(entities, entity_mask)[1])
        adv_final = adv_final.view(-1, self.embed_dim, 1)

        # Value stream
        value = self.hyper_v(entities, entity_mask)[0].view(-1, 1, 1)

        # Compute final output
        y = th.bmm(hidden, adv_final) + value
        q_tot = y.view(bs, -1, 1)
        return q_tot
