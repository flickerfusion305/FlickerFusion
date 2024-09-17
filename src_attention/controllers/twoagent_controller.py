from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
# Two types of agents use distinct two networks, only sharing parameters with same agent type
class TwoAgentMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents1 = args.n_agents1
        self.n_agents2 = args.n_agents2
        self.args = args
        self.scheme = scheme
        input_shape = self._get_input_shape(scheme)
        self.input_shape = input_shape
        self._build_agents(input_shape, scheme=scheme)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, ret_agent_outs=False, ret_attn_weights=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if ret_attn_weights:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, ret_attn_weights=True)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        if ret_agent_outs:
            return chosen_actions, agent_outputs[bs]
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True

        agent_inputs1, agent_inputs2 = self._build_inputs(ep_batch, t)

        if kwargs.get('imagine', False):
            pass
            #agent_outs, self.hidden_states, groups = self.agent(agent_inputs, self.hidden_states, **kwargs)
        elif kwargs.get('ret_attn_weights', False):
            pass
            #agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, ret_attn_weights=True)
        else:
            agent_outs1, self.hidden_states1 = self.agent1(agent_inputs1, self.hidden_states1)
            agent_outs2, self.hidden_states2 = self.agent2(agent_inputs2, self.hidden_states2)

        
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs1[avail_actions1 == 0] = -1e10
                agent_outs2[avail_actions2 == 0] = -1e10


            agent_outs1 = th.nn.functional.softmax(agent_outs1, dim=-1)
            agent_outs2 = th.nn.functional.softmax(agent_outs2, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs1.size(-1)
                epsilon_action_num = agent_outs2.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num1 = avail_actions1.sum(dim=-1, keepdim=True).float()
                    epsilon_action_num2 = avail_actions2.sum(dim=-1, keepdim=True).float()

                agent_outs1 = ((1 - self.action_selector.epsilon) * agent_outs1
                               + th.ones_like(agent_outs1) * self.action_selector.epsilon/epsilon_action_num)
                
                agent_outs2 = ((1 - self.action_selector.epsilon) * agent_outs2
                               + th.ones_like(agent_outs2) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs1[avail_actions1 == 0] = 0.0
                    agent_outs2[avail_actions2 == 0] = 0.0
        agent_outs = th.cat((agent_outs1, agent_outs2), dim=2)
        if int_t:
            return agent_outs.squeeze(1)
        #if kwargs.get('imagine', False):
        #    return agent_outs, groups
        return agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states1 = self.agent1.init_hidden().unsqueeze(0).expand(batch_size, self.args.n_agents1, -1)  # bav
        self.hidden_states2 = self.agent2.init_hidden().unsqueeze(0).expand(batch_size, self.args.n_agents2, -1)  # bav

    def parameters(self):
        return list(self.agent1.parameters()) + list(self.agent2.parameters())

    def load_state(self, other_mac):
        self.agent1.load_state_dict(other_mac.agent1.state_dict())
        self.agent2.load_state_dict(other_mac.agent2.state_dict())

    def cuda(self):
        self.agent1.cuda()
        self.agent2.cuda()

    def eval(self):
        self.agent1.eval()
        self.agent2.eval()

    def train(self):
        self.agent1.train()
        self.agent2.train()

    def save_models(self, path):
        th.save(self.agent1.state_dict(), "{}/agent1.th".format(path))
        th.save(self.agent2.state_dict(), "{}/agent2.th".format(path))

    def load_models(self, path):
        self.agent1.load_state_dict(th.load("{}/agent1.th".format(path), map_location=lambda storage, loc: storage))
        self.agent2.load_state_dict(th.load("{}/agent2.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape, scheme = None):
        if "vae" in self.args.agent:
            self.agent1 = agent_REGISTRY[self.args.agent](input_shape, self.args, scheme=scheme)
            self.agent2 = agent_REGISTRY[self.args.agent](input_shape, self.args)
        else:
            self.agent1 = agent_REGISTRY[self.args.agent](input_shape, self.args)
            self.agent2 = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs, ts, na, os = batch["obs"].shape
        inputs1 = []
        inputs2 = []

        if self.args.obs_type == "QMIX_vanilla" or "QMIX_M":
            if t.start == 0:
                acs = th.zeros_like(batch["actions_onehot"][:, t])
                acs[:, 1:] = batch["actions_onehot"][:, slice(0, t.stop - 1)]
            else:
                acs = batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)]

            inputs1.append(acs[:,:,:self.n_agents1,:])
            inputs2.append(acs[:,:,self.n_agents1:,:])
            inputs1.append(batch["obs"][:, t,:self.n_agents1,:])  # btav
            inputs2.append(batch["obs"][:, t,self.n_agents1:,:]) 
        
        if self.args.obs_agent_id:
            inputs1.append(th.eye(self.n_agents1, device=batch.device).view(1, 1, self.n_agents1, self.n_agents1).expand(bs, t.stop - t.start, -1, -1))
            inputs2.append(th.eye(self.n_agents2, device=batch.device).view(1, 1, self.n_agents2, self.n_agents2).expand(bs, t.stop - t.start, -1, -1))
        inputs1 = th.cat(inputs1, dim=3)
        inputs2 = th.cat(inputs2, dim=3)
        return inputs1, inputs2
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_type == "QMIX_vanilla" or self.args.obs_type == "QMIX_M":
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
