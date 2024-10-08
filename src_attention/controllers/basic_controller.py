from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents  # n of agents set in the environment config yml
        self.args = args
        self.scheme = scheme # REFIL, CAMA, QMIX, QMIX 2-agent-type
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
            agent_outputs, attn_weights = self.forward(ep_batch, t_ep, test_mode=test_mode, ret_attn_weights=True)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        if ret_agent_outs:
            return chosen_actions, agent_outputs[bs]
        elif ret_attn_weights:
            return chosen_actions, attn_weights[bs]
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if kwargs.get('imagine', False):
            agent_outs, self.hidden_states, groups = self.agent(agent_inputs, self.hidden_states, **kwargs)
        elif kwargs.get('ret_attn_weights', False):
            agent_outs, self.hidden_states, attn_weights = self.agent(agent_inputs, self.hidden_states, ret_attn_weights=True)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs[avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = avail_actions.sum(dim=-1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[avail_actions == 0] = 0.0
        if int_t:
            if kwargs.get('ret_attn_weights', False):
                 return agent_outs.squeeze(1), attn_weights
            else:
                return agent_outs.squeeze(1)
        if kwargs.get('imagine', False):
            return agent_outs, groups
        return agent_outs

    # initialize the hidden states of the agents before the start of a new episode or sequence.
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def eval(self):
        self.agent.eval()

    def train(self):
        self.agent.train()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape, scheme = None):
        if "vae" in self.args.agent:
            self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args, scheme=scheme)
        else:
            self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs, ts, na, os = batch["obs"].shape
        inputs = []

        if self.args.obs_type == "QMIX_vanilla" or "QMIX_M":
            if t.start == 0:
                acs = th.zeros_like(batch["actions_onehot"][:, t])
                acs[:, 1:] = batch["actions_onehot"][:, slice(0, t.stop - 1)]
            else:
                acs = batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)]
            inputs.append(acs)
            inputs.append(batch["obs"][:, t])  # btav
        
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).view(1, 1, self.n_agents, self.n_agents).expand(bs, t.stop - t.start, -1, -1))
        inputs = th.cat(inputs, dim=3)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_type == "QMIX_vanilla" or self.args.obs_type == "QMIX_M":
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
