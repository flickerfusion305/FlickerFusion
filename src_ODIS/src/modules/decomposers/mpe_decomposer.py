# from smac.env.multiagentenv import MultiAgentEnv
# from smac.env.starcraft2.maps import get_map_params

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np

#TODO MPEDecomposer 완성, 저희 환경에 맞게 
class MPEDecomposer:
    def __init__(self, args, env_info):
        # Load map params
        # self.map_name = args.env_args["map_name"]
        # map_params = get_map_params(self.map_name)
        
        self.episode_limit = args.env_args['scenario_config']['episode_limit']

        self.env_info = env_info
        onehot_dict = env_info["onehot_dict"]
        obs_info = env_info["obs_info"]
        entity_maxnum = env_info["entity_maxnum"]
        maxobscnt_dict = env_info["maxobscnt_dict"]
        entitynames = list(onehot_dict.keys())
        entitynums = {name: entity_maxnum[name] for name in entitynames}
        self.totobssz = sum(obs_info.values()) + len(onehot_dict.keys())

        n_entities = sum(entity_maxnum.values())
        self.n_agents = args.n_agents
        self.n_enemies = n_entities - self.n_agents # NOTE in mpe env: n_enemies = number of nonagent entities

        # # Observations and state
        # self.obs_own_health = args.env_args.get("obs_own_health", False)
        # self.obs_all_health = args.env_args.get("obs_all_health", False)
        # self.obs_instead_of_state = args.env_args.get("obs_instead_of_state", False)
        # self.obs_last_action = args.env_args.get("obs_last_action", False)
        # self.obs_pathing_grid = args.env_args.get("obs_pathing_grid", False)
        # self.obs_terrain_height = args.env_args.get("obs_terrain_height", False)
        # self.obs_timestep_number = args.env_args.get("obs_timestep_number", False)
        self.state_last_action = args.env_args.get("state_last_action", False)
        self.state_timestep_number = args.env_args.get("state_timestep_number", False)
        # if self.obs_all_health:
        #     self.obs_own_health = True
        # self.n_obs_pathing = 8
        # self.n_obs_height = 9

        # # Actions
        self.n_actions_no_attack = 0
        # self.n_actions_move = 0
        self.n_actions = args.n_actions

        # # Map info
        # self._agent_race = map_params["a_race"]
        # self._bot_race = map_params["b_race"]
        # self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        # self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        # self.unit_type_bits = map_params["unit_type_bits"]
        # self.map_type = map_params["map_type"]

        # get the shape of obs' components
        # self.move_feats, self.enemy_feats, self.ally_feats, self.own_feats, self.obs_nf_en, self.obs_nf_al = \
        #     self.get_obs_size()
        # self.own_obs_dim = self.move_feats + self.own_feats
        self.obs_dim = env_info["obs_shape"]

        # get the shape of state's components
        # self.enemy_state_dim, self.ally_state_dim, self.last_action_state_dim, self.timestep_number_state_dim, self.state_nf_en, self.state_nf_al = \
        #     self.get_state_size()

        # For state encoder
        self.state_nf_al = self.totobssz
        self.state_nf_en = self.totobssz
        self.state_dim = env_info["state_shape"]
        self.timestep_number_state_dim = None

        # For obs encoder
        self.wrapped_obs_own_dim = self.totobssz + self.n_actions
        self.obs_nf_al = self.totobssz
        self.obs_nf_en = self.totobssz
        self.obs_en_dim = self.totobssz
        self.obs_al_dim = self.totobssz


    def decompose_state(self, state_input):
        """
        NOTE MPE env
        state_input = [ally_state, enemy_state]
        """
        # state_input = [ally_state, enemy_state, last_action_state, timestep_number_state]
        # assume state_input.shape == [batch_size, seq_len, state]
        
        # extract ally_states
        ally_states = [state_input[:, :, i * self.state_nf_al:(i + 1) * self.state_nf_al] for i in range(self.n_agents)]
        # extract enemy_states
        base = self.n_agents * self.state_nf_al
        enemy_states = [state_input[:, :, base + i * self.state_nf_en:base + (i + 1) * self.state_nf_en] for i in range(self.n_enemies)]
        # extract last_action_states
        base += self.n_enemies * self.state_nf_en
        last_action_states = [state_input[:, :, base + i * self.n_actions:base + (i + 1) * self.n_actions] for i in range(self.n_agents)]
        # extract timestep_number_state
        base += self.n_agents * self.n_actions
        # NOTE MPE env
        # timestep_number_state = state_input[:, :, base:base+self.timestep_number_state_dim]   

        return ally_states, enemy_states,# last_action_states , timestep_number_state

    def decompose_obs(self, obs_input):
        """
        obs_input: env_obs
        env_obs = [own_feats, ally_feats, enemy_feats, last_action]
        NOTE in mpe env: obs_input -> own_obs, enemy_obs, ally_obs
        """
        # Fixed for MPE env
        own_obs = obs_input[:,:self.totobssz]
        base = self.totobssz
        ally_feats = [obs_input[:, base + i * self.obs_nf_al:base + (i + 1) * self.obs_nf_al] for i in range(self.n_agents - 1)]
        base += self.obs_nf_en * (self.n_agents - 1)
        enemy_feats = [obs_input[:, base + i * self.obs_nf_en:base + (i + 1) * self.obs_nf_en] for i in range(self.n_enemies)]
        
        return own_obs, enemy_feats, ally_feats

    def decompose_action_info(self, action_info):
        """
        action_info: shape [n_agent, n_action]
        """
        # shape = action_info.shape
        # if len(shape) > 2:
        #     action_info = action_info.reshape(np.prod(shape[:-1]), shape[-1])
        # no_attack_action_info = action_info[:, :self.n_actions_no_attack]
        # attack_action_info = action_info[:, self.n_actions_no_attack:self.n_actions_no_attack + self.n_enemies]
        # # recover shape
        # no_attack_action_info = no_attack_action_info.reshape(*shape[:-1], self.n_actions_no_attack)    
        # attack_action_info = attack_action_info.reshape(*shape[:-1], self.n_enemies)
        # # get compact action
        # bin_attack_info = th.sum(attack_action_info, dim=-1).unsqueeze(-1)
        # compact_action_info = th.cat([no_attack_action_info, bin_attack_info], dim=-1)
        return action_info # For MPE env