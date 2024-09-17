import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
from .scenarios import load as sload
from envs.multiagentenv import MultiAgentEnv
import time

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentFluidEnv(gym.Env, MultiAgentEnv):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, 
                scenario_id = "simple_adversary.py",
                seed=None,
                entity_scheme=False,
                shared_viewer=True,
                scenario_config=None,
                **kwargs):
        np.random.seed(seed)
        self.entity_scheme = entity_scheme

        self.scenario_id = scenario_id

        scenario = sload(scenario_id).Scenario()

        self.scenario = scenario
        self.world = scenario.make_world(scenario_config)
        self.policy_agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(self.world.policy_agents)
        # scenario callbacks
        self.reset_callback = scenario.reset_world
        self.reward_callback = scenario.reward
        self.info_callback = None
        self.done_callback = scenario.done
        self.entity_callback = scenario.get_entity
        self.mask_callback = scenario.get_mask
        self.observation_callback = scenario.get_obs
        self.state_callback = scenario.get_state
        self.additional_callback = getattr(scenario, 'additional_reports', None)
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = self.world.discrete_action if hasattr(self.world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.time = 0
        self.pre_transition_data = {}
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.episode_limit = self.world.episode_limit        
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n, pre_transition_data={}): #add scenario_config
        action_n = [int(a) for a in action_n]
        self.policy_agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.policy_agents):
            try: 
                self._set_action(action_n[i], agent, self.action_space[i])
            except: 
                print(f"action_n: {action_n} \n i: {i} \n action_space: {self.action_space} \n policy_agent num: {len(self.policy_agents)}")
        # advance world state
        self.world.step()
        if hasattr(self.scenario, 't_env'): 
            self.scenario.t_env += 1
        self.scenario.update_que(self.world, np_random = np.random) #add scenario_config
        reward = self._get_reward()
        info = {}
        self.update_landmarks()
        self.update_agents()
        done = self._get_done()
        self.pre_transition_data = pre_transition_data
        #self.render()
        additional = self._get_additionals()
        return reward, done, additional, info

    def update_landmarks(self):
        update_landmarks = []
        for landmark in self.world.landmarks:
            if landmark.name in self.world.del_landmark_que:
                pass
            else:
                update_landmarks.append(landmark)
        
        for landmark in self.world.new_landmark_que:
            update_landmarks.append(landmark)
            if hasattr(self.world, 'states_landmarks') and self.world.states_landmarks:
                self.world.states_landmarks.append(landmark)
        
        self.world.landmarks = update_landmarks
        self.landmarks = [landmark.name for landmark in self.world.landmarks]
        
        self.world.new_landmark_que.clear()
        self.world.del_landmark_que.clear()

    def update_agents(self): # Update agents according to self.world.del_agent_que, self.world.new_agent_que 
        update_agents = []
        #update agents list
        for agent in self.world.agents:
            if agent.name in self.world.del_agent_que:
                pass
            else:
                update_agents.append(agent)
        
        for agent in self.world.new_agent_que:
            update_agents.append(agent)
            if hasattr(self.world, 'states_agents') and self.world.states_agents:
                self.world.states_agents.append(agent)

        ## update world values / simpleenv values
        self.world.agents = update_agents
        self.policy_agents = self.world.policy_agents

        self.world.index_map = {entity.name: idx for idx, entity in enumerate(self.world.entities)} # this index_map is for calc_distmat
 
        for agent in self.world.new_agent_que:
            if agent.action_callback is None: 
                total_action_space = []
                # physical action space
                if self.discrete_action_space:
                    u_action_space = spaces.Discrete(self.world.dim_p * 2 + 2)
                else:
                    u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(self.world.dim_p,), dtype=np.float32)
                if agent.movable:
                    total_action_space.append(u_action_space)
                if len(total_action_space) > 1:
                    # all action spaces are discrete, so simplify to MultiDiscrete action space
                    if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                        act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                    else:
                        act_space = spaces.Tuple(total_action_space)
                    self.action_space.append(act_space)
                else:
                    self.action_space.append(total_action_space[0])
                # observation space
                self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.world.max_entity_size,), dtype=np.float32))

        self.world.calc_distmat()
        # clear queues
        self.world.new_agent_que.clear()
        self.world.del_agent_que.clear()
        
    def reset(self, constrain_num=None, test=False, index=None, domain_n = 0):
        # reset world
        self.reset_callback(self.world, np_random = np.random , test=test, domain_n=domain_n)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        self.policy_agents = self.world.policy_agents
        self.reset_action_space()
        self.n = len(self.world.policy_agents)
        return self.get_entities(), self.get_masks()
    
    def get_entities(self):
        if self.entity_callback is None:
            return None
        return self.entity_callback(self.world)
    
    def get_obs(self):
        if self.observation_callback is None:
            return None
        return self.observation_callback(self.world)

    def get_state(self):
        if self.state_callback is None:
            return None
        return self.state_callback(self.world)

    def get_masks(self):
        if self.mask_callback is None:
            return None
        return self.mask_callback(self.world)
    
    def get_env_info(self, args):
        env_info = {"entity_shape": self.world.max_entity_size,
                    "n_actions": self.world.n_actions,
                    # "agent_types": self.world.agent_types, # in case of two types of agent
                    "n_agents": self.world.max_n_agents,
                    "n_entities": self.world.max_n_entities,
                    "episode_limit": self.world.episode_limit,
                    "state_shape":self.world.state_shape,
                    "obs_shape":self.world.obs_shape,
                    "onehot_dict": self.world.onehot_dict,
                    "obs_info": self.world.obs_info,
                    "entity_maxnum": self.world.entity_maxnum,
                    "dropout_num": self.world.DOnum,
                    "maxobscnt_dict": self.world.maxobscnt_dict
                    }

        return env_info
    
    def get_avail_actions(self, n_agent):
        return [[1,1,1,1,1] for _ in range(n_agent)]

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        if self.done_callback is None:
            return False
        return self.done_callback(self.world)

    # get reward for a particular agent
    def _get_reward(self):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(self.world)
    
    # get additional report from MPEV2 environment
    def _get_additionals(self):
        if self.additional_callback is None:
            return 0
        return self.additional_callback(self.world)

    def reset_action_space(self):
        for agent in self.policy_agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.world.dim_p * 2 + 2)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(self.world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            #if self.discrete_action_space:
            #    c_action_space = spaces.Discrete(self.world.dim_c)
            #else:
            #    c_action_space = spaces.Box(low=0.0, high=1.0, shape=(self.world.dim_c,), dtype=np.float32)
            #if not agent.silent:
            #    total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.world.max_entity_size,), dtype=np.float32))
            #agent.action.c = np.zeros(self.world.dim_c)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        #agent.action.c = np.zeros(self.world.dim_c)
        # process action
        # if isinstance(action_space, MultiDiscrete):
        #     act = []
        #     size = action_space.high - action_space.low + 1
        #     index = 0
        #     for s in size:
        #         act.append(action[index:(index+s)])
        #         index += s
        #     action = act
        # else:
        action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            else:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                agent.action.u[0] += action[0][2] - action[0][1]
                agent.action.u[1] += action[0][4] - action[0][3]

            sensitivity = 1.5
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        #if not agent.silent:
        #    # communication action
        #    if self.discrete_action_input:
        #        agent.action.c = np.zeros(self.world.dim_c)
        #        agent.action.c[action[0]] = 1.0
        #    else:
        #        agent.action.c = action[0]
        #    action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human', srk=[1,2,4], sr_color=["#2CAFAC","#FB5607", "#1982C4"]):
        entity_list = []
        for entity_type in self.world.onehot_dict: 
            for index in range(self.world.entity_maxnum[entity_type]):
                entity_list.append(f"{entity_type}_{index}")

        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.p_pos == 0): #not c
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.p_pos)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from envs.mpe_fluid import rendering
                self.viewers[i] = rendering.Viewer(450,450)

        visible_entities = self.world.entities
        if self.pre_transition_data:
            if 'obs_mask' in self.pre_transition_data:
                # print(self.pre_transition_data['obs_mask'][0])
                entity_names = [entity_name for idx, entity_name in enumerate(entity_list) if not self.pre_transition_data['obs_mask'][0][0][idx]]
                # print(entity_names)
                visible_entities = [self.world.entities[self.world.index_map[entity_name]] for entity_name in entity_names]
            elif 'obs' in self.pre_transition_data: 
                # print(self.pre_transition_data['obs'][0][0])
                entity_names = [entity_name for idx, entity_name in enumerate(entity_list) if self.pre_transition_data['obs'][0][0][idx]]
                # print(entity_names)
                visible_entities = [self.world.entities[self.world.index_map[entity_name]] for entity_name in entity_names]

        # create rendering geometry
        # if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
        from envs.mpe_fluid import rendering
        self.render_geoms = []
        self.render_geoms_xform = []
        self.children_geoms = []
        self.children_geoms_xform = []
        for entity in visible_entities:
            geom = rendering.make_circle(entity.size)
            xform = rendering.Transform()
            if 'agent' in entity.name:
                geom.set_color(*entity.color)
            else:
                geom.set_color(*entity.color)
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)
            ##### for rendering AAD
            if hasattr(entity, 'attack_range') and entity.attack_range is not None:
                dashed_area, dashed_xform = self.create_dashed_area(20, entity.size, entity.attack_range)
                dashed_area.set_color(*entity.color, alpha=0.5)
                self.children_geoms.append(dashed_area)
                self.children_geoms_xform.append(dashed_xform)
            #####

        #### for rendering WPP
        # for entity in self.world.landmarks:
        #     if "truck" in entity.name and hasattr(self.world, 'trucks'):
        boundary, boundary_xform = self.draw_boundary()
        boundary.set_color(*np.array([0, 0, 0]), alpha=0.5)
        self.render_geoms.append(boundary)
        self.render_geoms_xform.append(boundary_xform)
                # break

        # add geoms to viewer
        for viewer in self.viewers:
            # FOR DEBUG
            #viewer.wait_for_input() 
            viewer.geoms = []
            for geom in self.render_geoms:
                viewer.add_geom(geom)
            for geom in self.children_geoms:
                viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from envs.mpe_fluid import rendering

            boundary = self.world.boundary+0.01
            lbd, rbd, ubd, dbd = -boundary,boundary,boundary,-boundary
            for entity in self.world.entities:
                lbd = min(lbd, entity.state.p_pos[0])
                rbd = max(rbd,entity.state.p_pos[0])
                ubd = max(ubd,entity.state.p_pos[1])
                dbd = min(dbd,entity.state.p_pos[1])
            
            self.viewers[i].set_bounds(lbd,rbd,dbd,ubd)
            child_idx = 0
            # update geometry positions
            for e, entity in enumerate(visible_entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                if hasattr(entity, 'attack_range') and entity.attack_range is not None:
                    self.children_geoms_xform[child_idx].set_translation(*entity.state.p_pos)
                    child_idx +=1 
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results
    
    def save_render(self, gif_output_name=""):
        for i in range(len(self.viewers)):
            from envs.mpe_fluid import rendering
            if gif_output_name: 
                    self.viewers[i].save_gif(gif_output_name, fps=50)

    def create_dashed_line(self, start_pos, end_pos, color=(0,0,0), width=1, dash_length=0.01):
        from envs.mpe_fluid import rendering
        import numpy as np

        start = np.array(start_pos)
        end = np.array(end_pos)
        length = np.linalg.norm(end - start)
        dash_amount = int(length / dash_length)

        dash_knots = np.array([np.linspace(start[i], end[i], dash_amount) for i in range(2)]).transpose()

        lines = []
        for n in range(0, dash_amount - 1, 2):
            line = rendering.Line(tuple(dash_knots[n]), tuple(dash_knots[n+1]))
            line.add_attr(rendering.LineWidth(width))
            lines.append(line)

        geom = rendering.Compound(lines)
        xform = rendering.Transform()
        geom.add_attr(xform)
        return geom, xform

    def create_dashed_area(self, num_line, agent_size, attack_area, color=(0,0,0), width=1):
        from envs.mpe_fluid import rendering
        import math

        lines = []
        for n in range(num_line):
            angle = 2 * math.pi * n / num_line
            x0 = agent_size * math.cos(angle)
            y0 = agent_size * math.sin(angle)
            x1 = attack_area * math.cos(angle)
            y1 = attack_area * math.sin(angle)
            
            dashed_line, _ = self.create_dashed_line((x0, y0), (x1, y1), color, width)
            lines.extend(dashed_line.gs)

        geom = rendering.Compound(lines)
        xform = rendering.Transform()
        geom.add_attr(xform)
        return geom, xform
    
    def create_dashed_circle(self, radius, dash_length=0.04):
        from envs.mpe_fluid import rendering
        import math
        lines = []
        num_segments = int(2 * math.pi * radius / dash_length)
        for i in range(0, num_segments, 2):
            angle1 = i * (2 * math.pi) / num_segments
            angle2 = ((i + 1) % num_segments) * (2 * math.pi) / num_segments
            x1 = radius * math.cos(angle1)
            y1 = radius * math.sin(angle1)
            x2 = radius * math.cos(angle2)
            y2 = radius * math.sin(angle2)
            line = rendering.Line((x1, y1), (x2, y2))
            lines.append(line)
        geom = rendering.Compound(lines)
        xform = rendering.Transform()
        geom.add_attr(xform)
        return geom, xform
        
    def draw_boundary(self):
        from envs.mpe_fluid import rendering
        lines = []
        boundary = self.world.boundary
        neg_x, pos_x, pos_y, neg_y = -boundary,boundary,boundary,-boundary

        # Solid lines
        solid_line1 = rendering.Line((pos_x, pos_y), (neg_x, pos_y))
        solid_line2 = rendering.Line((neg_x, pos_y), (neg_x, neg_y))
        solid_line3 = rendering.Line((neg_x, neg_y), (pos_x, neg_y))
        solid_line4 = rendering.Line((pos_x, pos_y), (pos_x, neg_y))
        lines.extend([solid_line1, solid_line2, solid_line3, solid_line4])

        geom = rendering.Compound(lines)
        xform = rendering.Transform()
        geom.add_attr(xform)
        return geom, xform

    # create receptor field locations in local coordinate frame
    
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx
    
    def close(self):
        print("Env closed.")
    
    def save_replay(self):
        print("Saving replay function not implemented.")

