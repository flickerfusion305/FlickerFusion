"""
MPEv2 Note:
Goal landmark is defined at the initializing step. Killing Goal landmark is prohibited
"""

import numpy as np
from envs.mpe_fluid.core import World, Agent, Landmark
from envs.mpe_fluid.scenario import BaseScenario
from envs.mpe_fluid.core import Action
from copy import deepcopy
from collections import defaultdict
import time

### TODO Read config.yml file to setup scenarios, and setup agents add / delete ###
class Scenario(BaseScenario):
    def make_world(self, scenario_config = {}):
        # Default world properties
        num_agents = 5
        num_adversaries = 1
        num_targets = 2
        num_decoys = 3
        episode_limit = 25

        self.boundary = 2 #define boundary
        
        world = World()

        world.max_n_agents = 0
        world.max_n_advs = 0
        world.max_n_targets = 0
        world.max_n_decoys = 0
        world.episode_limit = episode_limit
        world.n_actions = 5

        self.scenario_config = scenario_config
        self.intra_trajectory = None
        if 'num_agents' in scenario_config.keys(): num_agents = scenario_config['num_agents']
        if 'num_adversaries' in scenario_config.keys(): num_adversaries = scenario_config['num_adversaries']
        if 'num_targets' in scenario_config.keys(): num_targets = scenario_config['num_targets']
        if 'num_decoys' in scenario_config.keys(): num_decoys = scenario_config['num_decoys']        
        
        if 'max_n_agents' in scenario_config.keys(): world.max_n_agents = scenario_config['max_n_agents']
        if 'max_n_advs' in scenario_config.keys(): world.max_n_advs = scenario_config['max_n_advs']
        if 'max_n_targets' in scenario_config.keys(): world.max_n_targets = scenario_config['max_n_targets']
        if 'max_n_decoys' in scenario_config.keys(): world.max_n_decoys = scenario_config['max_n_decoys']

        if 'episode_limit' in scenario_config.keys(): world.episode_limit = scenario_config['episode_limit']
        if 'intra_trajectory' in scenario_config.keys(): self.intra_trajectory = scenario_config['intra_trajectory']
        

        world.agent_count = num_agents
        world.adv_count = num_adversaries
        world.target_count = num_targets
        world.decoy_count = num_decoys
        
        # add agents
        world.agents = []

        world.good_agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.good_agents):
            agent.adversary = False
            agent.name = "agent_%d" % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
            # world.max_n_agents += 1 # maybe will be used for tracking current num for further implementation, instead of max num
            world.dead_mask[agent.name] = 0

        world.adversaries = [Agent() for i in range(num_adversaries)]
        for i, agent in enumerate(world.adversaries):
            agent.adversary = True
            agent.name = "adv_%d" % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
            agent.action_callback = self.heuristic_agent
            # world.max_n_advs += 1
            world.dead_mask[agent.name] = 0
        
        world.agents += world.good_agents
        world.agents += world.adversaries

        # add landmarks
        world.landmarks = []

        world.targets = [Landmark() for i in range(num_targets)]
        for i, landmark in enumerate(world.targets):
            landmark.target = True
            landmark.name = "target_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
            # world.max_n_landmarks += 1
            world.dead_mask[landmark.name] = 0

        world.decoys = [Landmark() for i in range(num_decoys)]
        for i, landmark in enumerate(world.decoys):
            landmark.target = False
            landmark.name = "decoy_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
            # world.max_n_landmarks += 1
            world.dead_mask[landmark.name] = 0

        world.landmarks += world.decoys
        world.landmarks += world.targets

        world.init_agents = world.agents[:]
        world.init_landmarks = world.landmarks[:]
        # keep all agents/landmarks including new/dead entities
        world.states_agents = world.agents[:]
        world.states_landmarks = world.landmarks[:]

        # This is for building Neural Network w.r.t. corresponding size 

        # NOTE Define entity type for onehot encoding as #
        # obs_type = agent : 0, adv : 1, target : 2, decoy : 3
        world.onehot_dict = {"agent": 0, "adv": 1, "target": 2, "decoy": 3}
        # NOTE entity observation type : size #
        world.obs_info = {"pos": 2, "vel": 2}
        world.entity_maxnum = {"agent": world.max_n_agents, "adv": world.max_n_advs, "target": world.max_n_targets, "decoy": world.max_n_decoys}
        
        # agent: (4 (onehot) + 4 (obs element) / landmark: 4 (onehot) + 4 (obs element)
        world.state_shape = (world.max_n_agents + world.max_n_advs) * (4+4) + (world.max_n_targets + world.max_n_decoys) * (4+4)
        world.obs_shape = (world.max_n_agents + world.max_n_advs) *  (4+4) + (world.max_n_targets + world.max_n_decoys) * (4+4)
        world.max_entity_size = len(world.onehot_dict.keys()) + sum(world.obs_info.values())
        world.max_n_entities = world.max_n_agents + world.max_n_advs + world.max_n_targets + world.max_n_decoys 
        world.n_entities = len(world.good_agents) + len(world.adversaries) + len(world.targets) + len(world.decoys)
        return world

    def heuristic_agent(self, agent, world): # Needs fix later
        """
        Heuristic adversary agent sets landmark which has smallest sum of distance to agents as a target. Heuristic agent acts in
        """

        action = Action()
        action.u = np.zeros(world.dim_p)
        dis_dict = {}
        for lm in world.landmarks:
            dis_dict[lm.name] = 0
        
        for ag in self.good_agents(world):
            for lm in world.landmarks:
                dis_dict[lm.name] += world.get_distance(ag, lm)
        
        target = min(dis_dict, key=dis_dict.get)

        dx, dy = world.get_relpos_byname(target, agent.name)
        vx, vy = agent.state.p_vel
        damping = 0.25
        dt = 0.1
        max_accel = 1

        # velocity control
        vx1 = vx * (1-damping) # vx after next timestep
        vy1 = vy * (1-damping) # vy after next timestep
        dx1 = dx - vx * dt
        dy1 = dy - vy * dt
        vx1_t = dx1 / dt
        vy1_t = dy1 / dt
        dvx = vx1_t - vx1
        dvy = vy1_t - vy1

        action.u[0] += dvx
        action.u[1] += dvy

        if np.sqrt(dvx**2 + dvy**2) > max_accel:
            action.u *= max_accel / np.sqrt(dvx**2 + dvy**2)

        return action
    
    def reset_world(self, world, np_random):
        world.agents = world.init_agents[:]
        world.landmarks = world.init_landmarks[:]

        world.agent_count = len(self.good_agents(world))
        world.adv_count = len(self.adversaries(world))
        world.target_count = len(self.targets(world))
        world.decoy_count = len(self.decoys(world))

        # world.action_list = []
        world.ts_action = defaultdict(list)
        if self.intra_trajectory: 
            for action_key, action in self.intra_trajectory['delta_entities'].items():
                count = action["count"]
                for i in range(count): 
                    timestep_info = action["timesteps"][i]
                    if self.intra_trajectory["range_or_manual"] == "manual": 
                        manual_ts = timestep_info["manual"]
                        world.ts_action[manual_ts].append(action_key) 
                    else: 
                        rand_ts = np_random.randint(timestep_info["range"][0], timestep_info["range"][1])
                        world.ts_action[rand_ts].append(action_key)
                
        # random properties for agents
        for i, agent in enumerate(self.good_agents(world)):
            agent.color = np.array([0.35, 0.35, 0.85])
        for i, landmark in enumerate(self.adversaries(world)):
            landmark.color = np.array([0.85, 0.35, 0.35])
        # random properties for landmarks
        for i, landmark in enumerate(self.targets(world)):
            landmark.color = np.array([0.15, 0.65, 0.15])
        for i, landmark in enumerate(self.decoys(world)):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-self.boundary, +self.boundary, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-self.boundary, +self.boundary, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        world.index_map = {entity.name: idx for idx, entity in enumerate(world.entities)}
        world.calc_distmat()
        world.time_step = 0

    def update_que(self, world ,np_random):
        # This method will be called right after world.step, you can fill world.add_agent_que or world.del_agent_que to add delete agent
        # You can get information of current world / environment through world / simpleenv args.

        while world.time_step in world.ts_action and len(world.ts_action[world.time_step]) > 0: 
            action = world.ts_action[world.time_step].pop(0)
            if action == 'add_agents':
                self.add_new_agent(world, np_random)
            elif action == 'add_advs': 
                self.add_new_adv(world, np_random)
            elif action == 'add_targets':
                self.add_new_target(world, np_random)
            elif action == 'add_decoys':
                self.add_new_decoy(world, np_random)
            elif action == 'del_agents': 
                self.kill_random_agent(world, np_random)
            elif action == 'del_advs':
                self.kill_random_adv(world, np_random)
            elif action == 'del_targets':
                self.kill_random_target(world, np_random)
            elif action == 'del_decoys':
                self.kill_random_decoy(world, np_random)
        
        #demo code
        #if 'decreasing' in scenario_config.keys(): decreasing= scenario_config['decreasing']
            # if world.time_step in decreasing.values():
            #     for i in range(3):
        # if len(self.scenario_config)!=0:
        #     decreasing={}
        #     increasing={}
        #     increasing_decreasing={}
        #     if 'adversary' in self.scenario_config and 'intra_trajectory' in self.scenario_config['SAR'] and 'decreasing' in self.scenario_config['SAR']['intra_trajectory']:
        #         decreasing= self.scenario_config['SAR']['intra_trajectory']['decreasing']
        #         for entry in decreasing:
        #             if entry.get('t_step') == world.time_step: #진짜 demo라서 돌아가게만 만듬. 나중에 최적화 시켜야함.
        #                 for i in range(-1*entry['ally_n']):
        #                     self.kill_agent(world,agent)
        #                     self.kill_landmark(world,agent) 
        #                 for i in range(-1*entry['adv_n']):
        #                     self.kill_agent(world,agent)
                
        #     if 'adversary' in self.scenario_config and 'intra_trajectory' in self.scenario_config['SAR'] and 'increasing' in self.scenario_config['SAR']['intra_trajectory']:
        #         increasing= self.scenario_config['SAR']['intra_trajectory']['increasing']
        #         for entry in increasing:
        #             if entry.get('t_step') == world.time_step: #진짜 demo라서 돌아가게만 만듬. 나중에 최적화 시켜야함.
        #                 for i in range(entry['ally_n']):
        #                     self.add_new_agent(world, np_random)
        #                     self.add_new_landmark(world, np_random)
        #                 for i in range(entry['adv_n']):
        #                     self.add_new_agent(world, np_random)
    
    def done(self, world):
        if world.time_step >= world.episode_limit: return True
        return False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    # return all adversarial agents
    def targets(self, world):
        return [lm for lm in world.landmarks if lm.target]
    
    # return all adversarial agents
    def decoys(self, world):
        return [lm for lm in world.landmarks if not lm.target]

    def reward(self, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        rew = 0
        normalized_agent_values = 0
        sum_min_agent_dists = 0
        sum_min_adv_dists = 0
        normalized_adv_values = 0
        for target in self.targets(world):
            possible_max_dist = self.possible_max_dist(target)
            agent_dist = []
            adv_dist = []
            for agent in world.agents:
                # distance between adv and target
                if agent.adversary:
                    adv_dist.append(world.get_distance(agent, target)) # - target.size - agent.size
                # distance between agent and target    
                else:
                    agent_dist.append(world.get_distance(agent, target)) # - target.size - agent.size
            # Calculate positive reward for agents
            closest_agent_dist = min(agent_dist)
            sum_min_agent_dists += closest_agent_dist
            # normalized_agent_dist = (1 - closest_agent_dist / possible_max_dist) * 50
            # normalized_agent_values += (max(0, min(50, normalized_agent_dist)))
            # Calculate negative reward for agents
            closest_adv_dist = min(adv_dist)
            sum_min_adv_dists += closest_adv_dist
            # normalized_adv_dist = (1 - closest_adv_dist / possible_max_dist) * 50
            # normalized_adv_values += (max(0, min(50, normalized_adv_dist)))
        rew -= sum_min_agent_dists 
        rew += sum_min_adv_dists
        return rew
    
    def possible_max_dist(self, target):
        # x, y boundary: [-1, 1]
        if target.state.p_pos[0] >= 0 and target.state.p_pos[1] >= 0:
            delta_pos = target.state.p_pos - np.array([-self.boundary, -self.boundary])
            dist = np.sqrt(np.sum(np.square(delta_pos))) - target.size
            return dist
        elif target.state.p_pos[0] < 0 and target.state.p_pos[1] > 0:
            delta_pos = target.state.p_pos - np.array([self.boundary, -self.boundary])
            dist = np.sqrt(np.sum(np.square(delta_pos))) - target.size
            return dist
        elif target.state.p_pos[0] < 0 and target.state.p_pos[1] < 0:
            delta_pos = target.state.p_pos - np.array([self.boundary, self.boundary])
            dist = np.sqrt(np.sum(np.square(delta_pos))) - target.size
            return dist
        elif target.state.p_pos[0] > 0 and target.state.p_pos[1] < 0:
            delta_pos = target.state.p_pos - np.array([-self.boundary, self.boundary])
            dist = np.sqrt(np.sum(np.square(delta_pos))) - target.size
            return dist

    def get_obs(self, world): # return all agent obs Naively

        return

    def get_entity(self, world): # return all entities for CAMA / REFIL TODO
        
        return

    def get_mask(self, world): # get observation mask for CAMA / REFIL TODO
        obs_mask = np.zeros((world.max_n_entities, world.max_n_entities), dtype=np.uint8) # all observable
        entity_mask = np.ones((world.max_n_entities), dtype=np.uint8)

        agent_names = [agent.name for agent in world.states_agents]
        landmark_names = [lm.name for lm in world.states_landmarks]

        cnt = 0
        # agent
        for i in range(world.max_n_agents):
            if f"agent_{i}" in agent_names:
                if world.dead_mask[f"agent_{i}"]: # created, dead
                    obs_mask[cnt,:] = 1
                    obs_mask[:,cnt] = 1
                    # print("abesent from env: ", cnt) #for debug
                else: # created, alive
                    entity_mask[cnt] = 0
                cnt += 1
            else: # not created yet
                obs_mask[cnt,:] = 1
                obs_mask[:,cnt] = 1
                # print("abesent from env: ", cnt) #for debug
                cnt += 1
        # adv
        for i in range(world.max_n_advs):
            if f"adv_{i}" in agent_names:
                if world.dead_mask[f"adv_{i}"]: # created, dead
                    obs_mask[cnt,:] = 1
                    obs_mask[:,cnt] = 1
                    # print("abesent from env: ", cnt) #for debug
                cnt += 1
            else: # not created yet
                obs_mask[cnt,:] = 1
                obs_mask[:,cnt] = 1
                # print("abesent from env: ", cnt) #for debug
                cnt += 1
        # target
        for i in range(world.max_n_targets):
            if f"target_{i}" in landmark_names:
                if world.dead_mask[f"target_{i}"]: # created, dead
                    obs_mask[cnt,:] = 1
                    obs_mask[:,cnt] = 1
                    # print("abesent from env: ", cnt) #for debug
                cnt += 1
            else: # not created yet
                obs_mask[cnt,:] = 1
                obs_mask[:,cnt] = 1
                #print("abesent from env: ", cnt) #for debug
                cnt += 1
        # decoy
        for i in range(world.max_n_decoys):
            if f"decoy_{i}" in landmark_names:
                if world.dead_mask[f"decoy_{i}"]: # created, dead
                    obs_mask[cnt,:] = 1
                    obs_mask[:,cnt] = 1
                    # print("abesent from env: ", cnt) #for debug
                cnt += 1
            else: # not created yet
                obs_mask[cnt,:] = 1
                obs_mask[:,cnt] = 1
                #print("abesent from env: ", cnt) #for debug
                cnt += 1
        
        return obs_mask, entity_mask

    def get_state(self, world): # get global state
        # NOTE Define entity observation type for onehot encoding as #
        # obs_type = agent_pos : 0, agent_vel : 1, adv_pos : 2, adv_vel : 3, target_pos = 4, decoy_pos = 5
        # for each global state element, [obs_type, obs_element]
        
        states = {}
        for agent in world.states_agents:
            agenttype = agent.name.split('_')[0]
            agentidx = agent.name.split('_')[1]
            states[f"{agenttype}-pos_{agentidx}"] = {
                "element" : agent.state.p_pos.tolist() if world.dead_mask[agent.name] == 0 else np.zeros(2).tolist(),
                "mask" : world.dead_mask[agent.name]
            }
            states[f"{agenttype}-vel_{agentidx}"] = {
                "element" : agent.state.p_vel.tolist() if world.dead_mask[agent.name] == 0 else np.zeros(2).tolist(),
                "mask" : world.dead_mask[agent.name]
            }
        for landmark in world.states_landmarks:
            lmtype = landmark.name.split('_')[0]
            lmidx = landmark.name.split('_')[1]
            states[f"{lmtype}-pos_{lmidx}"] = {
                "element" : landmark.state.p_pos.tolist() if world.dead_mask[landmark.name] == 0 else np.zeros(2).tolist(),
                "mask" : world.dead_mask[landmark.name]
            }
        return states
    
    def spawn_position(self, mode, np_random, **kwargs):
        """
        mode: 'deterministic', 'uniform', 'gaussian'
        deterministic -> x: pos_x, y: pos_y
        uniform -> xlim: [minimum pos_x, maximum pos_x] ylim: [minimum pos_y, maximum pos_y]
        gaussian -> x: pos_x (mu), y: pos_y (mu), std: standard deviation
        """
        if mode == 'deterministic':
            return np.array([kwargs['x'],kwargs['y']])
        elif mode == 'uniform':
            return np_random.uniform(np.array([kwargs['xlim'][0], kwargs['ylim'][0]]), np.array([kwargs['xlim'][1], kwargs['ylim'][1]]))
        elif mode == 'gaussian':
            return np_random.normal(np.array([kwargs['x'], kwargs['y']]), kwargs['std'])

    def add_new_agent(self,world,np_random):
        agent = Agent()
        agent.adversary = False
        agent.name = f"agent_{world.agent_count}"  # naming agent based on created agent num
        agent.collide = False
        agent.silent = True
        agent.size = 0.05
        world.dead_mask[agent.name] = 0
        world.agent_count += 1

        agent.color = np.array([0.35, 0.35, 0.85])
        # agent.state.p_pos = np_random.uniform(-self.boundary, +self.boundary, world.dim_p)
        agent.state.p_pos = self.set_pos_random(agent, self.boundary*0.98, np_random)
        agent.state.p_vel = np.zeros(world.dim_p)

        world.new_agent_que.append(agent)

    def add_new_adv(self,world,np_random):
        agent = Agent()
        agent.adversary = True
        agent.name = f"adv_{world.adv_count}"  # naming agent based on created agent num
        agent.collide = False
        agent.silent = True
        agent.size = 0.05
        agent.action_callback = self.heuristic_agent
        world.dead_mask[agent.name] = 0
        world.adv_count += 1

        agent.color = np.array([0.85, 0.35, 0.35])
        # agent.state.p_pos = np_random.uniform(-self.boundary, +self.boundary, world.dim_p)
        agent.state.p_pos = self.set_pos_random(agent, self.boundary*0.98, np_random)
        agent.state.p_vel = np.zeros(world.dim_p)

        world.new_agent_que.append(agent)
    
    def add_new_target(self,world,np_random):
        landmark = Landmark()
        landmark.target = True
        landmark.name = f"target_{world.target_count}"
        landmark.collide = False
        landmark.movable = False
        landmark.size = 0.05
        world.dead_mask[landmark.name] = 0
        world.target_count += 1

        landmark.color = np.array([0.15, 0.65, 0.15])
        landmark.state.p_pos = np_random.uniform(-self.boundary, +self.boundary, world.dim_p)
        landmark.state.p_vel = np.zeros(world.dim_p)

        world.new_landmark_que.append(landmark)

    def add_new_decoy(self,world,np_random):
        landmark = Landmark()
        landmark.target = False
        landmark.name = f"decoy_{world.decoy_count}"
        landmark.collide = False
        landmark.movable = False
        landmark.size = 0.05
        world.dead_mask[landmark.name] = 0
        world.decoy_count += 1

        landmark.color = np.array([0.15, 0.15, 0.15])
        landmark.state.p_pos = np_random.uniform(-self.boundary, +self.boundary, world.dim_p)
        landmark.state.p_vel = np.zeros(world.dim_p)
        
        world.new_landmark_que.append(landmark)

    def kill_random_agent(self, world, np_random): 
        rand_agent = np_random.choice(self.good_agents(world))
        self.kill_agent(world, rand_agent)

    def kill_random_adv(self, world, np_random): 
        rand_agent = np_random.choice(self.adversaries(world))
        self.kill_agent(world, rand_agent)

    def kill_random_target(self, world, np_random):
        rand_landmark = np_random.choice(self.targets(world))
        self.kill_landmark(world, rand_landmark)

    def kill_random_decoy(self, world, np_random):
        rand_landmark = np_random.choice(self.decoys(world))
        self.kill_landmark(world, rand_landmark)

    def kill_agent(self, world, agent):
        world.del_agent_que.add(agent.name)
        world.dead_mask[agent.name] = 1
    
    def kill_landmark(self, world, landmark):
        world.del_landmark_que.add(landmark.name)
        world.dead_mask[landmark.name] = 1

    def set_pos_random(self, entity, boundary, np_random): 
        edge = np_random.randint(2)
        if edge == 0:  # d
            x = np_random.uniform(-boundary, boundary)
            y = -boundary
        else:  # r
            x = boundary
            y = np_random.uniform(-boundary, boundary)
        
        # entity.state.p_pos = np.array([x, y])
        return np.array([x, y])
