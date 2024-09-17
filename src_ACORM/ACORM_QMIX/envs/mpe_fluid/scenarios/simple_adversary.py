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

class Scenario(BaseScenario):
    def make_world(self, scenario_config = {}):
        # Default world properties
        num_agents = 5
        num_adversaries = 1
        num_targets = 2
        num_decoys = 3
        episode_limit = 25
        
        world = World()
        world.boundary = 1.3 #define boundary

        world.max_n_agents = 0
        world.max_n_advs = 0
        world.max_n_targets = 0
        world.max_n_decoys = 0
        world.episode_limit = episode_limit
        world.n_actions = 5
        
        world.DOnum = None
        world.maxobscnt_dict = {}
        self.scenario_config = scenario_config
        if 'num_agents' in scenario_config.keys(): 
            num_agents = scenario_config['num_agents']
            if self.scenario_config["dropout"]:
                if not self.scenario_config["dropout"]["inference"] or not self.scenario_config["domain_aware"]:
                    world.maxobscnt_dict["agent"] = scenario_config['max_n_agents']
                else:
                    world.maxobscnt_dict["agent"] = num_agents[1] + scenario_config['intra_trajectory'][0]  # max In Domain entity num (max init + max intra)
        if 'num_adversaries' in scenario_config.keys(): 
            num_adversaries = scenario_config['num_adversaries']
            if self.scenario_config["dropout"]:
                if not self.scenario_config["dropout"]["inference"] or not self.scenario_config["domain_aware"]:
                    world.maxobscnt_dict["adv"] = scenario_config['max_n_advs']
                else:
                    world.maxobscnt_dict["adv"] = num_adversaries[1] + scenario_config['intra_trajectory'][1]  # max In Domain entity num (max init + max intra)
        if 'num_targets' in scenario_config.keys(): 
            num_targets = scenario_config['num_targets']
            world.maxobscnt_dict["target"] = num_targets[1] + scenario_config['intra_trajectory'][2]
        if 'num_decoys' in scenario_config.keys(): 
            num_decoys = scenario_config['num_decoys'] 
            world.maxobscnt_dict["decoy"] = num_decoys[1] + scenario_config['intra_trajectory'][3]

        if 'max_n_agents' in scenario_config.keys(): world.max_n_agents = scenario_config['max_n_agents']
        if 'max_n_advs' in scenario_config.keys(): world.max_n_advs = scenario_config['max_n_advs']
        if 'max_n_targets' in scenario_config.keys(): world.max_n_targets = scenario_config['max_n_targets']
        if 'max_n_decoys' in scenario_config.keys(): world.max_n_decoys = scenario_config['max_n_decoys']

        if 'episode_limit' in scenario_config.keys(): world.episode_limit = scenario_config['episode_limit']

        world.agent_count = num_agents
        world.adv_count = num_adversaries
        world.target_count = num_targets
        world.decoy_count = num_decoys
        
        # add agents
        world.agents = []

        world.good_agents = [Agent() for i in range(num_agents[0])]
        for i, agent in enumerate(world.good_agents):
            agent.adversary = False
            agent.name = "agent_%d" % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
            world.dead_mask[agent.name] = 0

        world.adversaries = [Agent() for i in range(num_adversaries[0])]
        for i, agent in enumerate(world.adversaries):
            agent.adversary = True
            agent.name = "adv_%d" % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
            agent.action_callback = self.heuristic_agent
            world.dead_mask[agent.name] = 0
        
        world.agents += world.good_agents
        world.agents += world.adversaries

        # add landmarks
        world.landmarks = []

        world.targets = [Landmark() for i in range(num_targets[0])]
        for i, landmark in enumerate(world.targets):
            landmark.target = True
            landmark.name = "target_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
            # world.max_n_landmarks += 1
            world.dead_mask[landmark.name] = 0

        world.decoys = [Landmark() for i in range(num_decoys[0])]
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
        
        if self.scenario_config["dropout"]:
            if not self.scenario_config["dropout"]["inference"]:
                world.obs_shape = (world.max_n_agents + world.max_n_advs + world.max_n_targets + world.max_n_decoys) * (4+4)
            else:
                world.obs_shape = (sum(world.maxobscnt_dict.values())) * (4+4)
        else: # if train dropout and inference dropout is False in algs config, scenario_config["dropout"] automatically sets to False
            world.obs_shape = (world.max_n_agents + world.max_n_advs + world.max_n_targets + world.max_n_decoys) * (4+4)

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
    
    def reset_world(self, world, np_random, test, domain_n):
        world.agents = world.init_agents[:]
        world.landmarks = world.init_landmarks[:]

        world.agent_count = len(self.good_agents(world))
        world.adv_count = len(self.adversaries(world))
        world.target_count = len(self.targets(world))
        world.decoy_count = len(self.decoys(world))

        config = self.scenario_config

        if not test: 
            if config['dropout'] and config['dropout']['train']: # uniform sample in combinatorics
                agentsamples = []
                advsamples = []
                for k, v in config['dropout']['agent'].items():
                    for _ in range((v[0]+1) * (v[1]+1)):
                        agentsamples.append(k)
                for k, v in config['dropout']['adv'].items():
                    for _ in range((v[0]+1) * (v[1]+1)):
                        advsamples.append(k)
                num_agents = np_random.choice(agentsamples)
                num_adversaries = np_random.choice(advsamples)

                agentinitDOsamples = np.arange(config['dropout']['agent'][num_agents][0]+1)
                agentintraDOrasamples = np.arange(config['dropout']['agent'][num_agents][1]+1)
                advinitDOsamples = np.arange(config['dropout']['adv'][num_adversaries][0]+1)
                advintraDOrasamples = np.arange(config['dropout']['adv'][num_adversaries][1]+1)

                agentinitDOnum = np_random.choice(agentinitDOsamples)
                agentintraDOnum = np_random.choice(agentintraDOrasamples)
                advinitDOnum = np_random.choice(advinitDOsamples)
                advintraDOnum = np_random.choice(advintraDOrasamples)

                agentnum = self.random_initial_agents(world, np_random, [num_agents]) + self.scenario_config["num_agents"][0]
                advnum = self.random_initial_adversaries(world, np_random, [num_adversaries]) + self.scenario_config["num_adversaries"][0]
                targetnum = self.random_initial_targets(world, np_random, config["num_targets"]) + self.scenario_config["num_targets"][0]
                decoynum = self.random_initial_decoys(world, np_random, config["num_decoys"]) + self.scenario_config["num_decoys"][0]
                world.DOnum = {"agent": {agentnum: agentinitDOnum, agentnum+1: agentinitDOnum + agentintraDOnum},
                "adv": {advnum: advinitDOnum, advnum+1: advinitDOnum + advintraDOnum}}

            else:
                agentnum = self.random_initial_agents(world, np_random, config["num_agents"]) + self.scenario_config["num_agents"][0]
                advnum = self.random_initial_adversaries(world, np_random, config["num_adversaries"]) + self.scenario_config["num_adversaries"][0]
                targetnum = self.random_initial_targets(world, np_random, config["num_targets"]) + self.scenario_config["num_targets"][0]
                decoynum = self.random_initial_decoys(world, np_random, config["num_decoys"]) + self.scenario_config["num_decoys"][0]
                world.DOnum = {}

        if test: # uniform sample in range
            IDmaxagent = config["num_agents"][-1] + config["intra_trajectory"][0]
            IDmaxadv = config["num_adversaries"][-1] + config["intra_trajectory"][1]
            if domain_n == 0:
                config = self.scenario_config
            elif domain_n == 1:
                config = self.scenario_config["OOD"][0]
            elif domain_n == 2:
                config = self.scenario_config["OOD"][1]
            ##for Attention matrix
            elif domain_n == 3 or domain_n == 4: 
                config = self.scenario_config["OOD"][domain_n - 3]
                config["num_agents"][0] += config["intra_trajectory"][0]
                config["num_adversaries"][0] += config["intra_trajectory"][1]
                config["num_targets"] = [config["num_targets"][1] + config["intra_trajectory"][2]]
                config["num_decoys"] = [config["num_decoys"][1] + config["intra_trajectory"][3]]
                config["intra_trajectory"] = [0, 0, 0, 0]
            agentnum = self.random_initial_agents(world, np_random, config["num_agents"]) + self.scenario_config["num_agents"][0]
            advnum = self.random_initial_adversaries(world, np_random, config["num_adversaries"]) + self.scenario_config["num_adversaries"][0]
            targetnum = self.random_initial_targets(world, np_random, config["num_targets"]) + self.scenario_config["num_targets"][0]
            decoynum = self.random_initial_decoys(world, np_random, config["num_decoys"]) + self.scenario_config["num_decoys"][0]

            if self.scenario_config['dropout']:
                world.DOnum = {}
                if self.scenario_config['dropout']['inference'] == True:
                    if self.scenario_config["domain_aware"]:
                        world.DOnum["agent"] = {}
                        for x in range(agentnum, agentnum+config["intra_trajectory"][0]+1):
                            world.DOnum["agent"][x] = max(0, x - IDmaxagent)
                        world.DOnum["adv"] = {}
                        for x in range(advnum, advnum+config["intra_trajectory"][1]+1):
                            world.DOnum["adv"][x] = max(0, x - IDmaxadv)
                    else:
                        world.DOnum["agent"] = {}
                        randomagentDO = np.random.randint(0, max(1, agentnum+config["intra_trajectory"][0]+1 - IDmaxagent))
                        randomadvDO = np.random.randint(0, max(1, advnum+config["intra_trajectory"][1]+1 - IDmaxadv))
                        for x in range(agentnum, agentnum+config["intra_trajectory"][0]+1):
                            world.DOnum["agent"][x] = randomagentDO
                        world.DOnum["adv"] = {}
                        for x in range(advnum, advnum+config["intra_trajectory"][1]+1):
                            world.DOnum["adv"][x] = randomadvDO

                        
        world.states_agents = world.agents[:]
        world.states_landmarks = world.landmarks[:]

        world.n_entities = len(world.agents) + len(world.landmarks)
        intra_count = config["intra_trajectory"]
        world.ts_action = defaultdict(list)
        for action_idx, action_key in enumerate(self.scenario_config['delta_entities']):
            if intra_count[action_idx] <= 1:
                count = intra_count[action_idx]
            else: 
                count = np_random.randint(1, intra_count[action_idx]+1)
            for i in range(count): 
                if test:
                    rand_ts = np_random.randint(10, world.episode_limit -10)
                    world.ts_action[rand_ts].append(action_key)
                else:
                    rand_ts = np_random.randint(0, world.episode_limit)
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
            agent.state.p_pos = np_random.uniform(-world.boundary, +world.boundary, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-world.boundary, +world.boundary, world.dim_p)
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

    # Original reward

    # def reward(self, world):
    #     # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
    #     rew = 0
    #     normalized_agent_values = 0
    #     normalized_adv_values = 0
    #     for target in self.targets(world):
    #         possible_max_dist = self.possible_max_dist(target)
    #         agent_dist = []
    #         adv_dist = []
    #         for agent in world.agents:
    #             # distance between adv and target
    #             if agent.adversary:
    #                 adv_dist.append(world.get_distance(agent, target)) # - target.size - agent.size
    #             # distance between agent and target    
    #             else:
    #                 agent_dist.append(world.get_distance(agent, target)) # - target.size - agent.size
    #         # Calculate positive reward for agents
    #         closest_agent_dist = min(agent_dist)
    #         normalized_agent_dist = (1 - closest_agent_dist / possible_max_dist) * 50
    #         normalized_agent_values += (max(0, min(50, normalized_agent_dist)))
    #         # Calculate negative reward for agents
    #         closest_adv_dist = min(adv_dist)
    #         normalized_adv_dist = (1 - closest_adv_dist / possible_max_dist) * 50
    #         normalized_adv_values += (max(0, min(50, normalized_adv_dist)))
    #     rew += normalized_agent_values / (len(self.targets(world)) + 1e-10)
    #     rew -= normalized_adv_values / (len(self.targets(world)) + 1e-10)
    #     return rew

    # # Min-Sum reward

    def reward(self, world):
        rew = 0
        agent_min_dists = []
        adv_min_dists = []
        for target in self.targets(world):
            agent_dists = []
            adv_dists = []
            for a in world.agents:
                if a.adversary:
                    adv_dists.append(np.sqrt(np.sum(np.square(a.state.p_pos - target.state.p_pos))) - target.size - a.size)  
                else:
                    agent_dists.append(np.sqrt(np.sum(np.square(a.state.p_pos - target.state.p_pos))) - target.size - a.size)         
            agent_min_dists.append(min(agent_dists))
            adv_min_dists.append(min(adv_dists))
        rew -= sum(agent_min_dists)
        rew += sum(adv_min_dists)

        return rew
    
    def get_obs(self, world): # return all agent obs Naively

        return

    def get_entity(self, world): 
        
        return

    def get_mask(self, world): 
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

    def create_new_agent(self, world): 
        agent = Agent()
        agent.adversary = False
        agent.name = f"agent_{world.agent_count}"  # naming agent based on created agent num
        agent.collide = False
        agent.silent = True
        agent.size = 0.05
        world.dead_mask[agent.name] = 0
        world.agent_count += 1
        return agent
        
    def add_new_agent(self,world,np_random):
        agent = self.create_new_agent(world)
        agent.color = np.array([0.35, 0.35, 0.85])
        agent.state.p_pos = self.set_pos_random(agent, world.boundary, np_random)
        agent.state.p_vel = np.zeros(world.dim_p)

        world.new_agent_que.append(agent)

    def create_new_adv(self, world): 
        agent = Agent()
        agent.adversary = True
        agent.name = f"adv_{world.adv_count}"  # naming agent based on created agent num
        agent.collide = False
        agent.silent = True
        agent.size = 0.05
        agent.action_callback = self.heuristic_agent
        world.dead_mask[agent.name] = 0
        world.adv_count += 1
        return agent
        
    def add_new_adv(self,world,np_random):
        agent = self.create_new_adv(world)
        agent.color = np.array([0.85, 0.35, 0.35])
        agent.state.p_pos = self.set_pos_random(agent, world.boundary, np_random)
        agent.state.p_vel = np.zeros(world.dim_p)

        world.new_agent_que.append(agent)
    
    def create_new_target(self, world):
        landmark = Landmark()
        landmark.target = True
        landmark.name = f"target_{world.target_count}"
        landmark.collide = False
        landmark.movable = False
        landmark.size = 0.05
        world.dead_mask[landmark.name] = 0
        world.target_count += 1
        return landmark


    def add_new_target(self,world,np_random):
        landmark = self.create_new_target(world)
        landmark.color = np.array([0.15, 0.65, 0.15])
        landmark.state.p_pos = np_random.uniform(-world.boundary, +world.boundary, world.dim_p)
        landmark.state.p_vel = np.zeros(world.dim_p)

        world.new_landmark_que.append(landmark)

    def create_new_decoy(self, world):
        landmark = Landmark()
        landmark.target = False
        landmark.name = f"decoy_{world.decoy_count}"
        landmark.collide = False
        landmark.movable = False
        landmark.size = 0.05
        world.dead_mask[landmark.name] = 0
        world.decoy_count += 1
        
        return landmark
    
    def add_new_decoy(self,world,np_random):
        landmark = self.create_new_decoy(world)
        landmark.color = np.array([0.15, 0.15, 0.15])
        landmark.state.p_pos = np_random.uniform(-world.boundary, +world.boundary, world.dim_p)
        landmark.state.p_vel = np.zeros(world.dim_p)
        
        world.new_landmark_que.append(landmark)

    def kill_agent(self, world, agent):
        world.del_agent_que.add(agent.name)
        world.dead_mask[agent.name] = 1
    
    def kill_landmark(self, world, landmark):
        world.del_landmark_que.add(landmark.name)
        world.dead_mask[landmark.name] = 1

    def set_pos_random(self, entity, boundary, np_random): 
        edge = np_random.randint(4)
        if edge == 0:  # t
            x = np_random.uniform(-boundary, boundary)
            y = boundary
        elif edge == 1:  # r
            x = boundary
            y = np_random.uniform(-boundary, boundary)
        elif edge == 2:  # d
            x = np_random.uniform(-boundary, boundary)
            y = -boundary
        else:  # l
            x = -boundary
            y = np_random.uniform(-boundary, boundary)
        return np.array([x, y])

    def random_initial_agents(self, world, np_random, config): 
        if len(config) == 2:
            start, end = config
            initial_num = np_random.randint(start, end+1) - self.scenario_config['num_agents'][0]
        else:
            initial_num = config[0] - self.scenario_config['num_agents'][0]
        
        for i in range(initial_num): 
            agent = self.create_new_agent(world)
            world.agents.append(agent)
        return initial_num
            
    def random_initial_adversaries(self, world, np_random, config): 
        if len(config) == 2:
            start, end = config
            initial_num = np_random.randint(start, end+1) - self.scenario_config['num_adversaries'][0]
        else:
            initial_num = config[0] - self.scenario_config['num_adversaries'][0]
        
        for i in range(initial_num): 
            agent = self.create_new_adv(world)
            world.agents.append(agent)
        return initial_num
    
    def random_initial_targets(self, world, np_random, config):
        if len(config) == 2:
            start, end = config
            initial_num = np_random.randint(start, end+1) - self.scenario_config['num_targets'][0]
        else:
            initial_num = config[0] - self.scenario_config['num_targets'][0]
            
        for i in range(initial_num): 
            target = self.create_new_target(world)
            world.landmarks.append(target)
        return initial_num
        
    def random_initial_decoys(self, world, np_random, config):
        if len(config) == 2:
            start, end = config
            initial_num = np_random.randint(start, end+1) - self.scenario_config['num_decoys'][0]
        else:
            initial_num = config[0] - self.scenario_config['num_decoys'][0]
            
        for i in range(initial_num): 
            target = self.create_new_decoy(world)
            world.landmarks.append(target)
        return initial_num
    