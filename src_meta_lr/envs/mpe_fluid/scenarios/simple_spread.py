import numpy as np
from envs.mpe_fluid.core import World, Agent, Landmark
from envs.mpe_fluid.scenario import BaseScenario
from envs.mpe_fluid.core import Action
from copy import deepcopy
from collections import defaultdict
import time


class Scenario(BaseScenario):
    def make_world(self, scenario_config={}):
        num_agents = 5
        num_landmarks = 5
        episode_limit = 100
        n_actions = 5
        world = World()

        world.boundary = 1.0
        
        world.max_n_agents = 0
        world.max_n_landmarks = 0
        world.episode_limit = episode_limit
        world.n_actions = n_actions 

        # set random initial number of entities
        self.scenario_config = scenario_config
        if 'num_agents' in scenario_config.keys():
            start, end = scenario_config['num_agents']
            num_agents = np.random.randint(start, end)
        if 'num_landmarks' in scenario_config.keys():
            start, end = scenario_config['num_landmarks']
            num_agents = np.random.randint(start, end)
        
        if 'max_n_agents' in scenario_config.keys(): world.max_n_agents = scenario_config['max_n_agents']
        if 'max_n_landmarks' in scenario_config.keys(): world.max_n_landmarks = scenario_config['max_n_landmarks']
        
        if 'episode_limit' in scenario_config.keys(): world.episode_limit = scenario_config['episode_limit']
        if 'n_actions' in scenario_config.keys(): world.n_actions = scenario_config['n_actions']
        
        # # keep initial combination for episode reset 
        # world.init_agents = []
        # world.init_landmarks = []
        # for numbering new entities
        world.agent_count = num_agents
        world.landmark_count = num_landmarks

        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.04
        
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"target_{i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
          
        # world.init_agents += world.agents 
        # world.init_landmarks += world.landmarks
        world.set_dictionaries()
          
        world.onehot_dict = {"agent": 0, "target": 1}
        world.obs_info = {"pos": 2, "vel": 2}
        world.entity_maxnum = {"agent": world.max_n_agents, "target": world.max_n_landmarks}
        
        world.state_shape = (
            world.max_n_agents * (2+4) +
            world.max_n_landmarks * (2+4)
        )
        world.obs_shape = (
            world.max_n_agents * (2+4) +
            world.max_n_landmarks * (2+4)
        )
        
        world.max_n_entities = world.max_n_agents + world.max_n_landmarks
        world.max_entity_size = len(world.onehot_dict.keys()) + sum(world.obs_info.values())
        world.n_entities = len(world.agents) + len(world.landmarks)
        
        return world

    def heuristic_agent(self, agent, world): 
        pass 
    
    def reset_world(self, world, np_random, test, domain_n, meta_lr_scheme, meta_test_mode):
        world.agents = [] # world.init_agents[:]
        world.landmarks = [] # world.init_landmarks[:]

        world.agent_count = len(world.agents) 
        world.landmark_count = len(world.landmarks) 

        intra_count = self.scenario_config["intra_trajectory"]
        if (test == False) or (test == True and domain_n == 0):
            if meta_lr_scheme == "MLDG" or meta_lr_scheme == 'DG-MAML':
                self.MLDG_initial_agents(world, np_random, self.scenario_config["num_agents"], meta_test_mode)
                self.MLDG_initial_landmarks(world, np_random, self.scenario_config["num_landmarks"], meta_test_mode)

        # set OOD config for evalutation
        elif test == True and domain_n != 0:
            test_config = self.scenario_config["OOD"][0] if domain_n == 1 else self.scenario_config["OOD"][1]
            self.random_initial_agents(world, np_random, test_config["num_agents"])
            self.random_initial_landmarks(world, np_random, test_config["num_landmarks"])
            intra_count = test_config["intra_trajectory"]
    
        # set random timesteps for intra-trajectory
        world.ts_action = defaultdict(list)
        for action_idx, (action_key, manual_timesteps) in enumerate(self.scenario_config['delta_entities'].items()):
            if intra_count[action_idx] <= 1:
                count = intra_count[action_idx]
            else: 
                count = np_random.randint(1, intra_count[action_idx])
            for i in range(count): 
                if test:
                    rand_ts = np_random.randint(10, world.episode_limit -10)
                    world.ts_action[rand_ts].append(action_key)
                else:
                    rand_ts = np_random.randint(0, world.episode_limit)
                    world.ts_action[rand_ts].append(action_key)
                    # timestep_manual = manual_timesteps[i]
                    # world.ts_action[timestep_manual].append(action_key)
                
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.45, 0.95])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            
        for i, landmark in enumerate(world.landmarks):
            max_attempts = 100
            for _ in range(max_attempts):
                landmark.state.p_pos = np_random.uniform(-world.boundary, +world.boundary, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                if all(not self.is_collision(landmark, other) for other in world.landmarks[:i]):
                    break
        
        for i, agent in enumerate(world.agents):
            max_attempts = 100
            for _ in range(max_attempts):
                agent.state.p_pos = np_random.uniform(-world.boundary, +world.boundary, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                # agent.state.c = np.zeros(world.dim_c)
                if all(not self.is_collision(agent, other) for other in world.agents[:i]):
                    break
        
        world.set_dictionaries()
        world.calc_distmat()
        world.time_step = 0

    def update_que(self, world, np_random):
        while world.time_step in world.ts_action and len(world.ts_action[world.time_step]) > 0: 
            action = world.ts_action[world.time_step].pop(0)
            if action == 'add_agent': 
                self.add_new_agent(world, np_random)
            elif action == 'add_landmark': 
                self.add_new_landmark(world, np_random)
                
    def done(self, world):
        if world.time_step >= world.episode_limit: return True
        return False

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist <= dist_min else False
        
    def reward(self, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        min_dists = []
        for lm in world.landmarks:
            dist = []
            for a in world.agents: 
                dist.append(np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos))) - lm.size - a.size)         
            min_dists.append(min(dist))
        rew -= sum(min_dists)
        return rew

    def get_entity(self, world): 
        
        return

    def get_mask(self, world): 
        obs_mask = np.zeros((world.max_n_entities, world.max_n_entities), dtype=np.uint8)
        entity_mask = np.ones((world.max_n_entities), dtype=np.uint8)

        cnt = 0
        for i in range(world.max_n_agents):
            if f"agent_{i}" in world.all_agents.keys():
                if world.all_agents[f"agent_{i}"]: # created, dead
                    obs_mask[cnt,:] = 1
                    obs_mask[:,cnt] = 1
                    # print("absent from env: ", cnt) #for debug
                else: # created, alive
                    entity_mask[cnt] = 0
                cnt += 1
            else: # not created yet
                obs_mask[cnt,:] = 1
                obs_mask[:,cnt] = 1
                # print("absent from env: ", cnt) #for debug
                cnt += 1
        
        for i in range(world.max_n_landmarks):
            if f"target_{i}" in world.all_landmarks.keys():
                if world.all_landmarks[f"target_{i}"]: # created, dead
                    obs_mask[cnt,:] = 1
                    obs_mask[:,cnt] = 1
                    # print("absent from env: ", cnt) #for debug
                cnt += 1
            else: # not created yet
                obs_mask[cnt,:] = 1
                obs_mask[:,cnt] = 1
                #print("absent from env: ", cnt) #for debug
                cnt += 1
        
        return obs_mask, entity_mask
    
    def get_obs(self, world): 
        return 
    
    def get_state(self, world):
        state_global = {} 
        for agent_name, is_dead in world.all_agents.items():
            if(not is_dead): agent = world.entities[world.index_map[agent_name]]
            state_global[f"{agent_name.split('_')[0]}-pos_{agent_name.split('_')[1]}"] = {
                "element": agent.state.p_pos.tolist() if not is_dead else np.zeros(2).tolist(),
                "mask": is_dead
            }
            state_global[f"{agent_name.split('_')[0]}-vel_{agent_name.split('_')[1]}"]= {
                "element": agent.state.p_vel.tolist() if not is_dead else np.zeros(2).tolist(),
                "mask": is_dead 
            } 
        for landmark_name, is_dead in world.all_landmarks.items():
            if(not is_dead): landmark = world.entities[world.index_map[landmark_name]]
            state_global[f"{landmark_name.split('_')[0]}-pos_{landmark_name.split('_')[1]}"] = {
                "element": landmark.state.p_pos.tolist() if not is_dead else np.zeros(2).tolist(),
                "mask": is_dead 
            }
        return state_global
        
    def add_new_agent(self, world, np_random): 
        agent = Agent() 
        agent.name = f"agent_{world.agent_count}"
        agent.collide = True
        agent.size = 0.04
        agent.color = np.array([0.45, 0.45, 0.95])
        # agent.color = np.array([0.95, 0.45, 0.45]) #red
        self.set_pos_random(agent, world.boundary*0.98, np_random)
        agent.state.p_vel = np.zeros(world.dim_p)
        
        world.agent_count += 1
        world.all_agents[agent.name] = 0
        world.new_agent_que.append(agent)
        
    def add_new_landmark(self, world, np_random):
        landmark = Landmark() 
        landmark.name = f"target_{world.landmark_count}" 
        landmark.collide = False
        landmark.movable = False
        landmark.size = 0.04
        landmark.color = np.array([0.25, 0.25, 0.25])
        # landmark.color = np.array([0.95, 0.45, 0.45]) #red
        r_prime, theta = world.boundary * 0.8 * np.sqrt(np.random.uniform()), np.random.uniform(0, 2 * np.pi)
        landmark.state.p_pos = np.array([r_prime * np.cos(theta), r_prime * np.sin(theta)])
        landmark.state.p_vel = np.zeros(world.dim_p)
        
        world.landmark_count += 1
        world.all_landmarks[landmark.name] = 0
        world.new_landmark_que.append(landmark)

    def kill_agent(self, world, agent):
        if agent is not None: 
            world.del_agent_que.add(agent.name)
            world.all_agents[agent.name] = 1

    def kill_landmark(self, world, landmark):
        if landmark is not None:
            world.del_landmark_que.add(landmark.name)
            world.all_landmarks[landmark.name] = 1

    def kill_random_agent(self, world, np_random): 
        max_attempts = 50
        for _ in range(max_attempts):
            if not world.agents: 
                break 
            rand_agent = np_random.choice(world.agents)
            already_killed = world.all_agents[rand_agent.name]
            if not already_killed:
                self.kill_agent(world, rand_agent)
                break
            
    def kill_random_landmark(self, world, np_random): 
        max_attempts = 50
        for _ in range(max_attempts):
            if not world.landmarks: 
                break 
            rand_landmark = np_random.choice(world.landmarks)
            already_killed = world.all_landmarks[rand_landmark.name]
            if not already_killed:
                self.kill_landmark(world, rand_landmark)
                break
            
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
        
        entity.state.p_pos = np.array([x, y])

    def random_initial_agents(self, world, np_random, config):
        if len(config) == 2:
            start, end = config
            initial_num = np_random.randint(start, end+1)
        else:
            initial_num = config[0]

        for i in range(initial_num):
            agent = Agent()
            agent.name = f"agent_{world.agent_count}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.04

            world.all_agents[agent.name] = 0
            world.agent_count += 1
            world.agents.append(agent)
    
    def random_initial_landmarks(self, world, np_random, config):
        if len(config) == 2:
            start, end = config
            initial_num = np_random.randint(start, end+1)
        else:
            initial_num = config[0]

        for i in range(initial_num):
            landmark = Landmark()
            landmark.name = f"target_{world.landmark_count}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04

            world.landmark_count += 1
            world.all_landmarks[landmark.name] = 0
            world.landmarks.append(landmark)

    def MLDG_initial_agents(self, world, np_random, config, meta_test_mode):
        if len(config) == 2:
            if meta_test_mode:
                initial_num = config[-1]
            else:
                start, end = config
                initial_num = np_random.randint(start, end)
        else:
            initial_num = config[0]

        for i in range(initial_num):
            agent = Agent()
            agent.name = f"agent_{world.agent_count}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.04

            world.all_agents[agent.name] = 0
            world.agent_count += 1
            world.agents.append(agent)
    
    def MLDG_initial_landmarks(self, world, np_random, config, meta_test_mode):
        if len(config) == 2:
            if meta_test_mode:
                initial_num = config[-1]
            else:
                start, end = config
                initial_num = np_random.randint(start, end)
        else:
            initial_num = config[0]

        for i in range(initial_num):
            landmark = Landmark()
            landmark.name = f"target_{world.landmark_count}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04

            world.landmark_count += 1
            world.all_landmarks[landmark.name] = 0
            world.landmarks.append(landmark)
