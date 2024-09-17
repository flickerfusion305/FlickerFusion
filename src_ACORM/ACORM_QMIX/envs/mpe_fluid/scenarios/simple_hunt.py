import numpy as np
from envs.mpe_fluid.core import World, Agent, Landmark
from envs.mpe_fluid.scenario import BaseScenario
from envs.mpe_fluid.core import Action
from copy import deepcopy
from collections import defaultdict
import time


class Scenario(BaseScenario):
    def make_world(self, scenario_config = {}):
        num_agents = 3
        num_adversaries = 3
        episode_limit = 50
        n_actions = 5
        world = World()
        
        world.boundary = 2.0 # define boundary
        
        world.max_n_agents = 0
        world.max_n_advs = 0
        world.episode_limit = episode_limit 
        world.n_actions = n_actions

        # set random initial number of entities
        self.scenario_config = scenario_config
        if 'num_agents' in scenario_config.keys():
            start, end = scenario_config['num_agents']
            num_agents = np.random.randint(start, end)
        if 'num_adversaries' in scenario_config.keys():
            start, end = scenario_config['num_adversaries']
            num_adversaries = np.random.randint(start, end)

        if 'max_n_agents' in scenario_config.keys(): world.max_n_agents = scenario_config['max_n_agents']
        if 'max_n_advs' in scenario_config.keys(): world.max_n_advs = scenario_config['max_n_advs']

        if 'episode_limit' in scenario_config.keys(): world.episode_limit = scenario_config['episode_limit']
        if 'n_actions' in scenario_config.keys(): world.n_actions = scenario_config['n_actions']

        # # keep initial combination for episode reset        
        # world.init_agents = []
        # world.init_landmarks = []
        # for numbering new entities
        world.agent_count = num_agents
        world.adv_count = num_adversaries
        world.total_collision = 0
        # add agents
        world.agents = []
        world.good_agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.good_agents):
            agent.adversary = False
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 1.3
            
        world.adversaries = [Agent() for i in range(num_adversaries)]
        for i, agent in enumerate(world.adversaries):
            agent.adversary = True
            agent.name = f"adv_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 3.0 
            agent.max_speed = 1.0 
            agent.action_callback = self.heuristic_agent
                
        world.agents += world.good_agents
        world.agents += world.adversaries

        # world.init_agents += world.agents
        # world.init_landmarks += world.landmarks
        world.set_dictionaries()
        
        world.onehot_dict = {"agent": 0, "adv": 1}
        world.obs_info = {"pos": 2, "vel": 2}
        world.entity_maxnum = {"agent": world.max_n_agents, "adv": world.max_n_advs}
        
        world.state_shape = (world.max_n_agents + world.max_n_advs) * (2+4)
        world.obs_shape = (world.max_n_agents + world.max_n_advs) * (2+4)
        
        world.max_n_entities = world.max_n_agents + world.max_n_advs 
        world.max_entity_size = len(world.onehot_dict.keys()) + sum(world.obs_info.values())
        world.n_entities = len(world.agents)
                
        return world

    def heuristic_agent(self, agent, world): 
        action = Action() 
        action.u = np.zeros(world.dim_p)
        dist_dict = {}
        
        for target in self.good_agents(world): 
            dist_dict[target.name] = world.get_distance(target, agent)
            
        goal = min(dist_dict, key=dist_dict.get)
        dx, dy = world.get_relpos_byname(agent.name, goal)
        if not world.get_distance_byname(agent.name, goal) < world.boundary/2:
           dx, dy = np.random.uniform(-1, 1, world.dim_p)
        vx, vy = agent.state.p_vel
        damping = 0.25
        dt = 0.1
        max_accel = 1

        next_x, next_y = agent.state.p_pos + agent.state.p_vel * dt

        if next_x < -world.boundary:
            dx = (-world.boundary) - agent.state.p_pos[0]
        elif next_x > world.boundary:
            dx = (+world.boundary) - agent.state.p_pos[0]

        if next_y < -world.boundary:
            dy = (-world.boundary) - agent.state.p_pos[1]
        elif next_y > world.boundary:
            dy = (+world.boundary) - agent.state.p_pos[1]

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
        world.agents = [] # world.init_agents[:]
            
        world.agent_count = len(self.good_agents(world)) 
        world.adv_count = len(self.adversaries(world))
        world.total_collision = 0
        config = self.scenario_config
        if test and (domain_n > 0):
            config = self.scenario_config["OOD"][0] if domain_n == 1 else self.scenario_config["OOD"][1]
        self.random_initial_agents(world, np_random, config["num_agents"])
        self.random_initial_advs(world, np_random, config["num_adversaries"])
        intra_count = config["intra_trajectory"]
        # set random timesteps for intra-trajectory
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
                    # timestep_manual = manual_timesteps[i]
                    # world.ts_action[timestep_manual].append(action_key)
                
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.35, 0.85])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
        # set random initial states
        for i, agent in enumerate(world.agents):
            max_attempts = 100
            for _ in range(max_attempts):
                if (agent.adversary):
                    x_pos = np_random.uniform(-world.boundary/4, world.boundary/4)
                    y_pos = np_random.uniform(-world.boundary/4, world.boundary/4)
                else:
                    x_pos = np_random.choice([np_random.uniform(-world.boundary, -world.boundary/4), np_random.uniform(world.boundary/4, world.boundary)])
                    y_pos = np_random.choice([np_random.uniform(-world.boundary, -world.boundary/4), np_random.uniform(world.boundary/4, world.boundary)])
                agent.state.p_pos = np.array([x_pos, y_pos])
                agent.state.p_vel = np.zeros(world.dim_p)
                #agent.state.c = np.zeros(world.dim_c)
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
            elif action == 'add_adv': 
                self.add_new_adv(world, np_random)
            
    
    def done(self, world):
        if world.time_step >= world.episode_limit: return True
        return False
    
    def additional_reports(self, world):
        if self.done(world):
            return (world.total_collision, )
        return None
    
    def outside_boundary(self, world, agent):
        if (
            agent.state.p_pos[0] > world.boundary
            or agent.state.p_pos[0] < -world.boundary
            or agent.state.p_pos[1] > world.boundary
            or agent.state.p_pos[1] < -world.boundary
        ):
            return True
        else:
            return False  
        
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist <= dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
        
    def reward(self, world):
        rew = 0
        min_dists_adv = []
        for adv in self.adversaries(world):
            dist = []
            for a in self.good_agents(world): 
                dist.append(np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) - adv.size - a.size)         
            min_dists_adv.append(min(dist))
        rew -= sum(min_dists_adv)/len(self.adversaries(world))
        min_dists_agent = []
        for agent in self.good_agents(world):
            dist = []
            for a in self.adversaries(world):
                dist.append(np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) - agent.size - a.size)
            min_dists_agent.append(min(dist))
        rew -= sum(min_dists_agent)/len(self.good_agents(world))

        return rew

    def get_obs(self, world):

        return  
    
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
                else: # created, alive
                    entity_mask[cnt] = 0
                cnt += 1
            else: # not created yet
                obs_mask[cnt,:] = 1
                obs_mask[:,cnt] = 1
                cnt += 1

        for i in range(world.max_n_advs):
            if f"adv_{i}" in world.all_agents.keys():
                if world.all_agents[f"adv_{i}"]: # created, dead
                    obs_mask[cnt,:] = 1
                    obs_mask[:,cnt] = 1
                cnt += 1
            else: # not created yet
                obs_mask[cnt,:] = 1
                obs_mask[:,cnt] = 1
                cnt += 1
        
        
        return obs_mask, entity_mask
    
    def get_state(self, world):
        state_global = {} 
        for agent_name, is_dead in world.all_agents.items():
            if (not is_dead): 
                try: 
                    agent = world.entities[world.index_map[agent_name]]
                except: 
                    print(f"world.index_map: {world.index_map}")
            state_global[f"{agent_name.split('_')[0]}-pos_{agent_name.split('_')[1]}"] = {
                "element": agent.state.p_pos.tolist() if not is_dead else np.zeros(2).tolist(),
                "mask": is_dead
            }
            state_global[f"{agent_name.split('_')[0]}-vel_{agent_name.split('_')[1]}"]= {
                "element": agent.state.p_vel.tolist() if not is_dead else np.zeros(2).tolist(),
                "mask": is_dead 
            } 
        return state_global
    
    def add_new_agent(self, world, np_random): 
        agent = Agent()
        agent.adversary = False
        agent.name = f"agent_{world.agent_count}" 
        agent.collide = True
        agent.silent = True
        agent.size = 0.05
        agent.accel = 3.0
        agent.max_speed = 1.3
        
        agent.color = np.array([0.35, 0.35, 0.85])
        self.set_pos_random(agent, world.boundary*0.98, np_random)
        agent.state.p_vel = np.zeros(world.dim_p)
        
        world.agent_count += 1
        world.all_agents[agent.name] = 0
        world.new_agent_que.append(agent)
        
    def add_new_adv(self, world, np_random):
        agent = Agent()
        agent.adversary = True
        agent.name = f"adv_{world.adv_count}" 
        agent.collide = True
        agent.silent = True
        agent.size = 0.05
        agent.accel = 3.0 
        agent.max_speed = 1.0 
        agent.action_callback = self.heuristic_agent
    
        agent.color = np.array([0.85, 0.35, 0.35])
        self.set_pos_random(agent, world.boundary*0.98, np_random)
        agent.state.p_vel = np.zeros(world.dim_p)
        
        world.adv_count += 1
        world.all_agents[agent.name] = 0
        world.new_agent_que.append(agent)
        
    def kill_agent(self, world, agent):
        if agent is not None: 
            world.del_agent_que.add(agent.name)
            world.all_agents[agent.name] = 1
        
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
            agent.adversary = False
            agent.name = f"agent_{world.agent_count}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 1.3

            world.all_agents[agent.name] = 0
            world.agent_count += 1
            world.agents.append(agent)

    def random_initial_advs(self, world, np_random, config):
        if len(config) == 2:
            start, end = config
            initial_num = np_random.randint(start, end+1)
        else:
            initial_num = config[0]
        
        for i in range(initial_num):
            agent = Agent()
            agent.adversary = True
            agent.name = f"adv_{world.adv_count}" 
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 3.0 
            agent.max_speed = 1.0 
            agent.action_callback = self.heuristic_agent

            world.all_agents[agent.name] = 0
            world.adv_count += 1
            world.agents.append(agent)
    
        
