from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import copy
from components.transforms import process_agent_obs, process_state


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger, task):
        self.args = args
        self.logger = logger
        self.task = task
        self.batch_size = self.args.batch_size_run
        
        args.entity_dropout = False

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        worker_id2env_args = {}
        for worker_id in range(self.batch_size):
            worker_id2env_args[worker_id] = copy.deepcopy(self.args.env_args)
            # worker_id2env_args[worker_id]["seed"] += worker_id
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **worker_id2env_args[worker_id])), self.args))
                            for worker_id, worker_conn in enumerate(self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))
    
    def reset(self, **kwargs):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", kwargs))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, pretrain=False, domain_n = 0):
        self.reset(test=test_mode, domain_n = domain_n)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if pretrain:
                # If pretrain phase, just select action randomly
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=0, task=self.task, bs=envs_not_terminated, test_mode=False)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, task=self.task, bs=envs_not_terminated, test_mode=test_mode)
        
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run
        
        # Get stats back for each env NOTE MPE env
        #for parent_conn in self.parent_conns:
        #    parent_conn.send(("get_stats",None))

        # env_stats = []
        # for parent_conn in self.parent_conns:
        #     env_stat = parent_conn.recv()
        #     env_stats.append(env_stat)

        if not pretrain:        
            cur_stats = self.test_stats if test_mode else self.train_stats
            cur_returns = self.test_returns if test_mode else self.train_returns
            log_prefix = f"{self.task}/test_" if test_mode else f"{self.task}/"
            # infos = [cur_stats] + final_env_infos NOTE Mpe env
            # cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])}) NOTE Mpe env
            cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
            cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

            cur_returns.extend(episode_returns)

            n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
            if test_mode and (len(self.test_returns) == n_test_runs):
                self._log(cur_returns, cur_stats, log_prefix, domain_n=domain_n)
            elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                self._log(cur_returns, cur_stats, log_prefix)
                if hasattr(self.mac.action_selector, "epsilon"):
                    self.logger.log_stat(f"{self.task}/epsilon", self.mac.action_selector.epsilon, self.t_env)
                self.log_train_stats_t = self.t_env
        return self.batch

    def _log(self, returns, stats, prefix, domain_n = 0):
        self.logger.log_stat(prefix + f"return_mean_{domain_n}", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + f"return_std_{domain_n}", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn, args):
    # Make environment
    env = env_fn.x()
    env_info = env.get_env_info(args)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, _, env_info_ = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = process_state(args, env_info, env.get_state())
            obs = process_agent_obs(args, args.agentnames[0], env_info, env.get_state(), None)
            avail_actions = env.get_avail_actions(args.n_agents)

            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset(**data)
            state = process_state(args, env_info, env.get_state())
            obs = process_agent_obs(args, args.agentnames[0], env_info, env.get_state(), None)
            avail_actions = env.get_avail_actions(args.n_agents)
            remote.send({
                "state": state,
                "avail_actions": env.get_avail_actions(args.n_agents),
                "obs": obs
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info(args))
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

