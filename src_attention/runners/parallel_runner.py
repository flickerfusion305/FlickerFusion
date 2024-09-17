from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from components.transforms import process_agent_obs, process_state, process_entities, process_EntityDropoutAtten, process_EntityDropoutObs


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        # TODO: Add a delay when making sc2 envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])

        if self.args.entity_dropout_train or self.args.entity_dropout_inference:
            self.args.entity_dropout = True
            self.args.env_args["scenario_config"]["dropout"]["train"] = self.args.entity_dropout_train
            self.args.env_args["scenario_config"]["dropout"]["inference"] = self.args.entity_dropout_inference
        else:
            self.args.entity_dropout = False
            self.args.env_args["scenario_config"]["dropout"] = False

        env_fn = env_REGISTRY[self.args.env]
        # if ('sc2' in self.args.env) or ('group_matching' in self.args.env)\
        #      or ('particle' in self.args.env) or ('catch' in self.args.env):
        base_seed = self.args.env_args.pop('seed')
        self.ps = [Process(target=env_worker, args=(self.args, worker_conn, self.args.entity_scheme,
                                                    CloudpickleWrapper(partial(env_fn, seed=base_seed + rank,
                                                                                **self.args.env_args))))
                    for rank, worker_conn in enumerate(self.worker_conns)]
        self.args.env_args['seed']=base_seed
        # else:
        #     self.ps = [Process(target=env_worker, args=(worker_conn, self.args.entity_scheme,
        #                                                 CloudpickleWrapper(partial(env_fn, env_args=self.args.env_args, args=self.args))))
        #                for worker_conn in self.worker_conns]
        
        for p in self.ps:
            p.daemon = True
            p.start()

        # TODO: Close stuff if appropriate

        self.parent_conns[0].send(("get_env_info", args))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        # TODO: Will have to add stuff to episode batch for envs that terminate at different times to ensure filled is correct
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.test_additionals = []
        self.train_additionals = []

        self.log_train_stats_t = -100000
        self.n_agents = self.env_info["n_agents"]

        #for reward rendering
        self.episode_rewards = []

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        # TODO: Remove these if the runner doesn't need them
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        
    def plot_rewards(self, episode_rewards):
        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Episode Rewards Over Time')
        plt.show()

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

        pre_transition_data = {}
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            for k, v in data.items():
                if k in pre_transition_data:
                    pre_transition_data[k].append(data[k])
                else:
                    pre_transition_data[k] = [data[k]]

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0
        
    def run(self, test_mode=False, test_scen=None, index=None, vid_writer=None, constrain_num=None, domain_n =0):
        """
        test_mode: whether to use greedy action selection or sample actions
        test_scen: whether to run on test scenarios. defaults to matching test_mode.
        vid_writer: imageio video writer object (not supported in parallel runner)
        """
        if test_scen is None:
            test_scen = test_mode
        assert vid_writer is None, "Writing videos not supported for ParallelRunner"
        if self.args.test_unseen:
            constrain_num=self.args.test_map_num if test_mode else self.args.train_map_num
        else:
            constrain_num=None
        if self.args.env == "traffic_junction":
            self.reset(t_env=self.t_env)
        else:
            self.reset(test=test_scen, index=index, constrain_num=constrain_num, domain_n = domain_n)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        additional_values = []
        # initialize the hidden states of the agents before the start of a new episode or sequence.
        self.mac.init_hidden(batch_size=self.batch_size)
        # make sure things like dropout are disabled
        if test_mode:
            self.mac.eval()
        else:
            self.mac.train()
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            # TODO: find a bug here
            if self.args.mac == "comm_mac" or self.args.mac=="heucomm_mac" or self.args=="dppcomm_mac":
                actions, p_msg, h_msg = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode, ret_msg=True)
                cpu_actions = actions.to("cpu").numpy()
                cpu_p_msg = p_msg.detach().cpu().numpy()
                cpu_h_msg = h_msg.detach().cpu().numpy()
                actions_chosen = {
                    "actions": actions.unsqueeze(1),
                    "self_message": cpu_p_msg,
                    "head_message": cpu_h_msg
                }
            elif self.args.mac == "rlcomm_mac":
                actions, p_msg, h_msg, head_prob, election_actions = self.mac.select_actions(
                    self.batch, t_ep = self.t, t_env = self.t_env, bs=envs_not_terminated,
                    test_mode=test_mode, ret_msg=True)
                cpu_actions = actions.to("cpu").numpy()
                cpu_p_msg = p_msg.detach().cpu().numpy()
                cpu_h_msg = h_msg.detach().cpu().numpy()
                actions_chosen = {
                    "actions": actions.unsqueeze(1),
                    "self_message": cpu_p_msg,
                    "head_message": cpu_h_msg
                }
                # TODO: complete here???
                if head_prob is None:
                    bs = len(envs_not_terminated)
                    head_chosen = {
                        "head_probs": -1.0*np.ones((bs, 1)).astype(np.float32),
                        "head_actions": np.ones((bs, self.env_info["n_agents"])).astype(np.float32)
                    }
                else:
                    # TODO: we needs bp here
                    head_chosen = {
                        "head_probs": head_prob[envs_not_terminated],
                        "head_actions": election_actions[envs_not_terminated], 
                    }
                actions_chosen.update(head_chosen)
            else: # Ours don't need comm
                if self.args.save_entities_and_attn_weights:
                    actions, attention_values = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode, ret_attn_weights=True)
                else:
                    actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
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
                        parent_conn.send(("step", (cpu_actions[action_idx], attention_values[idx].detach().cpu().numpy())))
                    action_idx += 1 # actions is not a list over every env

            # Post step data we will insert for the current timestep
            post_transition_data = {
                # "actions": actions.unsqueeze(1),
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            if self.args.entity_scheme: # REFIL / CAMA?
                pre_transition_data = {
                    "entities": [],
                    "obs_mask": [],
                    "entity_mask": [],
                    "avail_actions": []
                }
            else: #QMIX
                pre_transition_data = {
                    "state": [],
                    "avail_actions": [],
                    "obs": []
                }

            # Update terminated envs after adding post_transition_data
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

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
                        additional_values.append(data["additional"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    for k in pre_transition_data:
                        pre_transition_data[k].append(data[k])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data

            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)


        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        if 'sc2' in self.args.env:
            for parent_conn in self.parent_conns:
                parent_conn.send(("get_stats",None))

            env_stats = []
            for parent_conn in self.parent_conns:
                env_stat = parent_conn.recv()
                env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_additionals = self.test_additionals if test_mode else self.train_additionals
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        if self.args.entity_scheme:
            vis, vis_b10, vis_p10  = self.calc_visibility(self.batch["obs_mask"], self.batch["entity_mask"], entities = self.batch["entities"])
            cur_stats["visibility"] = sum(vis) + cur_stats.get("visibility", 0)
            cur_stats["visibility_b10"] = sum(vis_b10) + cur_stats.get("visibility_b10", 0)
            cur_stats["visibility_p10"] = sum(vis_p10) + cur_stats.get("visibility_p10", 0)
        # vis, vis_b10, vis_p10  = self.calc_visibility(self.batch["obs_mask"], self.batch["entity_mask"], entities = self.batch["entities"])
        # cur_stats["visibility"] = sum(vis) + cur_stats.get("visibility", 0)
        # cur_stats["visibility_b10"] = sum(vis_b10) + cur_stats.get("visibility_b10", 0)
        # cur_stats["visibility_p10"] = sum(vis_p10) + cur_stats.get("visibility_p10", 0)
        


        cur_returns.extend(episode_returns)
        cur_additionals.extend(additional_values)
        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            if isinstance(cur_additionals[0], tuple):
                self.mean_additional = self._log_additional(cur_additionals, log_prefix)
            self.rm = self._log(cur_returns, cur_stats, log_prefix, domain_n)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            if 'sc2' in self.args.env:
                self.logger.log_stat("forced_restarts",
                                     sum(es['restarts'] for es in env_stats),
                                     self.t_env)
            self.log_train_stats_t = self.t_env
        return self.batch

    def _log(self, returns, stats, prefix, domain_n = 0):
        rm = np.mean(returns)
        self.logger.log_stat(prefix + f"return_mean_{domain_n}", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
        return rm
    
    def _log_additional(self, returns, prefix):
        mean_additional = np.mean(returns, axis=0)
        self.logger.log_stat(prefix + "addi_mean", np.mean(returns, axis=0)[0], self.t_env)
        self.logger.log_stat(prefix + "time_mean", np.mean(returns, axis=0)[0], self.t_env)
        returns.clear()
        return mean_additional

    def calc_visibility(self, obs_mask, agent_mask, entities=None):#[bs,t,ne,ne]; #[bs,t,ne,edim]
        health_ind = {"3-8sz_symmetric": 30, "3-8MMM_symmetric": 39, "3-8csz_symmetric": 31}
        if 'sc2custom' in self.args.env:
            ind =health_ind[self.args.scenario]
            agent_mask = (entities[:,:,:,ind] > 0).float() #In sc2custom, agent_mask includes the dead agents.
        else:
            agent_mask = 1-agent_mask
        obs_mask = 1-obs_mask
        obs_mask = obs_mask.masked_fill((1-agent_mask).bool().unsqueeze(-1),0)
        agent_num = agent_mask[:, :, :self.n_agents].sum(2) #[bs,t]
        entity_num = agent_mask.sum(2) #[bs,t]
        invalid_frame = th.logical_or((agent_num == 0), (entity_num == 0)) #[bs,t]
        seen_num=obs_mask.sum(3) #[bs,t,ne]
        vis_percent = seen_num[:,:,:self.n_agents].sum(2)/agent_num/entity_num #[bs,ts]
        vis_percent = vis_percent.masked_fill(invalid_frame, 0.0)
        t_length = th.logical_not(invalid_frame).sum(1) #[bs]
        visibility = (vis_percent.sum(1)/t_length).detach().cpu().numpy()
        t_length0 = th.logical_not(invalid_frame)[:,:10].sum(1)
        t_length1 = th.logical_not(invalid_frame)[:,10:].sum(1)
        visibility0 = (vis_percent[:,:10].sum(1)/t_length0).detach().cpu().numpy()
        visibility1 = (vis_percent[:,10:].sum(1)/t_length1).detach().cpu().numpy()
        return visibility, visibility0, visibility1


def env_worker(args, remote, entity_scheme, env_fn):
    # Make environment
    env = env_fn.x()

    if args.entity_dropout:
        pre_entity_dropout = None
        pre_entity_combination = None
        pre_entity_order = None
        dropout_timestep = 0
        temporalcnt = {}

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions, attenion_values = data
            # Take a step in the environment
            reward, terminated, additional, env_info = env.step(actions)
            send_dict = {
                "avail_actions": env.get_avail_actions(args.n_agents),
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info,
                "additional": additional
            }
        
            if entity_scheme: # REFIL / CAMA
                masks = env.get_masks()
                if len(masks) == 2:
                    obs_mask, entity_mask = masks
                    gt_mask = None
                else:
                    obs_mask, entity_mask, gt_mask = masks
                send_dict["obs_mask"] = obs_mask
                send_dict["entity_mask"] = entity_mask
                if gt_mask is not None:
                    send_dict["gt_mask"] = gt_mask
                send_dict["entities"] = process_entities(args, env.get_env_info(args), env.get_state())

                if args.entity_dropout and (args.dropout_type == "damoed" or args.dropout_type == "random" or args.dropout_type == "flickerfusion" or args.dropout_type == "attention"):
                    dropout_obs_mask, dropout_timestep, entity_order, entity_dropout, entity_combination, temporalcnt = process_EntityDropoutAtten(
                        obs_mask, args, dropout_timestep, pre_entity_order, pre_entity_dropout, pre_entity_combination, env.get_env_info(args), env.get_state(), temporalcnt, attention_values=attenion_values)

                    pre_entity_order = entity_order
                    pre_entity_dropout = entity_dropout
                    pre_entity_combination = entity_combination
                    send_dict["obs_mask"] = [dropout_obs_mask]

            else: # QMIX
                # Data for the next timestep needed to pick an action
                send_dict["state"] = process_state(args, env.get_env_info(args), env.get_state())
                if args.entity_dropout:
                    dropout_obs, dropout_timestep, entity_order, entity_dropout, entity_combination, temporalcnt = process_EntityDropoutObs(
                        args, args.agentnames[0], dropout_timestep, pre_entity_order, pre_entity_dropout, pre_entity_combination, env.get_env_info(args), env.get_state(), temporalcnt)
                    pre_entity_order = entity_order
                    pre_entity_dropout = entity_dropout
                    pre_entity_combination = entity_combination
                    send_dict["obs"] = dropout_obs
                else:
                    send_dict["obs"] = process_agent_obs(args, args.agentnames[0], env.get_env_info(args), env.get_state(), None)
            remote.send(send_dict)
        elif cmd == "reset":
            env.reset(**data)
            if entity_scheme: # REFIL / CAMA
                masks = env.get_masks()
                if len(masks) == 2:
                    obs_mask, entity_mask = masks
                    gt_mask = None
                else:
                    obs_mask, entity_mask, gt_mask = masks
                send_dict = {
                    "entities": process_entities(args, env.get_env_info(args), env.get_state()),
                    "avail_actions": env.get_avail_actions(args.n_agents),
                    "obs_mask": obs_mask,
                    "entity_mask": entity_mask
                }

                if args.entity_dropout and (args.dropout_type == "damoed" or args.dropout_type == "random" or args.dropout_type == "flickerfusion"):
                    dropout_obs_mask, dropout_timestep, entity_order, entity_dropout, entity_combination, temporalcnt = process_EntityDropoutAtten(
                        obs_mask, args, dropout_timestep, pre_entity_order, pre_entity_dropout, pre_entity_combination, env.get_env_info(args), env.get_state(), temporalcnt)
                    pre_entity_order = entity_order
                    pre_entity_dropout = entity_dropout
                    pre_entity_combination = entity_combination
                    send_dict["obs_mask"] = [dropout_obs_mask]

                if gt_mask is not None:
                    send_dict["gt_mask"] = gt_mask
            else: # QMIX
                if args.entity_dropout:
                    dropout_obs, dropout_timestep, entity_order, entity_dropout, entity_combination, temporalcnt = process_EntityDropoutObs(
                        args, args.agentnames[0], dropout_timestep, pre_entity_order, pre_entity_dropout, pre_entity_combination, env.get_env_info(args), env.get_state(), temporalcnt)
                    pre_entity_order = entity_order
                    pre_entity_dropout = entity_dropout
                    pre_entity_combination = entity_combination
                    send_dict = {
                        "state": [process_state(args, env.get_env_info(args), env.get_state())],
                        "avail_actions": [env.get_avail_actions(args.n_agents)],
                        "obs": [dropout_obs]
                    }
                else:
                    send_dict = {
                        "state": [process_state(args, env.get_env_info(args), env.get_state())],
                        "avail_actions": [env.get_avail_actions(args.n_agents)],
                        "obs": [process_agent_obs(args, args.agentnames[0], env.get_env_info(args), env.get_state(), None)]
                    }
            remote.send(send_dict)

        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info(data))
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        # TODO: unused now?
        # elif cmd == "agg_stats":
        #     agg_stats = env.get_agg_stats(data)
        #     remote.send(agg_stats)
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

