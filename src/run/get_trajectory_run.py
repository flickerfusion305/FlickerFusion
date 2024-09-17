import datetime
from functools import partial
from math import ceil
from re import A
import imageio
import os
import os.path as osp
import numpy as np
import pickle
import pprint
import time
import json
import threading
from copy import deepcopy
import torch as th
from numpy.random import RandomState
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath, basename, join, splitext

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from envs import s_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
import yaml


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}_".format(_config["seed"])
    unique_token += "{}".format(args.name)
    if args.dropout_flicker: 
        unique_token += "_ff"
    if args.entity_dropout_train: 
        unique_token += "_train"
    if args.entity_dropout_inference: 
        unique_token += "_inf"
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", args.tb_dirname)
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    if args.evaluate_multi_model:
        model_dir = deepcopy(args.checkpoint_path.split(","))
        sight_ranges = deepcopy(args.env_args["sight_range_kind"])
        args.evaluate=True
        rm_dic={}
        sc2_sr_kind_dict = {0:3, 1:3, 2:9, 3:9}
        sc2_group_kind_dict = {0:True, 1:False, 2:True, 3:False}
        for sr in sight_ranges:
            rmm = []
            for md in model_dir:
                args.checkpoint_path = osp.join(args.checkpoint_prefix, md)
                if "sc2" in args.env:
                    args.env_args["sight_range"] = sc2_sr_kind_dict[sr]
                    args.env_args["divide_group"] = sc2_group_kind_dict[sr]
                else:
                    args.env_args["sight_range_kind"] = sr
                rm = run_sequential(args=args, logger=logger)
                print("model:", md, ", sight range:", sr, ", return mean:", rm)
                rm_dic.update({(md, sr):rm})
                rmm.append(rm)
            # Clean up after finishing
                # print("Exiting Main")

                # print("Stopping all threads")
                for t in threading.enumerate():
                    if t.name != "MainThread":
                        # print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
                        t.join(timeout=1)
                        # print("Thread joined")
            rm_dic.update({('Average', sr):np.mean(rmm)})
        # for (md, sr), rm in rm_dic.items():
            # print("model:", md, ", sight range:", sr, ", return mean:", rm)
    else:
        run_sequential(args=args, logger=logger)
        # print("Exiting Main")

        # print("Stopping all threads")
        for t in threading.enumerate():
            if t.name != "MainThread":
                # print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
                t.join(timeout=1)
                # print("Thread joined")

    # print("Exiting script")

    # Making sure framework really exits
    try: 
        os._exit(os.EX_OK)
    except: 
        pass


def run_sequential(args, logger):
    # Init runner so we can get env info
    if 'entity_scheme' in args.env_args:
        args.entity_scheme = args.env_args['entity_scheme']
    else:
        args.entity_scheme = False

    if ('sc2custom' in args.env):
        rs = RandomState(0)
        args.env_args['scenario_dict'] = s_REGISTRY[args.scenario](rs=rs)
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    if not args.entity_scheme: # For Vanilla QMIX
        args.state_shape = env_info["state_shape"]
        # Default/Base scheme
        
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {"agents": args.n_agents}

        if 'masks' in env_info:
            # masks that identify what part of observation/state spaces correspond to each entity
            args.obs_masks, args.state_masks = env_info['masks']
        if 'unit_dim' in env_info:
            args.unit_dim = env_info['unit_dim']
    else: # For REFIL / CAMA?
        args.entity_shape = env_info["entity_shape"]
        args.n_entities = env_info["n_entities"]
        args.gt_mask_avail = env_info.get("gt_mask_avail", False)
        # Entity scheme
        scheme = {
            "entities": {"vshape": env_info["entity_shape"], "group": "entities"},
            "obs_mask": {"vshape": env_info["n_entities"], "group": "entities", "dtype": th.uint8},
            "entity_mask": {"vshape": env_info["n_entities"], "dtype": th.uint8},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        if args.gt_mask_avail:
            scheme["gt_mask"] = {"vshape": env_info["n_entities"], "group": "agents", "dtype": th.uint8}
        elif args.env == "particle" and args.env_args["scenario_id"] == "resource_collection.py" and args.env_args["comm_sr"] is not None:
            scheme["gt_mask"] = {"vshape": env_info["n_entities"], "group": "entities", "dtype": th.uint8}
        if args.use_msg:
            if args.no_summary:
                args.msg_dim=args.attn_embed_dim
            scheme["self_message"] = {"vshape":(args.msg_dim,), "group": "agents"}
            scheme["head_message"] = {"vshape":(args.msg_dim,), "group": "agents"}

        groups = {
            "agents": args.n_agents,
            "entities": args.n_entities
        }

    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        if type(args.load_step) == list:
            assert args.evaluate and not args.evaluate_multi_model
            print('eval multi-time step..')
        else:
            args.load_step = [args.load_step]
        for load_step in args.load_step:
            timesteps = []
            timestep_to_load = 0

            if not os.path.isdir(args.checkpoint_path):
                logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
                return
            
            # Go through all files in args.checkpoint_path
            for name in os.listdir(args.checkpoint_path):
                full_name = os.path.join(args.checkpoint_path, name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            if load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - load_step))

            model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

            logger.console_logger.info("Loading model from {}".format(model_path))
            learner.load_models(model_path, evaluate=args.evaluate)
            runner.t_env = timestep_to_load

            if args.evaluate or args.save_replay:
                rm = evaluate_sequential(args, runner, logger, load_time_step = timestep_to_load)
        args.load_step=0
        #return rm -evaluate_multi_model할때 아니면 무의미

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    insert_buffer_num = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Getting Episode buffer for {} episodes".format(args.save_interval))

    while insert_buffer_num <= args.save_interval:

        # Run for a whole episode at a time
        if args.runner == 'total_episode' and buffer.can_sample(runner.batch_size):
            replay_batch=buffer.sample(runner.batch_size)
            episode_batch = runner.run(test_mode=False, replay_batch=replay_batch)
        else:
            episode_batch = runner.run(test_mode=False)

        buffer.insert_episode_batch(episode_batch)
        insert_buffer_num += runner.batch_size
        if args.save_memory and insert_buffer_num == args.save_interval:
            # map_name = args.get('scenario', None)
            map_name = args.env_args['scenario_id'][:-3]
            if map_name is None:
                map_name = args.env_args["map_name"]
            os.makedirs(os.path.join(args.local_results_path, "replay_memory", args.unique_token+"_"+map_name), exist_ok=True)
            save_path = os.path.join(args.local_results_path, "replay_memory", args.unique_token+"_"+map_name, "{:07d}".format(insert_buffer_num)+".pkl")
            with open(save_path, "wb") as f:
                th.save(buffer, f)
            print("Replay saved to", save_path)


        # if buffer.can_sample(args.batch_size):
        #     for _ in range(args.training_iters):
        #         # if args.mi_message and args.club_mi:
        #         #     for _ in range(args.club_ratio):
        #         #         episode_sample = buffer.sample(args.batch_size)

        #         #         # Truncate batch to only filled timesteps
        #         #         max_ep_t = episode_sample.max_t_filled()
        #         #         episode_sample = episode_sample[:, :max_ep_t]

        #         #         if episode_sample.device != args.device:
        #         #             episode_sample.to(args.device)
        #         #         learner.train_logq(episode_sample, runner.t_env, episode)
        #         episode_sample = buffer.sample(args.batch_size)

        #         # Truncate batch to only filled timesteps
        #         max_ep_t = episode_sample.max_t_filled()
        #         episode_sample = episode_sample[:, :max_ep_t]

        #         if episode_sample.device != args.device:
        #             episode_sample.to(args.device)

        #         learner.train(episode_sample, runner.t_env, episode)
        
        # Execute test runs once in a while
        # n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        # if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

        #     logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
        #     logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
        #         time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
        #     last_time = time.time()

        #     last_test_T = runner.t_env
        #     #repeat inference 3 times
        #     for empirical_study_n in range(3): 
        #         logger.console_logger.info("Testing domain_{}".format(empirical_study_n))
        #         for _ in range(n_test_runs):
        #             if args.runner == 'total_episode':
        #                 replay_batch=buffer.sample(runner.batch_size)
        #                 runner.run(test_mode=True, replay_batch=replay_batch, domain_n = empirical_study_n)
        #             else:
        #                 runner.run(test_mode=True, domain_n = empirical_study_n)
        
        # if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or
        #                         model_save_time == 0 or
        #                         runner.t_env > args.t_max):
        #     model_save_time = runner.t_env
        #     save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
        #     #"results/models/{}".format(unique_token)
        #     os.makedirs(save_path, exist_ok=True)
        #     logger.console_logger.info("Saving models to {}".format(save_path))

        #     # learner should handle saving/loading -- delegate actor save/load to mac,
        #     # use appropriate filenames to do critics, optimizer states
        #     learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
            
    runner.close_env()
    logger.console_logger.info("Finished Training")


# TODO: Clean this up
def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    # assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
    #     "need to use replay buffer if running in parallel mode!"

    # assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    # if config["learner"] == "coma":
    #    assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
    #    (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
    #        "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config
