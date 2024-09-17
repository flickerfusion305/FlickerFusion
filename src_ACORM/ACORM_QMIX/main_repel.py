# import argparse
# from run import Runner
# import torch

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX,VDN and ACORM in SMAC environment")
#     parser.add_argument("--max_train_steps", type=int, default=5000000, help="Maximum number of training steps")
#     parser.add_argument("--evaluate_freq", type=int, default=10000, help="Evaluate the policy every 'evaluate_freq' steps")
#     parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
#     # parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")
#     parser.add_argument("--algorithm", type=str, default="ACORM", help="QMIX or VDN")
#     parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
#     parser.add_argument("--epsilon_decay_steps", type=float, default=80000, help="How many steps before the epsilon decays to the minimum")
#     parser.add_argument("--epsilon_min", type=float, default=0.02, help="Minimum epsilon")
#     parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
#     parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
#     parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
#     parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
#     parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
#     parser.add_argument("--hyper_layers_num", type=int, default=2, help="The number of layers of hyper-network")
#     parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
#     parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
#     parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
#     parser.add_argument("--use_hard_update", type=bool, default=False, help="Whether to use hard update")
#     parser.add_argument("--use_lr_decay", type=bool, default=True, help="Whether to use learning rate decay")
#     parser.add_argument("--lr_decay_steps", type=int, default=500, help="every steps decay steps")
#     parser.add_argument("--lr_decay_rate", type=float, default=0.98, help="learn decay rate")
#     parser.add_argument("--target_update_freq", type=int, default=100, help="Update frequency of the target network")
#     parser.add_argument("--tau", type=float, default=0.005, help="If use soft update")
#     parser.add_argument("--seed", type=int, default=123, help="random seed")
#     parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--env_name', type=str, default='MMM2')  #['3m', '8m', '2s3z']

#     # plot
#     parser.add_argument("--sns_plot", type=bool, default=False, help="Whether to use seaborn plot")
#     parser.add_argument("--tb_plot", type=bool, default=True, help="Whether to use tensorboard plot")
    
#     # RECL
#     parser.add_argument("--agent_embedding_dim", type=int, default=128, help="The dimension of the agent embedding")
#     parser.add_argument("--role_embedding_dim", type=int, default=64, help="The dimension of the role embedding")
#     parser.add_argument("--use_ln", type=bool, default=False, help="Whether to use layer normalization")
#     parser.add_argument("--cluster_num", type=int, default=int(3), help="the cluster number of knn")
#     parser.add_argument("--recl_lr", type=float, default=8e-4, help="Learning rate")
#     parser.add_argument("--agent_embedding_lr", type=float, default=1e-3, help="agent_embedding Learning rate")
#     parser.add_argument("--train_recl_freq", type=int, default=200, help="Train frequency of the RECL network")
#     parser.add_argument("--role_tau", type=float, default=0.005, help="If use soft update")
#     parser.add_argument("--multi_steps", type=int, default=1, help="Train frequency of the RECL network")
#     parser.add_argument("--role_mix_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the QMIX network")

#     # attention
#     parser.add_argument("--att_dim", type=int, default=128, help="The dimension of the attention net")
#     parser.add_argument("--att_out_dim", type=int, default=64, help="The dimension of the attention net")
#     parser.add_argument("--n_heads", type=int, default=4, help="multi-head attention")
#     parser.add_argument("--soft_temperature", type=float, default=1.0, help="multi-head attention")
#     parser.add_argument("--state_embed_dim", type=int, default=64, help="The dimension of the gru state net")

#     # save path
#     parser.add_argument('--save_path', type=str, default='./result/acorm')
#     parser.add_argument('--model_path', type=str, default='./model/acorm')

    # args = parser.parse_args()
    # args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps
    # torch.multiprocessing.set_start_method('spawn')

    # runner = Runner(args)
    # runner.run()

import os
import torch
from run import Runner
import yaml
import json
import argparse

class Args:
    def __init__(self, config_path=None):
        self.max_train_steps = 3000000
        self.evaluate_freq = 20000
        self.evaluate_times = 15
        self.algorithm = "ACORM"
        self.epsilon = 1.0
        self.epsilon_decay_steps = 5000000
        self.epsilon_min = 0.05
        self.buffer_size = 5000
        self.batch_size = 32
        self.lr = 0.0003
        self.gamma = 0.99
        self.qmix_hidden_dim = 32
        self.hyper_hidden_dim = 128
        self.hyper_layers_num = 2
        self.rnn_hidden_dim = 64
        self.mlp_hidden_dim = 64
        self.add_last_action = True
        self.use_hard_update = False
        self.use_lr_decay = True
        self.lr_decay_steps = 500
        self.lr_decay_rate = 0.98
        self.target_update_freq = 200
        self.tau = 0.005 # Regularize a param update between param and target_param(see soft_update_params from acorm.py)
        self.seed = 123456
        self.device = 'cuda:0'

        # plot
        self.sns_plot = True
        self.tb_plot = True

        # RECL
        self.agent_embedding_dim = 128
        self.role_embedding_dim = 64
        self.use_ln = False
        self.cluster_num = 1
        self.recl_lr = 8e-4
        self.agent_embedding_lr = 1e-3
        self.train_recl_freq = 200
        self.role_tau = 0.005
        self.multi_steps = 1
        self.role_mix_hidden_dim = 64

        # attention
        self.att_dim = 128
        self.att_out_dim = 128
        self.n_heads = 4
        self.soft_temperature = 1.0
        self.state_embed_dim = 64

        self.env_name = ""
        # Derived arguments
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps

        # Model saving
        self.save_model_interval = 100000
        
        if config_path:
            self.load_from_yaml(config_path)

        #Render
        self.save_gif = False
        self.save_gif_name = ""
        self.render = False
        self.evaluate = False
        self.eval_domain = 1 #0, 1, 2 for in-domain, ood1, ood2
        self.load_model_path = "example_path/env_seed123456_model"
        self.load_model_name = "env_seed123456"

    def set_save_paths(self, env_name = "mpe_fluid"): 
        # save path
        self.env_name = env_name
        self.save_path = f'./results/sacred/acorm/{self.env_name}/{self.env_name}_seed{self.seed}'
        self.model_path = f'./results/models/acorm/{self.env_name}/{self.env_name}_seed{self.seed}'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)

    def load_from_yaml(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            for key, value in config.items():
                setattr(self, key, value)
    
    def save_to_json(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(vars(self), file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123456, help='Random seed for training')
    args_cmd = parser.parse_args()

    env = "repel"
    args = Args(config_path=f'./src_ACORM/ACORM_QMIX/config/envs/{env}_mpev2.yaml')
    args.seed = args_cmd.seed
    args.set_save_paths(env)
    torch.multiprocessing.set_start_method('spawn')
    runner = Runner(args)
    args.save_to_json(f'{args.save_path}/config')
    runner.run()


