import torch
import numpy as np
from algorithm.vdn_qmix import VDN_QMIX
from algorithm.acorm import ACORM_Agent
from util.replay_buffer import ReplayBuffer
# from smac.env import StarCraft2Env
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from envs.mpe_fluid.environment_entity import MultiAgentFluidEnv
from components.transforms import *
from tqdm import tqdm 

class Runner:
    def __init__(self, args):
        # Your existing initialization code
        self.args = args
        self.env_name = self.args.env_name
        self.seed = self.args.seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = MultiAgentFluidEnv(**self.args.env_args)
        self.env_info = self.env.get_env_info(self.args)
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        self.save_path =  args.save_path
        self.model_path = args.model_path

        # Create N agents
        if args.algorithm in ['QMIX', 'VDN']:
            self.agent_n = VDN_QMIX(self.args)
        elif args.algorithm == 'ACORM':
            self.agent_n = ACORM_Agent(self.args)
        self.replay_buffer = ReplayBuffer(self.args, self.args.buffer_size)

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.evaluate_reward = []
        self.total_steps = 0
        self.agent_embed_pretrain_epoch, self.recl_pretrain_epoch = 0, 0
        self.pretrain_agent_embed_loss, self.pretrain_recl_loss = [], []
        self.args.agent_embed_pretrain_epochs = 120 #originally 120
        self.args.recl_pretrain_epochs = 100 #originally 100

    def run(self):
        evaluate_num = -1  # Record the number of evaluations
        pbar = tqdm(total=self.args.max_train_steps, desc='Training Progress', unit='step')  # Create a progress bar
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_mpe(evaluate=False, episode_time=self.total_steps)  # Run an episode
            
            if self.agent_embed_pretrain_epoch < self.args.agent_embed_pretrain_epochs:
                if self.replay_buffer.current_size >= self.args.batch_size:
                    print("agent_embed_pretrain: ", self.agent_embed_pretrain_epoch)
                    self.agent_embed_pretrain_epoch += 1
                    agent_embedding_loss = self.agent_n.pretrain_agent_embedding(self.replay_buffer)
                    self.pretrain_agent_embed_loss.append(agent_embedding_loss.item())
            else:
                if self.recl_pretrain_epoch < self.args.recl_pretrain_epochs:
                    print("recl pretrain: ", self.recl_pretrain_epoch)
                    self.recl_pretrain_epoch += 1
                    recl_loss = self.agent_n.pretrain_recl(self.replay_buffer)
                    self.pretrain_recl_loss.append(recl_loss.item())
                    
                else:                                                          
                    self.total_steps += episode_steps
                    if self.replay_buffer.current_size >= self.args.batch_size:
                        self.agent_n.train(self.replay_buffer)  # Training
            pbar.update(episode_steps)  # Update progress bar

        self.evaluate_policy()
        pbar.close()  # Close the progress bar
        # Save model
        model_path = f'{self.model_path}/{self.env_name}_seed{self.seed}_'
        torch.save(self.agent_n.eval_Q_net, model_path + 'q_net.pth')
        torch.save(self.agent_n.RECL.role_embedding_net, model_path + 'role_net.pth')
        torch.save(self.agent_n.RECL.agent_embedding_net, model_path + 'agent_embed_net.pth')
        torch.save(self.agent_n.eval_mix_net.attention_net, model_path + 'attention_net.pth')
        torch.save(self.agent_n.eval_mix_net, model_path + 'mix_net.pth')
        self.env.close()

    def evaluate_policy(self):
        win_times = 0
        evaluate_reward = 0
        for time in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_mpe(evaluate=True, episode_time=time)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_reward.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))

        if self.args.sns_plot:
            sns.set_style('whitegrid')
            plt.figure()
            x_step = np.array(range(len(self.win_rates)))
            ax = sns.lineplot(x=x_step, y=np.array(self.win_rates).flatten(), label=self.args.algorithm)
            plt.ylabel('win_rates', fontsize=14)
            plt.xlabel(f'step*{self.args.evaluate_freq}', fontsize=14)
            plt.title(f'{self.args.algorithm} on {self.env_name}')
            plt.savefig(f'{self.save_path}/{self.env_name}_seed{self.seed}.jpg')

            # Save the win rates
            np.save(f'{self.save_path}/{self.env_name}_seed{self.seed}.npy', np.array(self.win_rates))
            np.save(f'{self.save_path}/{self.env_name}_seed{self.seed}_return.npy', np.array(self.evaluate_reward))
        
    def run_episode_mpe(self, evaluate=False, episode_time=None):
        # Modified method for running an episode in Multi-Agent Particle Environment
        episode_reward = 0
        self.env.reset()
        
        self.agent_n.eval_Q_net.rnn_hidden = None
        if self.args.algorithm == 'ACORM':
            self.agent_n.RECL.agent_embedding_net.rnn_hidden = None

        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        for episode_step in range(self.args.episode_limit):
            obs_n = process_agent_obs(self.args, self.args.agentnames[0], self.env.get_env_info(self.args), self.env.get_state())
            s = process_agent_obs(self.args, self.args.agentnames[0], self.env.get_env_info(self.args), self.env.get_state())
            avail_a_n = self.env.get_avail_actions(self.args.N)  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            if self.args.algorithm == 'ACORM':
                role_embedding = self.agent_n.get_role_embedding(obs_n, last_onehot_a_n)
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, role_embedding, avail_a_n, epsilon)
            else:
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)

            r, done, _, info = self.env.step(a_n)  # Take a step, _ is a position for additional data from mpev2
            episode_reward += r

            if not evaluate:
                """"
                When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                dw means dead or win,there is no next state s';
                but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

        if not evaluate:
            obs_n = process_agent_obs(self.args, self.args.agentnames[0], self.env.get_env_info(self.args), self.env.get_state())
            s = process_agent_obs(self.args, self.args.agentnames[0], self.env.get_env_info(self.args), self.env.get_state())
            avail_a_n = self.env.get_avail_actions(self.args.N)
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n, last_onehot_a_n)

        return None, episode_reward, episode_step + 1


# class Runner:
#     def __init__(self, args):
#         self.args = args
#         self.env_name = self.args.env_name
#         self.seed = self.args.seed
#         # Set random seed
#         np.random.seed(self.seed)
#         torch.manual_seed(self.seed)
#         # Create env
#         # self.env = StarCraft2Env(map_name=self.env_name, seed=self.seed)
#         self.env = MultiAgentFluidEnv(**self.args.env_args)
#         self.env_info = self.env.get_env_info(self.args)
#         self.args.N = self.env_info["n_agents"]  # The number of agents
#         self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
#         self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
#         self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
#         self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
#         print("number of agents={}".format(self.args.N))
#         print("obs_dim={}".format(self.args.obs_dim))
#         print("state_dim={}".format(self.args.state_dim))
#         print("action_dim={}".format(self.args.action_dim))
#         print("episode_limit={}".format(self.args.episode_limit))
#         self.save_path =  args.save_path
#         self.model_path = args.model_path

# #         from tensorboardX import SummaryWriter
# #         time_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# #         self.writer = SummaryWriter(log_dir='./result/tb_logs/{}/{}/{}_seed_{}_{}'.format(self.args.algorithm, self.env_name, self.env_name, self.seed,time_path))

#         # Create N agents
#         if args.algorithm in ['QMIX', 'VDN']:
#             self.agent_n = VDN_QMIX(self.args)
#         elif args.algorithm == 'ACORM':
#             self.agent_n = ACORM_Agent(self.args)
#         self.replay_buffer = ReplayBuffer(self.args, self.args.buffer_size)

#         self.epsilon = self.args.epsilon  # Initialize the epsilon
#         self.win_rates = []  # Record the win rates
#         self.evaluate_reward = []
#         self.total_steps = 0
#         self.agent_embed_pretrain_epoch, self.recl_pretrain_epoch = 0, 0
#         self.pretrain_agent_embed_loss, self.pretrain_recl_loss = [], []
#         self.args.agent_embed_pretrain_epochs =120 #originally 120
#         self.args.recl_pretrain_epochs = 100 #originally 100

#     def run(self, ):
#         evaluate_num = -1  # Record the number of evaluations
#         while self.total_steps < self.args.max_train_steps:
#             if self.total_steps // self.args.evaluate_freq > evaluate_num:
#                 self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
#                 evaluate_num += 1

#             _, _, episode_steps = self.run_episode_mpe(evaluate=False, episode_time=self.total_steps)  # Run an episode
            
#             if self.agent_embed_pretrain_epoch < self.args.agent_embed_pretrain_epochs:
#                 if self.replay_buffer.current_size >= self.args.batch_size:
#                     print("agent_embed_pretrain: ", self.agent_embed_pretrain_epoch)
#                     self.agent_embed_pretrain_epoch += 1
#                     agent_embedding_loss = self.agent_n.pretrain_agent_embedding(self.replay_buffer)
#                     self.pretrain_agent_embed_loss.append(agent_embedding_loss.item())
#             else:
#                 if self.recl_pretrain_epoch < self.args.recl_pretrain_epochs:
#                     print("recl pretrain: ", self.recl_pretrain_epoch)
#                     self.recl_pretrain_epoch += 1
#                     recl_loss = self.agent_n.pretrain_recl(self.replay_buffer)
#                     self.pretrain_recl_loss.append(recl_loss.item())
                    
#                 else:                                                          
#                     self.total_steps += episode_steps
#                     if self.replay_buffer.current_size >= self.args.batch_size:
#                         self.agent_n.train(self.replay_buffer)  # Training
                    
#         self.evaluate_policy()
#          # save model
#         model_path = f'{self.model_path}/{self.env_name}_seed{self.seed}_'
#         torch.save(self.agent_n.eval_Q_net, model_path + 'q_net.pth')
#         torch.save(self.agent_n.RECL.role_embedding_net, model_path + 'role_net.pth')
#         torch.save(self.agent_n.RECL.agent_embedding_net, model_path+'agent_embed_net.pth')
#         torch.save(self.agent_n.eval_mix_net.attention_net, model_path+'attention_net.pth')
#         torch.save(self.agent_n.eval_mix_net, model_path+'mix_net.pth')
#         self.env.close()

#     def evaluate_policy(self, ):
#         win_times = 0
#         evaluate_reward = 0
#         for time in range(self.args.evaluate_times):
#             win_tag, episode_reward, _ = self.run_episode_mpe(evaluate=True, episode_time=time)
#             if win_tag:
#                 win_times += 1
#             evaluate_reward += episode_reward

#         # win_rate = win_times / self.args.evaluate_times
#         evaluate_reward = evaluate_reward / self.args.evaluate_times
#         # self.win_rates.append(win_rate)
#         self.evaluate_reward.append(evaluate_reward)
#         print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        
#         # if self.args.tb_plot:
#         #     self.writer.add_scalar('win_rate', win_rate, global_step=self.total_steps)
#         if self.args.sns_plot:
#             # # plot curve
#             sns.set_style('whitegrid')
#             plt.figure()
#             x_step = np.array(range(len(self.win_rates)))
#             ax = sns.lineplot(x=x_step, y=np.array(self.win_rates).flatten(), label=self.args.algorithm)
#             plt.ylabel('win_rates', fontsize=14)
#             plt.xlabel(f'step*{self.args.evaluate_freq}', fontsize=14)
#             plt.title(f'{self.args.algorithm} on {self.env_name}')
#             plt.savefig(f'{self.save_path}/{self.env_name}_seed{self.seed}.jpg')

#             # Save the win rates
#             np.save(f'{self.save_path}/{self.env_name}_seed{self.seed}.npy', np.array(self.win_rates))
#             np.save(f'{self.save_path}/{self.env_name}_seed{self.seed}_return.npy', np.array(self.evaluate_reward))
        
#     def run_episode_mpe(self, evaluate=False, episode_time=None):
#         # For those who review this code, I modified "run_episode_smac" from the original one.
#         # We do not have "win_tag" data for our envs. So I deleted all lines related to the win_tag value.
#         episode_reward = 0
#         self.env.reset()
        
#         self.agent_n.eval_Q_net.rnn_hidden = None
#         if self.args.algorithm == 'ACORM':
#             self.agent_n.RECL.agent_embedding_net.rnn_hidden = None

#         last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
#         for episode_step in range(self.args.episode_limit):
# #             obs_n = self.env.get_state()  # obs_n.shape=(N,obs_dim)
# #             s = self.env.get_state()  # s.shape=(state_dim,)
#             obs_n = process_agent_obs(self.args, self.args.agentnames[0], self.env.get_env_info(self.args), self.env.get_state())
#             s = process_agent_obs(self.args, self.args.agentnames[0], self.env.get_env_info(self.args), self.env.get_state())
#             avail_a_n = self.env.get_avail_actions(self.args.N)  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
#             epsilon = 0 if evaluate else self.epsilon
#             if self.args.algorithm == 'ACORM':
#                 role_embedding = self.agent_n.get_role_embedding(obs_n, last_onehot_a_n)
#                 a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, role_embedding, avail_a_n, epsilon)
#             else:
#                 a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)

#             r, done, _, info = self.env.step(a_n)  # Take a step, _ is a position for additional data from mpev2
#             # win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
#             episode_reward += r

#             if not evaluate:
#                 """"
#                     When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
#                     dw means dead or win,there is no next state s';
#                     but when reaching the max_episode_steps,there is a next state s' actually.
#                 """
#                 if done and episode_step + 1 != self.args.episode_limit:
#                     dw = True
#                 else:
#                     dw = False

#                 # Store the transition
#                 self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
#                 last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
#                 # obs_a_n_buffer[episode_step] = obs_n
#                 # Decay the epsilon
#                 self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

#             if done:
#                 break

#         if not evaluate:
#             # An episode is over, store obs_n, s and avail_a_n in the last step
# #             obs_n = self.env.get_state()
# #             s = self.env.get_state()
#             obs_n = process_agent_obs(self.args, self.args.agentnames[0], self.env.get_env_info(self.args), self.env.get_state())
#             s = process_agent_obs(self.args, self.args.agentnames[0], self.env.get_env_info(self.args), self.env.get_state())
#             avail_a_n = self.env.get_avail_actions(self.args.N)
#             self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n, last_onehot_a_n)
#         # There was "win_tag" for None
#         if evaluate:
#             print(f"eval running {episode_time}")
#         else:
#             print(f"train running {episode_time}")
#         return None, episode_reward, episode_step+1