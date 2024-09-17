import copy
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.flex_qmix import FlexQMixer, LinearFlexQMixer
from modules.mixers.weighted_vdn import WVDNMixer
import torch as th
from torch.optim import RMSprop
from typing import List
import numpy as np

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.local_q_weight=None
        self.unnorm_local_q_weight = None
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = FlexQMixer(args)
            elif args.mixer == "lin_flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = LinearFlexQMixer(args)
            elif args.mixer == "wvdn":
                assert args.entity_scheme, "WVDNMixer only available with entity scheme"
                self.mixer = WVDNMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def _get_mixer_ins(self, batch, repeat_batch=1):
        if not self.args.entity_scheme:
            return (batch["state"][:, :-1].repeat(repeat_batch, 1, 1),
                    batch["state"][:, 1:])
        else:
            entities = []
            bs, max_t, ne, ed = batch["entities"].shape
            entities.append(batch["entities"])
            if self.args.entity_last_action:
                last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                        device=batch.device,
                                        dtype=batch["entities"].dtype)
                last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
                entities.append(last_actions)

            entities = th.cat(entities, dim=3)
            return ((entities[:, :-1].repeat(repeat_batch, 1, 1, 1),
                     batch["entity_mask"][:, :-1].repeat(repeat_batch, 1, 1)),
                    (entities[:, 1:],
                     batch["entity_mask"][:, 1:]))
    
    def local_q_hook(self, grad):
        self.unnorm_local_q_weight = grad.detach()
        self.local_q_weight = (grad / grad.sum(-1).unsqueeze(-1)).detach()
        return grad

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        if self.args.reward_standardization == True: #reward standardization from marllib
            rewards = rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        will_log = (t_env - self.log_stats_t >= self.args.learner_log_interval)

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.mixer.train()
        self.target_mac.eval()
        self.target_mixer.eval()

        if 'imagine' in self.args.agent:
            all_mac_out, groups = self.mac.forward(batch, t=None, imagine=True,
                                                   use_gt_factors=self.args.train_gt_factors,
                                                   use_rand_gt_factors=self.args.train_rand_gt_factors)
            # Pick the Q-Values for the actions taken by each agent
            rep_actions = actions.repeat(3, 1, 1, 1)
            all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # Remove the last dim

            mac_out, moW, moI = all_mac_out.chunk(3, dim=0)
            chosen_action_qvals, caqW, caqI = all_chosen_action_qvals.chunk(3, dim=0)
            caq_imagine = th.cat([caqW, caqI], dim=2)

            if will_log and self.args.test_gt_factors:
                gt_all_mac_out, gt_groups = self.mac.forward(batch, t=None, imagine=True, use_gt_factors=True)
                # Pick the Q-Values for the actions taken by each agent
                gt_all_chosen_action_qvals = th.gather(gt_all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # Remove the last dim

                gt_mac_out, gt_moW, gt_moI = gt_all_mac_out.chunk(3, dim=0)
                gt_chosen_action_qvals, gt_caqW, gt_caqI = gt_all_chosen_action_qvals.chunk(3, dim=0)
                gt_caq_imagine = th.cat([gt_caqW, gt_caqI], dim=2)
        else:
            mac_out = self.mac.forward(batch, t=None)
            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
            
        if self.args.__dict__.get("local_constraint", False) and self.args.ave_tot:
            bs, t, n_agent, n_action = mac_out[:,:-1].shape
            repeat_actions = actions.repeat(1,1,1,n_agent*n_action) #bs*t*n_agent*(n_agent*n_action)
            for i in range(n_agent):
                repeat_actions[:,:,i,n_action*i:n_action*(i+1)] = th.arange(n_action).reshape(1,1,n_action).repeat(bs,t,1)
            all_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=repeat_actions) #bs,t,n,n_action*n_agent

            

        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, t=None)

        avail_actions_targ = avail_actions
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions_targ[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if 'imagine' in self.args.agent:
                mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                global_action_qvals = self.mixer(chosen_action_qvals,
                                                 mix_ins)
                # don't need last timestep
                groups = [gr[:, :-1] for gr in groups]
                if will_log and self.args.test_gt_factors:
                    caq_imagine, ingroup_prop = self.mixer(
                        caq_imagine, mix_ins,
                        imagine_groups=groups,
                        ret_ingroup_prop=True)
                    gt_groups = [gr[:, :-1] for gr in gt_groups]
                    gt_caq_imagine, gt_ingroup_prop = self.mixer(
                        gt_caq_imagine, mix_ins,
                        imagine_groups=gt_groups,
                        ret_ingroup_prop=True)
                else:
                    caq_imagine = self.mixer(caq_imagine, mix_ins,
                                             imagine_groups=groups)
            else:
                mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                global_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
                #Warning: No implementation for entity_scheme if ave_tot==True
                if self.args.__dict__.get("local_constraint", False) and self.args.ave_tot:
                    ct_mix_ins, _ = self._get_mixer_ins(batch, repeat_batch=n_action*n_agent)
                    all_action_qvals = all_action_qvals.permute(0,3,1,2).reshape(bs*n_agent*n_action, t, n_agent) #bs,t,n,n_a*n_ac -> bs,n_a*n_ac, t,n
                    global_ct_qvals = self.mixer(all_action_qvals, ct_mix_ins)
                    global_ct_qvals = global_ct_qvals.reshape(bs, n_agent*n_action, t, 1)
                    global_ct_qvals = th.mean(global_ct_qvals, dim=1)

            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (global_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if 'imagine' in self.args.agent:
            im_prop = self.args.lmbda
            im_td_error = (caq_imagine - targets.detach())
            im_masked_td_error = im_td_error * mask
            im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
            loss = (1 - im_prop) * loss + im_prop * im_loss
        if self.args.__dict__.get("local_constraint", False):
            if self.args.ave_tot:
                target_qtot = global_ct_qvals
            else:
                target_qtot = global_action_qvals
            q_constraint = th.clip((target_qtot.detach() - chosen_action_qvals)**2 - self.args.q_tol, min=0.0)
            local_mask = mask.expand_as(q_constraint)
            q_constraint = q_constraint*local_mask
            local_loss = th.tensor(0.0)
            valid_mask = (q_constraint > 0.0).sum()
            if valid_mask > 0:
                local_loss = self.args.local_constraint_weight * q_constraint.sum() / valid_mask
            loss += local_loss

            
        # hk=chosen_action_qvals.register_hook(self.local_q_hook)
        # orig_req_grad = [p.requires_grad for p in self.mac.parameters()]
        # for p in self.mac.parameters():
        #     p.requires_grad=False
        # global_action_qvals.sum().backward(retain_graph=True)
        # for p, rg in zip(self.mac.parameters(), orig_req_grad):
        #     p.requires_grad = rg
        # hk.remove()
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        try:
            grad_norm=grad_norm.item()
        except:
            pass
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("max_local_weight", self.local_q_weight.max(-1)[0].mean().item(), t_env)
            # self.logger.log_stat("min_local_weight", self.local_q_weight.min(-1)[0].mean().item(), t_env)
            # self.logger.log_stat("delta_local_weight", (self.local_q_weight.max(-1)[0].mean()-self.local_q_weight.min(-1)[0].mean()).item(), t_env)
            # self.logger.log_stat("max_local_weight[0,10]", self.local_q_weight[0,10].max(-1)[0].item(), t_env)
            # self.logger.log_stat("min_local_weight[0,10]", self.local_q_weight[0,10].min(-1)[0].item(), t_env)
            # self.logger.log_stat("unnorm_max_local_weight", self.unnorm_local_q_weight.max(-1)[0].mean().item(), t_env)
            # self.logger.log_stat("unnorm_min_local_weight", self.unnorm_local_q_weight.min(-1)[0].mean().item(), t_env)
            # self.logger.log_stat("unnorm_max_local_weight[0,10]", self.unnorm_local_q_weight[0,10].max(-1)[0].item(), t_env)
            # self.logger.log_stat("unnorm_min_local_weight[0,10]", self.unnorm_local_q_weight[0,10].min(-1)[0].item(), t_env)
            
            self.logger.log_stat("min_local_taken_q", chosen_action_qvals.min(-1)[0].mean().item(), t_env)
            self.logger.log_stat("max_local_taken_q", chosen_action_qvals.max(-1)[0].mean().item(), t_env)
            # self.logger.log_stat("mean_local_taken_q", chosen_action_qvals.mean(-1)[0].mean().item(), t_env)
            # self.logger.log_stat("min_local_taken_q[0,10]", chosen_action_qvals[0,10].min(-1)[0].item(), t_env)
            # self.logger.log_stat("max_local_taken_q[0,10]", chosen_action_qvals[0,10].max(-1)[0].item(), t_env)
            # self.logger.log_stat("mean_local_taken_q[0,10]", chosen_action_qvals[0,10].mean(-1).item(), t_env)
            
            if 'imagine' in self.args.agent:
                self.logger.log_stat("im_loss", im_loss.item(), t_env)
            if self.args.test_gt_factors:
                self.logger.log_stat("ingroup_prop", ingroup_prop.item(), t_env)
                self.logger.log_stat("gt_ingroup_prop", gt_ingroup_prop.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (global_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            if self.args.__dict__.get("local_constraint", False):
                self.logger.log_stat("local_loss", local_loss.item(), t_env)
            if batch.max_seq_length == 2:
                # We are in a 1-step env. Calculate the max Q-Value for logging
                max_agent_qvals = mac_out_detach[:,0].max(dim=2, keepdim=True)[0]
                max_qtots = self.mixer(max_agent_qvals, batch["state"][:,0].unsqueeze(1))
                self.logger.log_stat("max_qtot", max_qtots.mean().item(), t_env)
            self.log_stats_t = t_env

    def train_meta_dotprod(self, buffer_list: List[ReplayBuffer], t_env: int, episode_num: int):
        alpha = self.args.meta_alpha
        LR = self.args.lr
        LR_G1G2 = self.args.lr_g1g2

        original_params = copy.deepcopy(self.params)

        for i in range(len(buffer_list) // 2):
            batch_1 = buffer_list[i*2].sample(self.args.batch_size)
            max_ep_t = batch_1.max_t_filled()
            batch_1 = batch_1[:, :max_ep_t]
            if batch_1.device != self.args.device:
                batch_1.to(self.args.device)
                
            self.optimiser.zero_grad()
            loss_1 = self._compute_loss(batch_1)
            loss_1.backward()
            G1 = [param.grad.clone() for param in self.params]
            
            batch_2 = buffer_list[i*2 + 1].sample(self.args.batch_size)
            max_ep_t = batch_2.max_t_filled()
            batch_2 = batch_2[:, :max_ep_t]
            if batch_2.device != self.args.device:
                batch_2.to(self.args.device)

            self.optimiser.zero_grad()
            loss_2 = self._compute_loss(batch_2)
            loss_2.backward()
            G2 = [param.grad.clone() for param in self.params]

            for param, g1 in zip(self.params, G1):
                param.data -= alpha * g1
            
            self.optimiser.zero_grad()
            loss_2_1 = self._compute_loss(batch_2)
            loss_2_1.backward()
            G2_1 = [param.grad.clone() for param in self.params]

            G1G2 = [(g2_1 - g2) / alpha for g2_1, g2 in zip(G2_1, G2)]

            G = [g1 + g2 + LR_G1G2 * g1g2 for g1, g2, g1g2 in zip(G1, G2, G1G2)]
            for param, g in zip(self.params, G):
                param.data -= LR * g
        
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("final_loss", loss_2.item(), t_env)
            self.logger.log_stat("grad_norm", th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip).item(), t_env)
            self.log_stats_t = t_env
    
    def train_SMLDG(self, source_domains: List[ReplayBuffer], t_env: int, episode_num: int):
        th.autograd.set_detect_anomaly(True)
        alpha = self.args.meta_alpha
        beta = self.args.meta_beta
        gamma = self.args.meta_gamma

        num_iter = len(source_domains)
        shuffled = np.random.permutation(num_iter)  # Shuffle the domain indices

        D_tilde = [source_domains[i] for i in shuffled]  # Reorder the domains according to the shuffled indices
        
        original_params = [p.clone() for p in self.params]
        Batch_0_tilde = D_tilde[0].sample(self.args.batch_size)
        max_ep_t = Batch_0_tilde.max_t_filled()
        Batch_0_tilde = Batch_0_tilde[:, :max_ep_t]
        if Batch_0_tilde.device != self.args.device:
            Batch_0_tilde.to(self.args.device)
            
        L = self._compute_loss(Batch_0_tilde)
        self.optimiser.zero_grad()
        L.backward(retain_graph=True)
        print("it goes well")
        grad_from_first_loss = [p.grad.clone() if p.grad is not None else None for p in self.params]
        
        with th.no_grad():
            for (p, g) in zip(self.params, grad_from_first_loss):
                if g is not None:
                    p.data -= alpha * g
                
        for i in shuffled[1:]:
            print(i)
            Batch = D_tilde[i].sample(self.args.batch_size)
            max_ep_t = Batch.max_t_filled()
            Batch_0_tilde = Batch_0_tilde[:, :max_ep_t]
            if Batch.device != self.args.device:
                Batch.to(self.args.device)
            L += beta * (self._compute_loss(Batch))
            
        for original_p, param in zip(original_params, self.params):
            param.data.copy_(original_p)
            
        self.optimiser.zero_grad()
        L.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", L.item(), t_env)
            self.logger.log_stat("grad_norm", th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip).item(), t_env)
            self.log_stats_t = t_env
            
    def _compute_loss(self, batch: EpisodeBatch):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        if self.args.reward_standardization == True:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.mac.init_hidden(batch.batch_size)
        self.mac.train()
        self.mixer.train()
        self.target_mac.eval()
        self.target_mixer.eval()

        mac_out = self.mac.forward(batch, t=None)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, t=None)

        avail_actions_targ = avail_actions
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        invalid_action_value = th.finfo(th.float32).min
        target_mac_out[avail_actions_targ[:, 1:] == 0] = invalid_action_value

        # Max over target Q-Values
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions_targ == 0] = invalid_action_value
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
            global_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (global_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / (mask.sum() + 1e-8)

        return loss
    
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path, evaluate=False):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if not evaluate:
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

# import copy
# from components.episode_buffer import EpisodeBatch
# from modules.mixers.vdn import VDNMixer
# from modules.mixers.qmix import QMixer
# from modules.mixers.flex_qmix import FlexQMixer, LinearFlexQMixer
# from modules.mixers.weighted_vdn import WVDNMixer
# import torch as th
# from torch.optim import RMSprop


# class QLearner:
#     def __init__(self, mac, scheme, logger, args):
#         self.args = args
#         self.mac = mac
#         self.logger = logger
#         self.local_q_weight = None
#         self.unnorm_local_q_weight = None
#         self.params = list(mac.parameters())

#         self.last_target_update_episode = 0

#         self.mixer = None
#         if args.mixer is not None:
#             if args.mixer == "vdn":
#                 self.mixer = VDNMixer()
#             elif args.mixer == "qmix":
#                 self.mixer = QMixer(args)
#             elif args.mixer == "flex_qmix":
#                 assert args.entity_scheme, "FlexQMixer only available with entity scheme"
#                 self.mixer = FlexQMixer(args)
#             elif args.mixer == "lin_flex_qmix":
#                 assert args.entity_scheme, "FlexQMixer only available with entity scheme"
#                 self.mixer = LinearFlexQMixer(args)
#             elif args.mixer == "wvdn":
#                 assert args.entity_scheme, "WVDNMixer only available with entity scheme"
#                 self.mixer = WVDNMixer(args)
#             else:
#                 raise ValueError("Mixer {} not recognised.".format(args.mixer))
#             self.params += list(self.mixer.parameters())
#             self.target_mixer = copy.deepcopy(self.mixer)

#         self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
#                                  weight_decay=args.weight_decay)

#         # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
#         self.target_mac = copy.deepcopy(mac)

#         self.log_stats_t = -self.args.learner_log_interval - 1

#     def _get_mixer_ins(self, batch, repeat_batch=1):
#         if not self.args.entity_scheme:
#             return (batch["state"][:, :-1].repeat(repeat_batch, 1, 1),
#                     batch["state"][:, 1:])
#         else:
#             entities = []
#             bs, max_t, ne, ed = batch["entities"].shape
#             entities.append(batch["entities"])
#             if self.args.entity_last_action:
#                 last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
#                                         device=batch.device,
#                                         dtype=batch["entities"].dtype)
#                 last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
#                 entities.append(last_actions)

#             entities = th.cat(entities, dim=3)
#             return ((entities[:, :-1].repeat(repeat_batch, 1, 1, 1),
#                      batch["entity_mask"][:, :-1].repeat(repeat_batch, 1, 1)),
#                     (entities[:, 1:],
#                      batch["entity_mask"][:, 1:]))

#     def local_q_hook(self, grad):
#         self.unnorm_local_q_weight = grad.detach()
#         self.local_q_weight = (grad / grad.sum(-1).unsqueeze(-1)).detach()
#         return grad

#     def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
#         # accumulation steps = self.args.accumulation_steps
#         accumulation_steps = 4  # Gradient accumulation steps
#         total_loss = 0.0
#         self.optimiser.zero_grad()

#         batch_size = batch.batch_size
#         mini_batch_size = batch_size // accumulation_steps

#         for i in range(accumulation_steps):
#             start_idx = i * mini_batch_size
#             end_idx = start_idx + mini_batch_size

#             mini_batch = batch[start_idx:end_idx]

#             # Get the relevant quantities
#             rewards = mini_batch["reward"][:, :-1]
#             if self.args.reward_standardization == True:  # reward standardization from marllib
#                 rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
#             actions = mini_batch["actions"][:, :-1]
#             terminated = mini_batch["terminated"][:, :-1].float()
#             mask = mini_batch["filled"][:, :-1].float()
#             mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
#             avail_actions = mini_batch["avail_actions"]

#             will_log = (t_env - self.log_stats_t >= self.args.learner_log_interval)

#             self.mac.init_hidden(mini_batch_size)
#             self.mac.train()
#             self.mixer.train()
#             self.target_mac.eval()
#             self.target_mixer.eval()

#             if 'imagine' in self.args.agent:
#                 all_mac_out, groups = self.mac.forward(mini_batch, t=None, imagine=True,
#                                                        use_gt_factors=self.args.train_gt_factors,
#                                                        use_rand_gt_factors=self.args.train_rand_gt_factors)
#                 rep_actions = actions.repeat(3, 1, 1, 1)
#                 all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)

#                 mac_out, moW, moI = all_mac_out.chunk(3, dim=0)
#                 chosen_action_qvals, caqW, caqI = all_chosen_action_qvals.chunk(3, dim=0)
#                 caq_imagine = th.cat([caqW, caqI], dim=2)

#                 if will_log and self.args.test_gt_factors:
#                     gt_all_mac_out, gt_groups = self.mac.forward(mini_batch, t=None, imagine=True, use_gt_factors=True)
#                     gt_all_chosen_action_qvals = th.gather(gt_all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)

#                     gt_mac_out, gt_moW, gt_moI = gt_all_mac_out.chunk(3, dim=0)
#                     gt_chosen_action_qvals, gt_caqW, gt_caqI = gt_all_chosen_action_qvals.chunk(3, dim=0)
#                     gt_caq_imagine = th.cat([gt_caqW, gt_caqI], dim=2)
#             else:
#                 mac_out = self.mac.forward(mini_batch, t=None)
#                 chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

#             if self.args.__dict__.get("local_constraint", False) and self.args.ave_tot:
#                 bs, t, n_agent, n_action = mac_out[:,:-1].shape
#                 repeat_actions = actions.repeat(1,1,1,n_agent*n_action)
#                 for j in range(n_agent):
#                     repeat_actions[:,:,j,n_action*j:n_action*(j+1)] = th.arange(n_action).reshape(1,1,n_action).repeat(bs,t,1)
#                 all_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=repeat_actions)

#             self.target_mac.init_hidden(mini_batch_size)
#             target_mac_out = self.target_mac.forward(mini_batch, t=None)

#             avail_actions_targ = avail_actions
#             target_mac_out = target_mac_out[:, 1:]

#             target_mac_out[avail_actions_targ[:, 1:] == 0] = -9999999

#             if self.args.double_q:
#                 mac_out_detach = mac_out.clone().detach()
#                 mac_out_detach[avail_actions_targ == 0] = -9999999
#                 cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
#                 target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
#             else:
#                 target_max_qvals = target_mac_out.max(dim=3)[0]

#             if self.mixer is not None:
#                 if 'imagine' in self.args.agent:
#                     mix_ins, targ_mix_ins = self._get_mixer_ins(mini_batch)
#                     global_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
#                     groups = [gr[:, :-1] for gr in groups]
#                     if will_log and self.args.test_gt_factors:
#                         caq_imagine, ingroup_prop = self.mixer(
#                             caq_imagine, mix_ins,
#                             imagine_groups=groups,
#                             ret_ingroup_prop=True)
#                         gt_groups = [gr[:, :-1] for gr in gt_groups]
#                         gt_caq_imagine, gt_ingroup_prop = self.mixer(
#                             gt_caq_imagine, mix_ins,
#                             imagine_groups=gt_groups,
#                             ret_ingroup_prop=True)
#                     else:
#                         caq_imagine = self.mixer(caq_imagine, mix_ins,
#                                                  imagine_groups=groups)
#                 else:
#                     mix_ins, targ_mix_ins = self._get_mixer_ins(mini_batch)
#                     global_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
#                     if self.args.__dict__.get("local_constraint", False) and self.args.ave_tot:
#                         ct_mix_ins, _ = self._get_mixer_ins(mini_batch, repeat_batch=n_action*n_agent)
#                         all_action_qvals = all_action_qvals.permute(0,3,1,2).reshape(bs*n_agent*n_action, t, n_agent)
#                         global_ct_qvals = self.mixer(all_action_qvals, ct_mix_ins)
#                         global_ct_qvals = global_ct_qvals.reshape(bs, n_agent*n_action, t, 1)
#                         global_ct_qvals = th.mean(global_ct_qvals, dim=1)

#                 target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)

#             targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

#             td_error = (global_action_qvals - targets.detach())
#             mask = mask.expand_as(td_error)
#             masked_td_error = td_error * mask
#             loss = (masked_td_error ** 2).sum() / mask.sum()

#             if 'imagine' in self.args.agent:
#                 im_prop = self.args.lmbda
#                 im_td_error = (caq_imagine - targets.detach())
#                 im_masked_td_error = im_td_error * mask
#                 im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
#                 loss = (1 - im_prop) * loss + im_prop * im_loss
#             if self.args.__dict__.get("local_constraint", False):
#                 if self.args.ave_tot:
#                     target_qtot = global_ct_qvals
#                 else:
#                     target_qtot = global_action_qvals
#                 q_constraint = th.clip((target_qtot.detach() - chosen_action_qvals)**2 - self.args.q_tol, min=0.0)
#                 local_mask = mask.expand_as(q_constraint)
#                 q_constraint = q_constraint * local_mask
#                 local_loss = th.tensor(0.0)
#                 valid_mask = (q_constraint > 0.0).sum()
#                 if valid_mask > 0:
#                     local_loss = self.args.local_constraint_weight * q_constraint.sum() / valid_mask
#                 loss += local_loss

#             total_loss += loss.item()
#             loss.backward()

#             if (i + 1) % accumulation_steps == 0:
#                 grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
#                 self.optimiser.step()
#                 self.optimiser.zero_grad()

#         if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
#             self._update_targets()
#             self.last_target_update_episode = episode_num

#         if t_env - self.log_stats_t >= self.args.learner_log_interval:
#             self.logger.log_stat("loss", total_loss / accumulation_steps, t_env)
#             self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
#             self.log_stats_t = t_env

#     def _update_targets(self):
#         self.target_mac.load_state(self.mac)
#         if self.mixer is not None:
#             self.target_mixer.load_state_dict(self.mixer.state_dict())
#         self.logger.console_logger.info("Updated target network")

#     def cuda(self):
#         self.mac.cuda()
#         self.target_mac.cuda()
#         if self.mixer is not None:
#             self.mixer.cuda()
#             self.target_mixer.cuda()

#     def save_models(self, path):
#         self.mac.save_models(path)
#         if self.mixer is not None:
#             th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
#         th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

#     def load_models(self, path, evaluate=False):
#         self.mac.load_models(path)
#         # Not quite right but I don't want to save target networks
#         self.target_mac.load_models(path)
#         if not evaluate:
#             if self.mixer is not None:
#                 self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
#             self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
