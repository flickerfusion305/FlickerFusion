import numpy as np
import torch as th
import matplotlib.pyplot as plt
import time

class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError

class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        # TODO: Check this shouldn't be here
        # assert vshape_in == (1,)
        return (self.out_dim,), th.float32





### Added transform functions includes: (1) state global dict -> obs, state, entities array (2) ITID (3) Entity Dropout

def process_agent_obs(args, agentname, env_info, state_global, cur_dropout):
    onehot_dict = env_info["onehot_dict"]
    obs_info = env_info["obs_info"]
    entity_maxnum = env_info["entity_maxnum"]
    maxobscnt_dict = env_info["maxobscnt_dict"]
    entitynames = list(onehot_dict.keys())
    entitynums = {name: entity_maxnum[name] for name in entitynames}
    totobssz = sum(obs_info.values()) + len(onehot_dict.keys())

    obs = []
    onehotsz = len(onehot_dict.keys())
    for aidx in range(entitynums[agentname]):
        curobs = []
        if f"{agentname}-pos_{aidx}" not in state_global.keys():
            curobs = np.zeros((env_info["obs_shape"]))
            obs.append(curobs)
            continue
        
        elif state_global[f"{agentname}-pos_{aidx}"]["mask"]:
            curobs = np.zeros((env_info["obs_shape"]))
            obs.append(curobs)
            continue

        onehotidx = onehot_dict[agentname]
        enobs = create_onehot(onehotidx, onehotsz)
        for ot, osz in obs_info.items():
            if f"{agentname}-{ot}_{aidx}" in state_global.keys():
                enobs += state_global[f"{agentname}-{ot}_{aidx}"]["element"]
            else:
                enobs += [0 for _ in range(osz)]
        curobs.append(enobs)
        
        for enname in entitynames: # entity type - wise
            onehotidx = onehot_dict[enname]
            curobscnt = 0
            if args.entity_dropout:
                maxobscnt = maxobscnt_dict[enname]
            for enidx in range(entitynums[enname]): # entity idx - wise
                if enname == agentname and enidx == aidx:
                    curobscnt += 1
                    continue
                
                if args.entity_dropout:
                    if enidx in cur_dropout[aidx][enname]: continue

                if f"{enname}-pos_{enidx}" not in state_global.keys():
                    if args.entity_dropout: continue
                    curobs.append([0 for _ in range(totobssz)])
                    continue

                elif state_global[f"{enname}-pos_{enidx}"]["mask"]:
                    if args.entity_dropout: continue
                    curobs.append([0 for _ in range(totobssz)])
                    continue

                enobs = create_onehot(onehotidx, onehotsz)
                for ot, osz in obs_info.items():
                    if f"{enname}-{ot}_{enidx}" in state_global.keys():
                        enobs += state_global[f"{enname}-{ot}_{enidx}"]["element"]
                    else:
                        enobs += [0 for _ in range(osz)]
                curobs.append(enobs)
                curobscnt += 1
                
            if args.entity_dropout:
                curobs.append([0 for _ in range(totobssz * (maxobscnt - curobscnt))])
        obs.append(np.concatenate(curobs))
    return obs

def getEntityDropout(args, entitynums, entitynames, env_info, cur_order, cur_comb, comb_idx, state_global, temporalcnt):
    cur_dropout = {}
    if args.dropout_type == "damoed":
        cnt = {} # cnts which entity idx has been dropped out
        for enname, enidxs in comb_idx.items():
            cnt[enname] = {}
            for enidx in enidxs:
                cnt[enname][enidx] = 0
        for aidx in range(entitynums['agent']): #agent idx
            if f"agent-pos_{aidx}" in state_global.keys(): # agent alive
                cur_dropout[aidx] = {}
                for enname in entitynames: # entity type wise
                    cur_dropout[aidx][enname] = set()
                    if enname in env_info["dropout_num"].keys(): # if entity type dropout
                        if cur_comb[enname] in env_info["dropout_num"][enname].keys(): # if entity type dropout meets number requirement
                            DOnum = env_info["dropout_num"][enname][cur_comb[enname]]
                            samples = [] # lists for sampling
                            for enidx in cur_order[enname]:
                                if enname == "agent" and enidx == aidx: continue # in case of self obs
                                samples.append((cnt[enname][enidx], enidx))
                            np.random.shuffle(samples) # random shuffle
                            samples = sorted(samples, key=lambda x: x[0]) # sort with respect to drop out count
                            for _, DOidx in samples[:DOnum]:
                                cur_dropout[aidx][enname].add(DOidx)
                                cnt[enname][DOidx] += 1

    elif args.dropout_type == "random":
        for aidx in range(entitynums['agent']): #agent idx
            if f"agent-pos_{aidx}" in state_global.keys(): # agent alive
                cur_dropout[aidx] = {}
                for enname in entitynames: # entity type wise
                    cur_dropout[aidx][enname] = set()
                    if enname in env_info["dropout_num"].keys(): # if entity type dropout
                        if cur_comb[enname] in env_info["dropout_num"][enname].keys(): # if entity type dropout meets number requirement
                            DOnum = env_info["dropout_num"][enname][cur_comb[enname]]
                            samples = [] # lists for sampling
                            for enidx in cur_order[enname]:
                                if enname == "agent" and enidx == aidx: continue # in case of self obs
                                samples.append(enidx)
                            np.random.shuffle(samples) # random shuffle
                            for DOidx in samples[:DOnum]:
                                cur_dropout[aidx][enname].add(DOidx)


    return cur_dropout

def init_temporalcnt(temporalcnt, env_info, state_global):
    onehot_dict = env_info["onehot_dict"]
    entity_maxnum = env_info["entity_maxnum"]
    entitynames = list(onehot_dict.keys())
    entitynums = {name: entity_maxnum[name] for name in entitynames}
    cur_comb = {name: 0 for name in entitynames}
    comb_idx = {name: [] for name in entitynames}

    for aidx in range(entitynums['agent']): #agent idx
        temporalcnt[aidx] = {}
        for enname in entitynames:
            temporalcnt[aidx][enname] = {}
            for enidx in range(entitynums[enname]):
                temporalcnt[aidx][enname][enidx] = 0
    
    return temporalcnt


def process_EntityDropoutObs(args, agentname, dropout_timestep, pre_order, pre_dropout, pre_combination, env_info, state_global, temporalcnt, test):
    dropout_timestep += 1
    onehot_dict = env_info["onehot_dict"]
    entity_maxnum = env_info["entity_maxnum"]
    entitynames = list(onehot_dict.keys())
    entitynums = {name: entity_maxnum[name] for name in entitynames}
    cur_comb = {name: 0 for name in entitynames}
    comb_idx = {name: [] for name in entitynames}

    obs = []

    idx = 0

    if test:
        args.dropout_type = args.test_dropout_type
    else:
        args.dropout_type = args.train_dropout_type

    for enname in entitynames: # entity type - wise
        idx = 0 # Not in atten_entity_dropout
        for enidx in range(entitynums[enname]): # entity idx - wise
            if f"{enname}-pos_{enidx}" not in state_global.keys(): # entity absent
                idx += 1
                continue
            if state_global[f"{enname}-pos_{enidx}"]["mask"]: # entity dead
                idx += 1
                continue
            cur_comb[enname] += 1
            comb_idx[enname].append(idx)
            idx += 1

    if pre_combination != cur_comb:
        cur_order = ITID(env_info, state_global)
        cur_dropout = getEntityDropout(args, entitynums, entitynames, env_info, cur_order, cur_comb, comb_idx, state_global, temporalcnt)
        obs = process_agent_obs(args, agentname, env_info, state_global, cur_dropout)
        dropout_timestep = 0

    elif args.dropout_flicker == True and dropout_timestep == args.flicker_interval:
        cur_order = pre_order
        cur_dropout = getEntityDropout(args, entitynums, entitynames, env_info, cur_order, cur_comb, comb_idx, state_global, temporalcnt)
        obs = process_agent_obs(args, agentname, env_info, state_global, cur_dropout)
        dropout_timestep = 0

    else: # No Change
        cur_dropout = pre_dropout
        cur_order = pre_order
        obs = process_agent_obs(args, agentname, env_info, state_global, cur_dropout)

    return obs, dropout_timestep, cur_order, cur_dropout, cur_comb, temporalcnt

def process_state(args, env_info, state_global):
    onehot_dict = env_info["onehot_dict"]
    obs_info = env_info["obs_info"]
    entity_maxnum = env_info["entity_maxnum"]
    entitynames = list(onehot_dict.keys())
    entitynums = {name: entity_maxnum[name] for name in entitynames}
    totobssz = sum(obs_info.values()) + len(onehot_dict.keys())

    state = []
    onehotsz = len(onehot_dict.keys())

    for enname in entitynames: # entity type - wise
        onehotidx = onehot_dict[enname]
        for enidx in range(entitynums[enname]): # entity idx - wise
            if f"{enname}-pos_{enidx}" not in state_global.keys():
                state.append([0 for _ in range(totobssz)])
                continue

            elif state_global[f"{enname}-pos_{enidx}"]["mask"]:
                state.append([0 for _ in range(totobssz)])
                continue

            enobs = create_onehot(onehotidx, onehotsz)
            for ot, osz in obs_info.items():
                if f"{enname}-{ot}_{enidx}" in state_global.keys():
                    enobs += state_global[f"{enname}-{ot}_{enidx}"]["element"]
                else:
                    enobs += [0 for _ in range(osz)]
            state.append(enobs)
    return np.concatenate(state)

def process_entities(args, env_info, state_global):
    onehot_dict = env_info["onehot_dict"]
    obs_info = env_info["obs_info"]
    entity_maxnum = env_info["entity_maxnum"]
    entitynames = list(onehot_dict.keys())
    entitynums = {name: entity_maxnum[name] for name in entitynames}
    onehotsz = len(onehot_dict.keys())
    totobssz = sum(obs_info.values()) + onehotsz

    entities = []
    if args.render_atten:
        plt.close("all")
        fig, ax = plt.subplots()
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        ax.set_aspect('equal')
    for enname in entitynames: # entity type - wise
        onehotidx = onehot_dict[enname]
        for enidx in range(entitynums[enname]): # entity idx - wise
            if f"{enname}-pos_{enidx}" not in state_global.keys(): # entity absent
                entities.append([0 for _ in range(totobssz)]) #empty obs
                # print("absent from entities:", len(entities)-1) #for debug
                continue
            if state_global[f"{enname}-pos_{enidx}"]["mask"]: # entity dead
                entities.append([0 for _ in range(totobssz)]) #empty obs
                # print("absent from entities:", len(entities)-1) #for debug
                continue
            if args.render_atten:
                pos = state_global[f"{enname}-pos_{enidx}"]["element"]
                ax.text(pos[0], pos[1], f"{len(entities)}", fontsize=12, ha='center', va='center', color='black')
            enobs = create_onehot(onehotidx, onehotsz)
            for ot, osz in obs_info.items():
                if f"{enname}-{ot}_{enidx}" in state_global.keys():
                    enobs += state_global[f"{enname}-{ot}_{enidx}"]["element"]
                else:
                    enobs += [0 for _ in range(osz)]
            entities.append(enobs)
    entities = np.array(entities)
    return entities

def ITID(env_info, state_global):
    onehot_dict = env_info["onehot_dict"]
    entity_maxnum = env_info["entity_maxnum"]
    entitynames = list(onehot_dict.keys())
    entitynums = {name: entity_maxnum[name] for name in entitynames}
    cur_comb = {name: 0 for name in entitynames}
    comb_idx = {name: [] for name in entitynames}
    comb_order = {name: [] for name in entitynames}

    idx = 0
    for enname in entitynames: # entity type - wise
        entitypositions = []
        idx = 0
        for enidx in range(entitynums[enname]): # entity idx - wise
            if f"{enname}-pos_{enidx}" not in state_global.keys(): # entity absent
                idx += 1
                continue
            if state_global[f"{enname}-pos_{enidx}"]["mask"]: # entity dead
                idx += 1
                continue
            cur_comb[enname] += 1
            comb_idx[enname].append(idx)
            entitypositions.append((state_global[f"{enname}-pos_{enidx}"]["element"] + [idx]))
            idx += 1
        entitypositions = sorted(entitypositions, key = lambda x: (x[0], x[1])) # define order w.r.t. position
        for orderidx, posidx in enumerate(entitypositions):
            comb_order[enname].append(posidx[2])
    #print("comb order:", comb_order)
    return comb_order # comb_order: Assume agents have defined order for each entity type w.r.t. position
    

def obs_mask_dropout(obs_mask, cur_dropout):
    cur_dropout = np.array(cur_dropout)
    cur_dropout_pad = np.pad(cur_dropout, ((0, obs_mask.shape[0]-cur_dropout.shape[0]), (0,0)), constant_values=0)
    obs_mask = np.logical_or(obs_mask, cur_dropout_pad)
    return obs_mask

def getEntityDropoutAtten(args, comb_idx, entitynums, entitysize, state_global, entitynames, env_info, cur_comb, cur_order, temporalcnt):
    entity_maxnum = env_info["entity_maxnum"]
    entitynames = list(env_info["onehot_dict"].keys())
    add_entity_index = {}

    sumidx = 0
    for idx, enname in enumerate(entitynames):
        add_entity_index[enname] = sumidx
        sumidx += entity_maxnum[enname]
    
    cur_dropout = []
    if args.dropout_type == "damoed":
        cnt = {} # cnts which entity idx has been dropped out
        for enname, enidxs in comb_idx.items():
            cnt[enname] = {}
            for enidx in enidxs:
                cnt[enname][enidx] = 0
        for aidx in range(entitynums['agent']): #agent idx
            agent_do = np.zeros((entitysize), dtype = np.uint8)
            if f"agent-pos_{aidx}" in state_global.keys(): # agent alive
                for enname in entitynames: # entity type wise
                    if enname in env_info["dropout_num"].keys(): # if entity type dropout
                        if cur_comb[enname] in env_info["dropout_num"][enname].keys(): # if entity type dropout meets number requirement
                            donum = env_info["dropout_num"][enname][cur_comb[enname]]
                            samples = [] # lists for sampling
                            for enidx in cur_order[enname]:
                                if enname == "agent" and enidx == aidx: continue # in case of self obs
                                samples.append((cnt[enname][enidx], enidx))
                            np.random.shuffle(samples) # random shuffle
                            samples = sorted(samples, key=lambda x: x[0]) # sort with respect to drop out count
                            for _, maskidx in samples[:donum]:
                                agent_do[add_entity_index[enname] + maskidx] = 1
                                cnt[enname][maskidx] += 1
            cur_dropout.append(agent_do)

    elif args.dropout_type == "random":
        for aidx in range(entitynums['agent']): #agent idx
            agent_do = np.zeros((entitysize), dtype = np.uint8)
            if f"agent-pos_{aidx}" in state_global.keys(): # agent alive
                for enname in entitynames: # entity type wise
                    if enname in env_info["dropout_num"].keys(): # if entity type dropout
                        if cur_comb[enname] in env_info["dropout_num"][enname].keys(): # if entity type dropout meets number requirement
                            donum = env_info["dropout_num"][enname][cur_comb[enname]]
                            samples = [] # lists for sampling
                            for enidx in comb_idx[enname]:
                                if enidx == aidx: continue # in case of self obs
                                samples.append(enidx)
                            np.random.shuffle(samples) # random shuffle
                            for maskidx in samples[:donum]:
                                agent_do[add_entity_index[enname] + maskidx] = 1
            cur_dropout.append(agent_do)
    
    return cur_dropout


def process_EntityDropoutAtten(obs_mask, args, dropout_timestep, pre_order, pre_dropout, pre_combination, env_info, state_global, temporalcnt, test):
    dropout_timestep += 1
    onehot_dict = env_info["onehot_dict"]
    entity_maxnum = env_info["entity_maxnum"]
    entitynames = list(onehot_dict.keys())
    entitynums = {name: entity_maxnum[name] for name in entitynames}
    entitysize = sum(entitynums.values())
    cur_comb = {name: 0 for name in entitynames}
    comb_idx = {name: [] for name in entitynames}

    if test:
        args.dropout_type = args.test_dropout_type
    else:
        args.dropout_type = args.train_dropout_type

    idx = 0
    for enname in entitynames: # entity type - wise
        idx = 0
        for enidx in range(entitynums[enname]): # entity idx - wise
            if f"{enname}-pos_{enidx}" not in state_global.keys(): # entity absent
                idx += 1
                continue
            if state_global[f"{enname}-pos_{enidx}"]["mask"]: # entity dead
                idx += 1
                continue
            cur_comb[enname] += 1
            comb_idx[enname].append(idx)
            idx += 1

    if pre_combination != cur_comb:
        cur_order = ITID(env_info, state_global) # Reset order
        cur_dropout = getEntityDropoutAtten(args, comb_idx, entitynums, entitysize, state_global, entitynames, env_info, cur_comb, cur_order, temporalcnt)
        obs_mask = obs_mask_dropout(obs_mask, cur_dropout)
        dropout_timestep = 0
    
    elif args.dropout_flicker == True and dropout_timestep == args.flicker_interval: # No order change, but Dropout change (rainbow)
        cur_order = pre_order
        cur_dropout = getEntityDropoutAtten(args, comb_idx, entitynums, entitysize, state_global, entitynames, env_info, cur_comb, cur_order, temporalcnt)
        obs_mask = obs_mask_dropout(obs_mask, cur_dropout)
        dropout_timestep = 0
    
    else: # No Order / Dropout Change
        cur_order = pre_order
        cur_dropout = pre_dropout
        obs_mask = obs_mask_dropout(obs_mask, cur_dropout)

    return obs_mask, dropout_timestep, cur_order, cur_dropout, cur_comb, temporalcnt

def create_onehot(idx,total):
    a = [0 for _ in range(total)]
    a[idx]=1
    return a