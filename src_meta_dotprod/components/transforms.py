import numpy as np
import torch as th
import matplotlib.pyplot as plt

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

# For implementation dict state -> vector state

def process_agent_obs(args, agentname, env_info, state_global):
    onehot_dict = env_info["onehot_dict"]
    obs_info = env_info["obs_info"]
    entity_maxnum = env_info["entity_maxnum"]
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
            for enidx in range(entitynums[enname]): # entity idx - wise
                if enname == agentname and enidx == aidx: continue

                if f"{enname}-pos_{enidx}" not in state_global.keys():
                    curobs.append([0 for _ in range(totobssz)])
                    continue

                elif state_global[f"{enname}-pos_{enidx}"]["mask"]:
                    curobs.append([0 for _ in range(totobssz)])
                    continue

                enobs = create_onehot(onehotidx, onehotsz)
                for ot, osz in obs_info.items():
                    if f"{enname}-{ot}_{enidx}" in state_global.keys():
                        enobs += state_global[f"{enname}-{ot}_{enidx}"]["element"]
                    else:
                        enobs += [0 for _ in range(osz)]
                curobs.append(enobs)
        obs.append(np.concatenate(curobs))
    return obs


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

def create_onehot(idx,total):
    a = [0 for _ in range(total)]
    a[idx]=1
    return a