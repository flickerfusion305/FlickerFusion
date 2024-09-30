# FlickerFusion: Intra-trajectory Domain Generalizing Multi-agent Reinforcement Learning

## Contents

- [Project site](#project-site)
- [Experiment](#experiment)
    - [Installation](#installation)
    - [Run an experiment](#run-an-experiment)
    - [Save and load learned model](#save-and-load-learned-models)
    - [Render models](#render-models)
- [MPEv2](#mpev2)
    - [Environments](#environments)
    - [Customize config](#customize-config)
- [System specifications](#system-specifications)
- [Reference](#reference)

## Project site

Demo video renderings of trained models are available on [our project site](https://flickerfusion305.github.io/).

# Experiment

## Installation

Assuming you have Python 3 (version 3.8) installed, you can use Git to download and install it:
```
git clone https://github.com/flickerfusion305/FlickerFusion.git
cd FlickerFusion
pip install -r requirements.txt
```
The requirements.txt file is used to install the necessary packages for **most of methods** into a virtual environment.

> [!CAUTION]
> You can execute the experiments using **14 methods** (*Qmix*, *REFIL*, *ACORM*, *ODIS*, etc). While importing sources for these methods, there are requirement conflicts between `src_ACORM`, `src_ODIS`, and other sources. Therefore, you need to install the packages using **different requirements files** when running experiments with *ACORM* or *ODIS* (especially during the **training** stage). If you want to execute experiments with
> - methods except for *ACORM*, *ODIS* (train): use `pip install -r requirements.txt`
> - method *ACORM*: use `pip install -r requirements_acorm.txt`
> - method *ODIS* (train): use `pip install -r requirements_odis.txt`

## Run an experiment

You can run an experiment with this command format:
```
python3 main.py --env {env config} --method {method config}
```

For instance, if you want to execute an experiment with the *Tag* scenario, and *Qmix* method, you can use this command:
```
python3 main.py --env tag --method qmix
```

There are 6 environment configurations and 15 method configurations that you can use:
<table>
  <tr>

<!-- envs -->
  
| Scenario    | Config File          |
|-------------|----------------------|
| `tag`       | tag_mpev2.yaml       |
| `spread`    | spread_mpev2.yaml    |
| `guard`     | guard_mpev2.yaml     |
| `repel`     | repel_mpev2.yaml     |
| `adversary` | adversary_mpev2.yaml |
| `hunt`      | hunt_mpev2.yaml      |

<!-- methods -->

| Method                              | Config File                                  |
|-------------------------------------|----------------------------------------------|
| `qmix`                              | src/.../qmix.yaml                            |
| `qmix_atten`                        | src/.../qmix_atten.yaml                      |
| `qmix_dropout`                      | src/.../qmix_dropout.yaml                    |
| `qmix_atten_dropout`                | src/.../qmix_atten_dropout.yaml              |
| `refil`                             | src/.../refil.yaml                           |
| `cama`                              | src/.../cama_qmix_atten.yaml                 |
| `acorm`                             | src_ACORM/ACORM_QMIX/main_{envs' name}.py    |
| `updet`                             | src_UPDET/.../qmix_updet.yaml                |
| `dgmaml`                              | src_meta_lr/.../qmix_atten_DGMAML.yaml       |
| `mldg`                            | src_meta_lr/.../qmix_atten_MLDG.yaml         |
| `qmix_atten_daaged`                 | src_attention/.../qmix_atten.yaml            |
| `smldg`                             | src_meta_dotprod/.../qmix_atten_SMLDG.yaml   |
| `dotprod`                           | src_meta_dotprod/.../qmix_atten_DOTPROD.yaml |
| ODIS(train): `odis_train`           | src_ODIS/.../odis.yaml                       |
| ODIS(trajectory): `odis_trajectory` | src/.../qmix_get_trajectory.yaml             |

  </tr>
</table>

The config files are located in `{src directory}/config`.  
- `--env` refers to the config files in `{src directory}/config/envs`  
- `--method` refers to the config files in `{src directory}/config/algs`  

You can adjust the hyperparameters through the config files in `{src directory}/config/envs` or `{src directory}/config/algs` (except for *ACORM*).  

Each result of models / configurations will be stored in the `{src directory}/results` / `results` folder respectively.  

**NOTE:** For the *ACORM* method, you can adjust the configs in the `src_ACORM/.../main.py` file. Additionally, there are *main_{envs' name}.py* files in the `src_ACORM` directory, which have already been tuned for hyperparameters specific to each environment.

## Save and load learned models

### Saving models

You can save the trained models to the disk by setting `save_model = True` (it is set to `False` by default). The frequency of saving models can be adjusted using the `save_model_interval` configuration.  
Models will be saved in the `{src directory}/results` directory, under a folder named *models*. The directory corresponding to each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since the learning process began.  

Configs will be saved in the `results` directory, under the folder named *sacred*.

### Loading models

Learned models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep.

## Render models

You can render models through method config setting. Example below:
```yaml
...
checkpoint_path: "./results/models/guard_flicker"
load_step: 0
evaluate: True
render: True
eval_domain: 0 

#to render flicker
render_flicker: False

#to render attention matrix
render_atten: False
render_atten_ts: 0

#to save model eval as gif
save_gif: True
save_gif_name: "test"
...
```

The `render: True` setting allows rendering of models loaded using `checkpoint_path` with `load_step`. If `load_step: 0` is specified, it will automatically load the model from the directory with the greatest timestep number. Once the model is successfully loaded, `test_nepisode` episodes are run in test mode with `evaluate = True`. Use `eval_domain` to specify the domain for testing the model, either *in-domain* (train) or *out-of-domain* (OOD1/OOD2).  
Additionally, setting `render_flicker = True` enables the rendering of Flicker, which is part of our new method. The `render_atten: True` setting renders the attention matrix at a specific timestep, defined by `render_atten_ts`. If you want to save renderings, setting `save_gif: True` will save them as GIFs.

**NOTE:** Rendering is executed by Pyglet.

# MPEv2

MPEv2 is a benchmark for multi-agent reinforcement learning. It presents 6 environments: 3 are updates to the original MPE, and the others are entirely new.  

MPEv2 enables benchmarking in (i) dynamic entity compositions, (ii) intra-trajectory stochastic entity additions, and (iii) out-of-domain generalization. These changes can help bridge the gap between laboratory research and real-world deployment, which is also our motivation.  
Additionally, you can customize the configurations to specify how many and when additional entities will spawn within **a single episode**.

## Environments

MPEv2 presents 6 environments: 3 enhanced environments (*adversary, spread, tag*) and 3 novel environments (*repel, guard, hunt*).  

Below, you can see GIFs of the renderings for each environment. The blue circles represent agents, the red circles represent adversaries, the green circles represent targets, and the black circles represent landmarks. All entities are of the same size. Please refer to our paper for detailed explanation of the environments

 Adversary                      | Spread                            | Tag
--------------------------------|-----------------------------------|---------------------------------
![](images/adv_qattenood1.gif)  |  ![](images/spread_acormood1.gif) |  ![](images/tag_acormood2.gif)

 Repel                          | Guard                             | Hunt
--------------------------------|-----------------------------------|---------------------------
![](images/repel_camaood1.gif)  |  ![](images/guard_odisood1.gif)   | ![](images/hunt_acormood1.gif)

## Customize config

You can adjust the hyperparameters for generating content in MPEv2 through the config files located in `{src directory}/config/envs`. If you are using the *ACORM* method, the config files are found in `ACORM/ACORM_qmix/config/envs`.  

As an example, consider the below config:
```yaml
env: mpe_fluid

render_args:
  srk: [1,2,4]
  sr_color: ["#2CAFAC","#FB5607", "#1982C4"]

twoagent: False
agentnames:
  - agent
n_agents: 8
mac: "basic_mac" # "entity_mac" for REFIL/CAMA

env_args:
  scenario_id: "simple_spread.py"
  scenario_config:
    episode_limit: 100
    max_n_agents : 8
    max_n_landmarks: 8
    
    # train(in-domain)
    num_agents : [1, 3]
    num_landmarks: [1, 3]
    intra_trajectory: [1, 1]

    OOD:      
      # OOD 1
      - num_agents : [5]
        num_landmarks: [5]
        intra_trajectory: [2, 0]
      # OOD 2
      - num_agents : [5]
        num_landmarks: [5]
        intra_trajectory: [0, 2]

    delta_entities: # only keys got meaning
      add_agent: [50, 70] 
      add_landmark: [70]
    
    dropout:
      train: False
      inference: True
      agent: # [MOED at initialization, MOED at intratrajectory]
        1: [0, 0]
        2: [0, 1]
        3: [1, 1]
      target:
        1: [0, 0]
        2: [1, 1]
        3: [2, 1]
```
This example config is the default config for the *Spread* scenario. The `episode_limit` key defines the number of timesteps executed in 1 episode. You can adjust the number of initialized entities for each domain (*train (in-domain)*, *OOD (out-of-domian)1 and 2*) using the `num_agents` and `num_landmarks` keys. In the in-domain case, the number of entities is randomly sampled based on the range specified in the `num_{entity type}` key, such as [1, 3]. The `intra_trajectory` key allows for the addition of entities **during a single episode**. This key takes a list of length *n* (*n* being the number of entity types in the scenario). Each value in the list corresponds to the number of additions for each entity type, in the order defined by the config. In this example, in OOD 1, `intra_trajectory: [2, 0]` means 2 agents and 0 landmarks are added during the episode.

## System specifications
Experiments with this repo has been tested on the following specifications:

- **GPU**
    - NVIDIA RTX TITAN (24GB)
    - NVIDIA RTX 3090 (24GB)
    - NVIDIA RTX A5000 (24GB)
    - NVIDIA RTX A6000 (48GB)

- **CPU**
    - AMD EPYC 7452 with 60GB memory

- **OS**
    - Ubuntu 20.04 LTS

We recommend at least 24GB VRAM and 16 GB of RAM.



## Reference
Please consider citing the original MPE paper and the new MPEv2 paper. MPEv2 is under review and not yet available here to preserve anonymity.
