from functools import partial

from .multiagentenv import MultiAgentEnv
from .mpe_fluid import MultiAgentFluidEnv



# TODO: Do we need this?
def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["mpe_fluid"] = partial(env_fn, env=MultiAgentFluidEnv)

s_REGISTRY = {}
