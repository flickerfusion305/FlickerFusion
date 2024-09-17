REGISTRY = {}

from .basic_controller import BasicMAC
from .twoagent_controller import TwoAgentMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["two_mac"] = TwoAgentMAC