import yaml
import argparse
import re
import itertools
import lib.utils as utils
from lib.constants import *
from collections import OrderedDict

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))
parser = argparse.ArgumentParser()
parser.add_argument('--config_file','-c',
                    default="config.yaml",
                    type=str,
                    help="Configuration file.")

args = parser.parse_args()
f = open(args.config_file)
config = OrderedDict(yaml.load(f,Loader=loader))

to_search = {
    'parameters':{"c_1": [0.5,1,1.5,2],
                  "c_2": [0.5,1,1.5,2],
                  "w": [0.4,0.6,0.8],
                  "eid": list(range(1,NUM_EXECUTIONS+1)),
                  "topology": ["FullyConnectedTopology",
                               "VonNeumannTopology",
                               "RingTopology"],
                  "objective_name": ["RosenbrockFunction",
                                     "ChungReynoldsFunction"],
                  },
}

keys_to_value, combinations=utils.get_names_combinations(config,to_search)
i = 0
for combination in combinations:
    for keys, v in zip(keys_to_value,combination):
        tmp = config
        for k in keys[:-1]:
            tmp = tmp[k]
        tmp[keys[-1]] = v
    yaml.dump(dict(config),open(f"{DIRS['CONFIGS']}{i}.yaml",'w+'),sort_keys=False)
    i+=1
