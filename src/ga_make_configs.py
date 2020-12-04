import yaml
import argparse
import re
import itertools
import lib.utils as utils
from lib.constants import *
from collections import OrderedDict
import lib.utils as utils

config = utils.parameters_init()

to_search = {
    'algorithms':{'ga':
                  {
                  "cross_rate": [0.6,0.8,1.0],
                  "mutation_rate": [0.2,0.1,0.05,0.01,0.005,0.001],
                  "selection_policy": ['RankRoulette','Roulette'],
                  }},
    'parameters':{
        "eid": list(range(1,NUM_EXECUTIONS+1)),
    }
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
