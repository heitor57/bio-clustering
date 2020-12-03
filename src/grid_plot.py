import os
from concurrent.futures import ProcessPoolExecutor
import itertools
import yaml
import sys
import copy

import numpy as np
import pandas as pd

from lib.constants import *
from lib.utils import *
TOP_N = 15
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
config = yaml.load(f,Loader=loader)

to_search = {
    'parameters':{"c_1": [0.5,1,1.5,2],
                  "c_2": [0.5,1,1.5,2],
                  "w": [0.4,0.6,0.8],
                  "eid": list(range(1,NUM_EXECUTIONS+1)),
                  "topology": ["FullyConnectedTopology",
                               "VonNeumannTopology",
                               "RingTopology"],
                  "objective_name":[
                  "RosenbrockFunction",
                  # "ChungReynoldsFunction",
                  ],
                  },
}


keys_to_value, combinations=utils.get_names_combinations(config,to_search)
result_df = pd.DataFrame(columns=
                         [keys[-1] for keys in keys_to_value])

parameters_names = [i[-1] for i in keys_to_value]
i = 0
for combination in combinations:
    for keys, v in zip(keys_to_value,combination):
        tmp = config
        for k in keys[:-1]:
            tmp = tmp[k]
        tmp[keys[-1]] = v

        result_df.loc[i,keys[-1]] = v
        
    pso = PSO(**config['parameters'])
    df = pso.load_results()
    result_df.loc[i,parameters_names] = combination
    result_df.loc[i,'Best global fitness'] = df.iloc[-1]['Best global fitness']
    result_df.loc[i,'Best fitness'] = df.iloc[-1]['Best fitness']
    result_df.loc[i,'Mean fitness'] = df.iloc[-1]['Mean fitness']
    result_df.loc[i,'Median fitness'] = df.iloc[-1]['Median fitness']
    result_df.loc[i,'Worst fitness'] = df.iloc[-1]['Worst fitness']
    i += 1

result_df['eid']=pd.to_numeric(result_df['eid'])
# print('Top best fitness')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(result_df)
    pd.set_option('display.expand_frame_repr', False)
    tmp = copy.copy(parameters_names)
    tmp.remove('eid')
    a=result_df.groupby(list(set(result_df.columns)-{'Best global fitness','Best fitness','Mean fitness','Median fitness','Worst fitness', 'eid'})).\
        agg({i: ['mean','std'] for i in {'Best global fitness', 'Best fitness','Mean fitness','Median fitness','Worst fitness', 'eid'}}).\
        sort_values(by=[('Best global fitness','mean')],ascending=True).reset_index()[tmp+['Best global fitness','Best fitness','Mean fitness','Median fitness','Worst fitness',]].head(TOP_N)
    open(f"../doc/{config['parameters']['objective_name']}_output.tex",'w').write(a.to_latex())


# print('Top mean fitness')
# print(result_df.groupby(list(set(result_df.columns)-{'Best fitness','Mean fitness', 'eid'})).\
    #       agg({i: ['mean','median','std'] for i in {'Best fitness','Mean fitness', 'eid'}}).\
    #       sort_values(by=[('Mean fitness','mean')],ascending=True).reset_index()[list(set(to_update.keys())-{'eid'})+['Best fitness','Mean fitness']].head(TOP_N))
