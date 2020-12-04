import argparse
from pathlib import Path
import yaml
from lib.utils import *
import pandas as pd
import matplotlib.pyplot as plt
from lib.constants import *
from collections import OrderedDict
from lib.ga.GA import *
import functools
config = utils.parameters_init()

cross_policy_kwargs = config['cross_policy'][config['algorithms']['ga']['cross_policy']]
cross_policy_kwargs= cross_policy_kwargs if cross_policy_kwargs else dict()
ga = GA(**config['algorithms']['ga'],**config['parameters'],
        cross_policy_kwargs=cross_policy_kwargs)
config = OrderedDict(config)
fig, ax = plt.subplots()
name = ga.get_name()
dfs = []
for i in range(1,NUM_EXECUTIONS+1):
    ga.eid = i
    # name=get_parameters_name({k: v['value'] for k,v in parameters.items()})
    # df = pd.read_json(DIRS['RESULTS']+name+'.json')
    df = ga.load_results()
    dfs.append(df)
df = pd.DataFrame(functools.reduce(lambda x,y: x+y,dfs))/len(dfs)
# df = ga.load_results()
ax.plot(df['Best global fitness'],label='Melhor aptidão global')
ax.plot(df['Best fitness'],label='Melhor aptidão')
ax.plot(df['Mean fitness'],label='Aptidão média')
ax.plot(df['Median fitness'],label='Aptidão mediana')
ax.plot(df['Worst fitness'],label='Pior Aptidão')
ax.set_ylabel("Aptidão")
ax.set_xlabel("Iteração")
ax.legend()
# fig.savefig(f"{DIRS['IMG']}{ga.objective_name}_{ga.c_1}_{ga.c_2}_{ga.w}_{ga.topology}_mmb.eps",bbox_inches="tight")
fig.savefig(f"{DIRS['IMG']}{ga.instance_name}_{ga.cross_rate}_{ga.mutation_rate}_{ga.selection_policy}_mmb.png",bbox_inches="tight")
# fig.savefig(f"{DIRS['IMG']}{ga.instance_name}_{ga.pheromony_kwargs['rho']}_{ga.pheromony_kwargs['Q']}_{ga.selection_policy_kwargs['beta']}_mean_and_median_and_best.png",bbox_inches="tight")

fig, ax = plt.subplots()
for i in range(1,NUM_EXECUTIONS+1):
    ga.eid = i
    # name=get_parameters_name({k: v['value'] for k,v in parameters.items()})
    # df = pd.read_json(DIRS['RESULTS']+name+'.json')
    df = ga.load_results()
    ax.plot(df['Best global fitness'],label=f'Execução {i}')

ax.set_ylabel("Aptidão")
ax.set_xlabel("Iteração")
ax.legend()

# fig.savefig(f"{DIRS['IMG']}{ga.objective_name}_{ga.c_1}_{ga.c_2}_{ga.w}_{ga.topology}_me.eps",bbox_inches="tight")
fig.savefig(f"{DIRS['IMG']}{ga.instance_name}_{ga.cross_rate}_{ga.mutation_rate}_{ga.selection_policy}_me.png",bbox_inches="tight")

# dfs.append(df)
    # Path(os.path.dirname(DIRS['DATA']+name)).mkdir(parents=True, exist_ok=True)
