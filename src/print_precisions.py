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
pso = PSO(**config['algorithms']['pso'],**config['parameters'])

cross_policy_kwargs = config['cross_policy'][config['algorithms']['ga']['cross_policy']]
cross_policy_kwargs= cross_policy_kwargs if cross_policy_kwargs else dict()
ga = GA(**config['algorithms']['ga'],**config['parameters'],
        cross_policy_kwargs=cross_policy_kwargs)

config = OrderedDict(config)
name = pso.get_name()
last_rows = []
for i in range(1,NUM_EXECUTIONS+1):
    pso.eid = i
    df = pso.load_results()
    last_rows.append(np.array(df.tail(1)))

print(config['parameters']['instance_name'])
print("PSO")
print(f"Precision: {np.mean(last_rows, axis=0)[-1][-1]:.4f} ({np.std(last_rows, axis=0)[-1][-1]:.4f})")


name = ga.get_name()
last_rows = []
for i in range(1,NUM_EXECUTIONS+1):
    pso.eid = i
    df = ga.load_results()
    last_rows.append(np.array(df.tail(1)))

print("GA")
print(f"Precision: {np.mean(last_rows, axis=0)[-1][-1]:.4f} ({np.std(last_rows, axis=0)[-1][-1]:.4f})")

