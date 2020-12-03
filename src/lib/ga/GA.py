# coding: utf-8
import numpy as np
import pandas as pd
import yaml
import logging

import math
import random
import argparse
import collections
import sys
from pathlib import Path
import os
from tqdm import tqdm

from lib.utils import *
from lib.ClassificationProblem import *


class GA:
    def __init__(self, population_size, cross_rate, num_generations, elitism, mutation_rate,instance_name,mutation_policy,cross_policy,selection_policy,cross_policy_kwargs,eid):
        self.population_size= population_size
        self.cross_rate=cross_rate
        self.num_generations=num_generations
        self.cross_rate=cross_rate
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.instance_name = instance_name
        self.cross_policy = cross_policy
        self.mutation_policy = mutation_policy
        self.selection_policy = selection_policy
        self.cross_policy_kwargs = cross_policy_kwargs

    def run(self):
        num_cross = int((self.cross_rate * self.population_size)/2)
        num_no_cross = self.population_size-2*num_cross

        classification_problem = ClassificationProblem()
        classification_problem.load(DIRS['INPUTS']+self.instance_name)
        num_labels = len(set(classification_problem.labels))
        objective = InertiaObjective(classification_problem.features,num_labels)
        cross_policy = eval(self.cross_policy)(**self.cross_policy_kwargs)
        mutation_policy = eval(self.mutation_policy)(mutation_rate,0,num_labels-1)
        
        population = []
        for i in range(num_pop):
            ind = Individual()
            ind.rand_genome_int(len(classification_problem.weights))
            population.append(ind)
            objective.compute(ind)

        columns = ['#Iterations','Best global fitness','Best fitness','Mean fitness', 'Median fitness', 'Worst fitness']
        df = pd.DataFrame([],columns = columns)
        df = df.set_index(columns[0])
        objective_values = [ind.objective_value for ind in population]
        best_ind = population[np.argmin(objective_values)]
        best_objective_value = np.min(objective_values)
        df.loc[1] = [f'{np.min(objective_values):.4E}',f'{best_objective_value:.4E}',f'{np.mean(objective_values):.4E}',f'{np.median(objective_values):.4E}',f'{np.max(objective_values):.4E}']
        if logger.level <= logging.INFO:
            progress_bar = tqdm
        else:
            progress_bar = lambda x: x

        for i in progress_bar(range(2,self.num_generations+1)):
            new_population = []
            # Cross
            selection_policy=eval(self.selection_policy)(population)
            for j in range(num_cross):
                ind1=selection_policy.select()
                ind2=selection_policy.select()
                nind1, nind2 = cross_policy.cross(ind1,ind2)
                new_population.append(nind1)
                new_population.append(nind2)
                # Select the rest left of individuals if the cross rate is not 100%
            new_population.extend([population[i].__copy__()
                                   for i in random.sample(list(range(len(population))),num_no_cross)])

            # Mutate these new individuals - Mutation
            for j in range(num_pop):
                mutation_policy.mutate(new_population[j])

            # Select best individual from previous population - Elitism
            if elitism:
                new_population[random.randint(0,len(new_population)-1)]=best_ind.__copy__()
                population = new_population

            for ind in population:
                objective.compute(ind.genome)

            objective_values = [ind.objective_value for ind in population]
            best_objective_value = np.min(np.min(objective_values),best_objective_value)

            best_ind = population[np.argmin(objective_values)]
            df.loc[i] = [f'{np.min(objective_values):.4E}',f'{best_objective_value:.4E}',f'{np.mean(objective_values):.4E}',f'{np.median(objective_values):.4E}',f'{np.max(objective_values):.4E}']

        logger.info(f"\n{df}")
        self.save_results(df)