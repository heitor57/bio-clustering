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
from lib.objectives import *
from .cross_policy import *
from .mutation_policy import *
from .selection_policy import *
from lib.ParametricAlgorithm import *
import sklearn.metrics
import lib.metrics as metrics


class GA(ParametricAlgorithm):
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
        self.eid = eid

    def run(self):
        num_cross = int((self.cross_rate * self.population_size)/2)
        num_no_cross = self.population_size-2*num_cross

        problem_instance = ClassificationProblem()
        problem_instance.load(DIRS['INPUTS']+self.instance_name)
        num_classes = problem_instance.num_classes
        objective = InertiaObjective(problem_instance.points,num_classes)
        
        dimensions = (problem_instance.num_classes,problem_instance.num_features)
        num_dimensions = np.prod(dimensions)
        features_min = np.tile(np.min(problem_instance.points,axis=0),problem_instance.num_classes)
        features_max = np.tile(np.max(problem_instance.points,axis=0),problem_instance.num_classes)

        mutation_policy = eval(self.mutation_policy)(self.mutation_rate,features_min,features_max)

        cross_policy = eval(self.cross_policy)(**self.cross_policy_kwargs,min_values=features_min,max_values=features_max)

        population = []
        for i in range(self.population_size):
            ind = Individual()
            ind.genome = np.random.random_sample(num_dimensions)*(features_max-features_min)+features_min
            ind.objective_value = objective.compute(centroids=ind.genome.reshape(dimensions))
            population.append(ind)

        columns = ['#Iterations','Best global fitness','Best fitness','Mean fitness', 'Median fitness', 'Worst fitness','Precision']
        df = pd.DataFrame([],columns = columns)
        df = df.set_index(columns[0])
        objective_values = [ind.objective_value for ind in population]

        best_ind = population[np.argmin(objective_values)]
        best_objective_value = np.min(objective_values)

        # tfpn=metrics.clustering_tfpn(problem_instance.classes,compute_classifications(particles[global_best_particle].best_position,problem_instance.points))
        # precision = metrics.precision(tfpn['tp'],tfpn['fp'])
        tfpn=metrics.clustering_tfpn(problem_instance.classes,compute_classifications(best_ind.genome.reshape(dimensions),problem_instance.points))
        precision = metrics.precision(tfpn['tp'],tfpn['fp'])
        df.loc[1] = [f'{best_objective_value:.4E}',f'{np.min(objective_values):.4E}',f'{np.mean(objective_values):.4E}',f'{np.median(objective_values):.4E}',f'{np.max(objective_values):.4E}',f'{precision:.4E}']

        logger = logging.getLogger('default')
        if logger.level <= logging.INFO:
            progress_bar = tqdm
        else:
            progress_bar = lambda x: x

        for i in progress_bar(range(1,self.num_generations+1)):
            if i == 1:
                continue
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
            new_population.extend([population[j].__copy__()
                                   for j in random.sample(list(range(len(population))),num_no_cross)])

            # Mutate these new individuals - Mutation
            for j in range(self.population_size):
                mutation_policy.mutate(new_population[j])

            # Select best individual from previous population - Elitism
            if self.elitism:
                new_population[random.randint(0,len(new_population)-1)]=best_ind.__copy__()
                population = new_population

            for ind in population:
                ind.objective_value = objective.compute(centroids=ind.genome.reshape(dimensions))

            objective_values = [ind.objective_value for ind in population]
            best_objective_value = min(np.min(objective_values),best_objective_value)

            best_ind = population[np.argmin(objective_values)]
            tfpn=metrics.clustering_tfpn(problem_instance.classes,compute_classifications(best_ind.genome.reshape(dimensions),problem_instance.points))
            precision = metrics.precision(tfpn['tp'],tfpn['fp'])
            df.loc[i] = [f'{best_objective_value:.4E}',f'{np.min(objective_values):.4E}',f'{np.mean(objective_values):.4E}',f'{np.median(objective_values):.4E}',f'{np.max(objective_values):.4E}',f'{precision:.4E}']

        # print(np.sum(np.where(best_ind.genome==problem_instance.labels,True,False))/len(problem_instance.labels))
        # logger.info(f"\n{np.sum(best_ind.genome==problem_instance.classes)/len(problem_instance.labels)}")
        # logger.info(f"\n{sklearn.metrics.classification_report(problem_instance.classes,best_ind.genome)}")
        logger.info(f"Adjusted Rand Index: {sklearn.metrics.adjusted_rand_score(problem_instance.classes,compute_classifications(best_ind.genome.reshape(dimensions),problem_instance.points))}")
        tfpn=metrics.clustering_tfpn(problem_instance.classes,compute_classifications(best_ind.genome.reshape(dimensions),problem_instance.points))
        logger.info(f"Precision: {metrics.precision(tfpn['tp'],tfpn['fp'])}")

        logger.info(f"\n{df}")
        self.save_results(df)
