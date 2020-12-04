import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import logging

import math
import random
import argparse
import collections
import sys
from pathlib import Path
import os
import copy
import re

from lib.constants import *

import lib.utils as utils
import lib.objectives as objectives
from .topologies import *
from lib.ClassificationProblem import ClassificationProblem
from lib.ParametricAlgorithm import ParametricAlgorithm
from lib.objectives import *
import sklearn.metrics

class PSO(ParametricAlgorithm):
    def __init__(self, w, c_1, c_2,num_particles,num_iterations, instance_name, eid,topology):
        self.w = w
        self.c_1 = c_1
        self.c_2 = c_2
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.instance_name = instance_name
        self.eid = eid
        self.topology = topology

    def run(self):
        problem_instance = ClassificationProblem()
        problem_instance.load(DIRS['INPUTS']+self.instance_name)
        features_min = np.tile(np.min(problem_instance.points,axis=0),problem_instance.num_classes).reshape(problem_instance.num_classes,problem_instance.num_features)
        features_max = np.tile(np.max(problem_instance.points,axis=0),problem_instance.num_classes).reshape(problem_instance.num_classes,problem_instance.num_features)
        
        objective = InertiaObjective(problem_instance.points,problem_instance.num_classes)
        particles = []
        num_dimensions = (problem_instance.num_classes,problem_instance.num_features)
        topology = eval(self.topology)(num_particles=self.num_particles)
        for i in range(self.num_particles):
            particles.append(Particle(features_min,features_max))
            particles[-1].init_values(num_dimensions,objective)
        
        for i in range(self.num_particles):
            topology.update_neighborhood_best(i,particles[i])
            
        global_best_particle = np.argmin([p.objective_value for p in particles])
        columns = ['#Iterations','Best global fitness','Best fitness','Mean fitness', 'Median fitness', 'Worst fitness']
        df = pd.DataFrame([],columns = columns)
        df = df.set_index(columns[0])

        logger = logging.getLogger('default')
        if logger.level <= logging.INFO:
            progress_bar = tqdm
        else:
            progress_bar = lambda x: x
            
        for i in progress_bar(range(1,self.num_iterations+1)):
            for j, particle in enumerate(particles):
                r_1 = np.random.random_sample(num_dimensions)
                r_2 = np.random.random_sample(num_dimensions)
                particle.velocity=self.w*particle.velocity+\
                    self.c_1*r_1*(particle.best_position - particle.position)+\
                    self.c_2*r_2*(topology.get_best_neighbor_particle(j).best_position - particle.position)
                particle.position = particle.position + particle.velocity
                particle.objective_value = objective.compute(centroids=particle.position)

                if particle.objective_value < particle.best_objective_value:
                    particle.best_objective_value = particle.objective_value
                    particle.best_position = particle.position
                    topology.update_neighborhood_best(j,particle)
                    if particle.objective_value < particles[global_best_particle].best_objective_value:
                        global_best_particle = j
            
            objective_values = [p.objective_value for p in particles]
            df.loc[i] = [f'{particles[global_best_particle].best_objective_value:.4E}',f'{np.min(objective_values):.4E}',f'{np.mean(objective_values):.4E}',f'{np.median(objective_values):.4E}',f'{np.max(objective_values):.4E}']

        # logger.info(f"\n{np.sum(best_ind.genome==problem_instance.classes)/len(problem_instance.labels)}")
        
        
        logger.info(f"\n{sklearn.metrics.classification_report(problem_instance.classes,compute_classifications(particles[global_best_particle].best_position,problem_instance.points))}")
        logger.info(f"\n{df}")
        self.save_results(df)
