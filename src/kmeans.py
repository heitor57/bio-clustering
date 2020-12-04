import argparse
import yaml
import re
import logging

import lib.utils as utils
from lib.pso.PSO import PSO
from lib.constants import *
import sklearn.cluster
import sklearn
from lib.ClassificationProblem import *

config = utils.parameters_init()
logging.basicConfig()
logger = logging.getLogger('default')
logger.setLevel(eval(f"logging.{config['general']['logging_level']}"))
instance_name = config['parameters']['instance_name']
problem_instance = ClassificationProblem()
problem_instance.load(DIRS['INPUTS']+instance_name)

kmeans = sklearn.cluster.KMeans(n_clusters=problem_instance.num_classes)
kmeans.fit(problem_instance.points)

logger.info(f"Adjusted Rand Index: {sklearn.metrics.adjusted_rand_score(problem_instance.classes,kmeans.labels_)}")

