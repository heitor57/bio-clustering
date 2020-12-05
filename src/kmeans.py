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
import lib.metrics as metrics

config = utils.parameters_init()
logging.basicConfig()
logger = logging.getLogger('default')
logger.setLevel(eval(f"logging.{config['general']['logging_level']}"))
instance_name = config['parameters']['instance_name']
problem_instance = ClassificationProblem()
problem_instance.load(DIRS['INPUTS']+instance_name)


precisions = []
for i in range(NUM_EXECUTIONS):
    kmeans = sklearn.cluster.KMeans(n_clusters=problem_instance.num_classes,**config['algorithms']['kmeans'])

    kmeans.fit(problem_instance.points)

    tfpn=metrics.clustering_tfpn(problem_instance.classes,kmeans.labels_)
    precision = metrics.precision(tfpn['tp'],tfpn['fp'])
    precisions.append(precision)

logger.info(instance_name)
logger.info(f"Precision: {np.mean(precisions):.4f} ({np.std(precisions):.4f})")

# logger.info(f"Adjusted Rand Index: {sklearn.metrics.adjusted_rand_score(problem_instance.classes,kmeans.labels_)}")

