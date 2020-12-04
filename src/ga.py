import argparse
import yaml
import re
import logging

import lib.utils as utils
from lib.ga.GA import GA
from lib.constants import *


config = utils.parameters_init()
cross_policy_kwargs = config['cross_policy'][config['algorithms']['ga']['cross_policy']]
cross_policy_kwargs= cross_policy_kwargs if cross_policy_kwargs else dict()
ga = GA(**config['algorithms']['ga'],**config['parameters'],
        cross_policy_kwargs=cross_policy_kwargs)
logging.basicConfig()
logger = logging.getLogger('default')
logger.setLevel(eval(f"logging.{config['general']['logging_level']}"))
ga.run()
