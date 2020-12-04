import argparse
import yaml
import re
import logging

import lib.utils as utils
from lib.pso.PSO import PSO
from lib.constants import *


config = utils.parameters_init()
pso = PSO(**{**config['algorithms']['pso'],**config['parameters']})
logging.basicConfig()
logger = logging.getLogger('default')
logger.setLevel(eval(f"logging.{config['general']['logging_level']}"))
pso.run()
