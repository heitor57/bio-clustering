import random
import numpy as np
class MutationPolicy:
    pass

class UniformRandomDiscreteSolutionSpace(MutationPolicy):
    def __init__(self,probability,min_value,max_value):
        self.probability = probability
        self.min_value = min_value
        self.max_value = max_value
    
    def mutate(self,ind):
        new_genome = np.array([random.randint(self.min_value,self.max_value)
                      if self.probability > random.random()
                      else
                      i
                      for i in ind.genome])
        ind.genome = new_genome
