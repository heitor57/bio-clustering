import random
import numpy as np
class Individual:
    def __init__(self,genome=[]):
        self.genome = genome
        self.objective_value = None

    def rand_genome_int(self,num_ints,num_genes):
        self.genome = np.random.choice(a=np.arange(num_ints), size=(num_genes,))
        return self.genome

    def __copy__(self):
        new_ind = Individual()
        new_ind.genome = self.genome.copy()
        new_ind.objective_value = self.objective_value
        return new_ind

    def __str__(self):
        return 'genome = '+','.join(map(lambda x: f'{x}',self.genome)) + f', fo = {self.objective_value}, {len(self.genome)} genes'
