import numpy as np
from .Individual import *
import copy

class CrossPolicy:
    pass

class BLXa(CrossPolicy):
    def __init__(self,min_values,max_values,alpha):
        self.alpha=alpha
        self.min_values=min_values
        self.max_values=max_values
    def cross(self,ind1,ind2):
        nind1= Individual(genome=[0]*len(ind1.genome))
        nind2= Individual(genome=[0]*len(ind1.genome))
        i = 0
        for gene1, gene2 in zip(ind1.genome,ind2.genome):
            d= abs(gene1-gene2)
            min_values = min(gene1,gene2)
            max_values = max(gene1,gene2)
            nind1.genome[i] = min(max(random.uniform(min_values-self.alpha*d,max_values+self.alpha*d),self.min_values[i]),self.max_values[i])
            nind2.genome[i] = min(max(random.uniform(min_values-self.alpha*d,max_values+self.alpha*d),self.min_values[i]),self.max_values[i])
            i+=1

        return nind1, nind2

class BLXab(CrossPolicy):
    def __init__(self,min_values,max_values,alpha,beta):
        self.alpha=alpha
        self.beta=beta
        self.min_values=min_values
        self.max_values=max_values

    def cross(self,ind1,ind2):
        nind1= Individual(genome=[0]*len(ind1.genome))
        nind2= Individual(genome=[0]*len(ind1.genome))
        if ind1.objective_value > ind2.objective_value:
            ind1, ind2 = ind2, ind1

        i = 0
        for gene1, gene2 in zip(ind1.genome,ind2.genome):
            d= abs(gene1-gene2)
            if gene1 <= gene2:
                nind1.genome[i]=min(max(random.uniform(gene1-self.alpha*d,gene2+self.beta*d),self.min_values[i]),self.max_values[i])
                nind2.genome[i]=min(max(random.uniform(gene1-self.alpha*d,gene2+self.beta*d),self.min_values[i]),self.max_values[i])
            else:
                nind1.genome[i]=min(max(random.uniform(gene2-self.beta*d,gene1+self.alpha*d),self.min_values[i]),self.max_values[i])
                nind2.genome[i]=min(max(random.uniform(gene2-self.beta*d,gene1+self.alpha*d),self.min_values[i]),self.max_values[i])
            i+=1

        nind1.genome=np.array(nind1.genome)
        nind2.genome=np.array(nind2.genome)
        return nind1, nind2


class PointCross(CrossPolicy):
    def __init__(self):
        pass

    def cross(self,ind1,ind2):
        # nind1= Individual(genome=np.zeros(len(ind1.genome),dtype=np.bool))
        # nind2= Individual(genome=np.zeros(len(ind1.genome),dtype=np.bool))
        idx = random.randint(1,len(ind1.genome)-2)

        new_genome = copy.copy(ind1.genome)
        new_genome[:idx] = ind1.genome[:idx]
        new_genome[idx:] = ind2.genome[idx:]
        # new_genome = np.append(new_genome,ind2.genome[idx:])
        nind1 = Individual(new_genome)

        new_genome = copy.copy(ind1.genome)
        new_genome[:idx] = ind2.genome[:idx]
        new_genome[idx:] = ind1.genome[idx:]
        # new_genome = ind2.genome[:idx]
        # new_genome = np.append(new_genome,ind1.genome[idx:])
        nind2 = Individual(new_genome)
        return nind1, nind2


class TwoPointCross(CrossPolicy):
    def __init__(self):
        pass

    def cross(self,ind1,ind2):
        idx_1 = random.randint(1,len(ind1.genome)-2)
        idx_2 = random.randint(1,len(ind1.genome)-2)
        idx_1, idx_2 = min(idx_1,idx_2),max(idx_1,idx_2)
        
        new_genome = copy.copy(ind1.genome)
        new_genome[idx_1:idx_2+1] = ind2.genome[idx_1:idx_2+1]
        nind1 = Individual(new_genome)

        new_genome = copy.copy(ind2.genome)
        new_genome[idx_1:idx_2+1] = ind1.genome[idx_1:idx_2+1]
        nind2 = Individual(new_genome)
        return nind1, nind2
