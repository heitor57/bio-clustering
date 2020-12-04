import sys
import random
import numpy as np

class SelectionPolicy:
    def __init__(self,population):
        self.population = population

class Tournament(SelectionPolicy):
    def select(self):
        population=self.population
        num_pop = len(self.population)
        fathers = []
        for i in range(2):
            inds = []
            for j in range(2):
                inds.append(random.randint(0,num_pop-1))
                ind = None
                get_winner = np.random.rand() <= winner_prob
            if population[inds[0]].objective_value < population[inds[1]].objective_value:
                ind = population[inds[0]]
                if not get_winner:
                    ind = population[inds[1]]
            else:
                ind = population[inds[1]]
                if not get_winner:
                    ind = population[inds[0]]

            fathers.append(ind)
        return fathers

class Roulette(SelectionPolicy):
    def __init__(self,population):
        super().__init__(population)
        objective_values = np.array([ind.objective_value for ind in self.population])
        # objective_values = 1/(objective_values+np.finfo(objective_values.dtype).min)
        objective_values = 1/objective_values
        self.probabilities = objective_values/np.sum(objective_values)
        # print(self.probabilities)
    def select(self):
        r = np.random.random()
        cumulated = 0
        chosen_ind = None
        for p, ind in zip(self.probabilities,self.population):
            cumulated += p
            if cumulated >= r:
                chosen_ind = ind
                break
        return chosen_ind


class RankRoulette(SelectionPolicy):
    def __init__(self,population):
        super().__init__(population)
        indexes = np.argsort([ind.objective_value for ind in self.population])
        i=1
        ranks = np.zeros(len(self.population))
        for index in indexes:
            ranks[index] = i
        self.probabilities = ranks/np.sum(ranks)
    def select(self):
        r = np.random.random()
        cumulated = 0
        chosen_ind = None
        for p, ind in zip(self.probabilities,self.population):
            cumulated += p
            if cumulated >= r:
                chosen_ind = ind
                break
        return chosen_ind
