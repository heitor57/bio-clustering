from .Particle import Particle
import numpy as np

class Topology:
    def __init__(self, num_particles):
        self.num_particles= num_particles

    def get_best_neighbor_particle(self,particle_index):
        return None

    def update_neighborhood_best(self,particle_index,particle):
        return None

class FullyConnectedTopology(Topology):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.best_particle = Particle(0,0,objective_value=np.inf,best_objective_value=np.inf)

    def get_best_neighbor_particle(self,particle_index):
        return self.best_particle

    def update_neighborhood_best(self,particle_index,particle):
        if particle.best_objective_value < self.best_particle.best_objective_value:
            self.best_particle = particle

class VonNeumannTopology(Topology):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.num_neighbors = 4
        self.neighborhoods = []
        for i in range(self.num_particles):
            self.neighborhoods.append([])
            for j in range(1,int(self.num_neighbors/2+1)):
                self.neighborhoods[i].append(int((i-j)%(self.num_particles)))

            for j in range(1,int(self.num_neighbors/2+1)):
                self.neighborhoods[i].append(int((i+j)%(self.num_particles)))
        self.best_particles = [Particle(0,0,objective_value=np.inf,best_objective_value=np.inf)]*self.num_particles

    def get_best_neighbor_particle(self,particle_index):
        return self.best_particles[particle_index]

    def update_neighborhood_best(self,particle_index,particle):
        for i in self.neighborhoods[particle_index]+[particle_index]:
            if particle.best_objective_value < self.best_particles[i].best_objective_value:
                self.best_particles[i] = particle

class RingTopology(Topology):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.num_neighbors = 2
        self.neighborhoods = []
        for i in range(self.num_particles):
            self.neighborhoods.append([])
            self.neighborhoods[i].append(int((i-1)%(self.num_particles)))
            self.neighborhoods[i].append(int((i+1)%(self.num_particles)))
        self.best_particles = [Particle(0,0,objective_value=np.inf,best_objective_value=np.inf)]*self.num_particles

    def get_best_neighbor_particle(self,particle_index):
        return self.best_particles[particle_index]

    def update_neighborhood_best(self,particle_index,particle):
        for i in self.neighborhoods[particle_index]+[particle_index]:
            if particle.best_objective_value < self.best_particles[i].best_objective_value:
                self.best_particles[i] = particle
