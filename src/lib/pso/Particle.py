import numpy as np
import copy

class Particle:
    def __init__(self, x_min, x_max, position=None, velocity=None, objective_value=None,best_objective_value=None):
        self.x_min = x_min
        self.x_max = x_max
        self.position= position
        self.velocity= velocity
        self.objective_value= objective_value
        self.best_objective_value= best_objective_value

    # @property
    # def velocity(self):
    #     return self._velocity
    # @velocity.setter
    # def velocity(self,value):
    #     self._velocity = value
    #     if isinstance(value,np.ndarray):
    #         self._velocity = np.maximum(self._velocity,np.abs(self.x_min)*0.1)
    #         self._velocity = np.minimum(self._velocity,self.x_max*0.1)
    
    def init_values(self,num_dimensions,objective):
        self.position = np.random.rand(num_dimensions)*(self.x_max-self.x_min)+self.x_min
        self.velocity = np.zeros(num_dimensions)
        self.objective_value=objective.compute(self.position)
        self.best_position = copy.copy(self.position)
        self.best_objective_value = self.objective_value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self,value):
        self._position = value
        if isinstance(value,np.ndarray):
            self._position = np.maximum(self._position,self.x_min)
            self._position = np.minimum(self._position,self.x_max)
        
        
