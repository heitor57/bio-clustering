import numpy as np
class ClassificationProblem:
    def __init__(self):
        pass

    def load(self,instance_name):
        self.features = []
        self.labels = []
        for line in open(instance_name):
            l = line.split(',')
            self.features.append(l[:-1])
            self.labels.append(l[-1])
        self.features = np.array(self.features)
        
    def __str__(self):
        return f"""{self.features}
{self.labels}
"""
