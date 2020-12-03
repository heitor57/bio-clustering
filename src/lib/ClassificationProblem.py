import numpy as np
class ClassificationProblem:
    def __init__(self):
        pass

    def load(self,instance_name):
        self.points = []
        self.labels = []
        for line in open(instance_name):
            if line!='\n':
                l = line.split(',')
                self.points.append(list(map(float,l[:-1])))
                self.labels.append(l[-1])
        self.points = np.array(self.points)

        self.label_to_class = {label: i for i, label in enumerate(set(self.labels))}
        self.classes = np.array([self.label_to_class[l] for l in self.labels])

        self.num_classes = len(set(self.classes))
        
    def __str__(self):
        return f"""{self.points}
{self.labels}
"""
