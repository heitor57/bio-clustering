from lib.constants import *
import lib.utils as utils  
import logging

class ParametricAlgorithm:
    def __str__(self):
        string=""
        for k, v in self.__dict__.items():
            string+=f"{k} = {v}\n"
        return string

    def get_name(self):
        name = f"{DIRS['RESULTS']}"+utils.get_parameters_name(self.__dict__,num_dirs=3)+".json"
        l = name.split('/')
        for i in range(2,len(l)):
            directory = '/'.join(l[:i])
            logger = logging.getLogger('default')
            logger.debug(directory)
            Path(directory).mkdir(parents=True, exist_ok=True)
        return name

    def save_results(self, df):
        f = open(self.get_name(),'w')
        f.write(df.to_json(orient='records',lines=False))
        f.close()

    def load_results(self):
        string = self.get_name()
        print(string)
        return pd.read_json(string)
