import os
import pickle
import numpy as np

class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        with open(os.path.join('myapp', 'model.pkl'), "rb") as f:
            loaded_model = pickle.load(f)
        self.model = loaded_model

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        x_np = np.array(x)
        # x_flatten = [i.flatten() for i in x]
        x_flatten = x_np.flatten().reshape(1, -1)
        # print('x_flatten', x_flatten)
        pred = self.model.predict(x_flatten)
        return pred
