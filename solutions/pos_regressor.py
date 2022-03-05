from base import Regressor
from sklearn.linear_model import LinearRegression
import numpy as np

class PositionRegressor(Regressor):

    """ Implement solution for Part 1 below  """
    def train(self, data):
        # for key, val in data.items():
            # print(key, val)
        image = data.get('obs')
        image = image.flatten().reshape(500, 12288)
        # normalize the matrix
        image = image.astype('float32')/255 
        # print("image shape: ", image.shape)
        inf = data.get('info')
        real_pos =  []
        for dic in inf:
            real_pos.append(dic.get('agent_pos'))
        global reg
        reg = LinearRegression().fit(image, real_pos)
        # print("pos shape: ", np.array(real_pos).shape)
        pass

    def predict(self, Xs):
        batch_size = Xs.shape[0]
        res = np.zeros((batch_size, 2))
        for i in range(batch_size):
            pre = Xs[i].flatten().reshape(1, 12288)
            pre = pre.astype('float32')/255
            prediction = reg.predict(pre)
            prediction = np.concatenate(prediction)
            # print(prediction)
            res[i] = prediction
            # ret.append(prediction)
        return res
