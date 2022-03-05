from base import RobotPolicy
import numpy as np
from sklearn.linear_model import RidgeClassifier

class POSBCRobot(RobotPolicy):
    
    """ Implement solution for Part 2 below """

    def train(self, data):
        observation = data.get('obs')
        action = data.get('actions')
        new_matrix = []
        for coor in observation:
            gram = coor * coor.reshape(-1, 1)
            new_matrix.append(gram)
        new_matrix = np.array(new_matrix)
        observation = new_matrix.flatten().reshape(500, 4)
        action = action.flatten().reshape(500,)
        global clf
        clf = RidgeClassifier()
        clf = clf.fit(observation, action)

    def get_actions(self, observations):
        
        new_matrix = []
        for coor in observations:
            gram = coor * coor.reshape(-1, 1)
            new_matrix.append(gram)
        new_matrix = np.array(new_matrix)
        observations = new_matrix.flatten().reshape(observations.shape[0], 4)
        prediction = clf.predict(observations)
        return prediction
