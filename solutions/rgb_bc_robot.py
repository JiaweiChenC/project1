from base import RobotPolicy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np


class RGBBCRobot(RobotPolicy):
    """ Implement solution for Part3 below """
    def train(self, data):
        # for key, val in data.items():
            # print(key, val.shape)
        obs = data.get('obs')
        # print(obs.shape)
        r, g, b = obs[:, :, :, 0], obs[:, :, :, 1], obs[:, :, :, 2]
        # for row in r[0]:
            # print(''.join(["*" if x > 0 else " " for x in row]))
        # for row in g[0]:
            # print(''.join(["*" if x > 0 else " " for x in row]))
        # for row in b[0]:
            # print(''.join(["*" if x > 0 else " " for x in row]))
        obs = (r + b + g)/3
        # print(obs.shape)
        action = data.get('actions')
        obs = obs.flatten().reshape(400, 4096)
        actions = action.flatten().reshape(400,)
        # standardlize the data
        global scaler
        scaler = StandardScaler()
        scaler.fit(obs)
        obs = scaler.transform(obs)

        global pca
        pca = KernelPCA(n_components=35, kernel='linear')
        pca.fit(obs)
        res = pca.transform(obs)

        # new_matrix = []
        # for coor in res:
            # gram = coor * coor.reshape(-1, 1)
            # new_matrix.append(gram)
        # new_matrix = np.array(new_matrix)
        # res = new_matrix.reshape(400, 2500) 
        global clf
        clf = RandomForestClassifier(max_depth=10, random_state=0).fit(res, actions)
        # clf = LogisticRegression(random_state=0, max_iter=10000).fit(res, actions)
        # clf = LogisticRegression(max_iter=10000).fit(res, actions)
        # clf = MLPClassifier(max_iter=2000).fit(res, actions)
        # score = clf.score(res, actions)
        # print(res.shape)
        # print("score: ", score)
        # print("real one ", actions)
        # print("Using dummy solution for RGBBCRobot")

    def get_actions(self, observations):
        observations = observations.reshape(64, 64, 3)
        r, g, b = observations[:, :, 0], observations[:, :, 1], observations[:, :, 2]
        # for row in r:
            # print(''.join(["*" if x > 0 else " " for x in row]))
        # for row in g:
            # print(''.join(["*" if x > 0 else " " for x in row]))
        # for row in b:
            # print(''.join(["*" if x > 0 else " " for x in row]))
        observations = (r + g + b) / 3
        observations = observations.flatten().reshape(1, -1)
        observations = scaler.transform(observations)
        observations = pca.transform(observations)
        # # gram = observations * observations.reshape(-1, 1)
        # # observations = np.array(gram.flatten())
        ret = clf.predict(observations)
        return ret

