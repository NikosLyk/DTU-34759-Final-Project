import numpy as np

class kalman:

    def __init__(self, initial_position, width, height, dt):
        self.delta_time = dt
        self.age = 0   # Frames this instance has been alive
        self.missed_frames = 0

        # The initial state vector (x, y, z, vx, vy, vz) (6x1).
        self.x = np.array([[initial_position[0]],
                      [initial_position[1]],
                      [initial_position[2]],
                      [0],
                      [0],
                      [0]])

        # The initial uncertainty (6x6).
        self.P = np.array([[1000000000000, 0, 0, 0, 0, 0],
                      [0, 1000000000000, 0, 0, 0, 0],
                      [0, 0, 1000000, 0, 0, 0],
                      [0, 0, 0, 1000000, 0, 0],
                      [0, 0, 0, 0, 1000000, 0],
                      [0, 0, 0, 0, 0, 1000000]])

        # The external motion (6x1).
        self.u = np.array([[0],
                      [0],
                      [0],
                      [0],
                      [0],
                      [0]])

        # The transition matrix (6x6). NB: NOT LIKE THIS
        self.F = np.array([[1, 0, dt, 0, 0.5 * dt ** 2, 0],
                      [0, 1, 0, dt, 0, 0.5 * dt ** 2],
                      [0, 0, 1, 0, dt, 0],
                      [0, 0, 0, 1, 0, dt],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        # The observation matrix (2x6). NB: NOT LIKE THIS
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]])

        # The measurement uncertainty.
        self.R = np.array([[1000000, 0],
                      [0, 1000000]])

        self.I = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        self.width = width
        self.height = height

    def update(self, Z, width, height):
        y = Z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.pinv(S))
        self.x = self.x + np.dot(K, y)
        P = np.dot(self.I - np.dot(K, self.H), self.P)
        self.width = width
        self.height = height

    def predict(self):
        self.x = np.dot(self.F, self.x) + self.u
        self.P = np.dot(np.dot(self.F, self.P), np.transpose(self.F))

    def get_box_corners(self):
        box = []
        box.append(self.x[0] - self.width / 2)  # x1
        box.append(self.x[1] - self.height / 2) # y1
        box.append(self.x[0] + self.width / 2)  # x2
        box.append(self.x[1] + self.height / 2) # y2
        return box