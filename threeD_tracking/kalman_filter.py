import numpy as np


class Kalman:

    def __init__(self, id, initial_position, width, height, dt):
        self.id = id
        self.delta_time = dt
        self.age = 0   # Frames this instance has been alive
        self.missed_frames = 0  # Frames since last detection of object

        # The initial state vector (x, y, z, vx, vy, vz) (6x1).
        self.x = np.array([[initial_position[0]],
                      [initial_position[1]],
                      [initial_position[2]],
                      [0],
                      [0],
                      [0]])

        # The initial uncertainty (6x6).
        self.P = np.array([[10, 0, 0, 0, 0, 0],
                      [0, 10, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 100, 0, 0],
                      [0, 0, 0, 0, 100, 0],
                      [0, 0, 0, 0, 0, 10]])

        # The external motion (6x1).
        self.u = np.array([[0],
                      [0],
                      [0],
                      [0],
                      [0],
                      [0]])

        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Added the process noise matrix, to try to improve performance of filter
        s_pos = 1.0  # Position noise (pixels)
        s_vel = 50.0  # Velocity noise (pixels/frame) - Lower this to stop overshooting!
        s_z = 0.5  # Depth noise (meters)
        s_vz = 10.0  # Depth velocity noise

        self.Q = np.diag([s_pos, s_pos, s_z, s_vel, s_vel, s_vz])

        # The observation matrix (2x6). NB: NOT LIKE THIS
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0]])

        # The measurement uncertainty.
        self.R = np.array([[150, 0, 0],
                    [0, 150, 0],
                    [0, 0, 1.0]])

        self.I = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        self.width = width
        self.height = height

    def update(self, Z, width, height):
        Z = np.array(Z).reshape(-1, 1)
        y = Z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.pinv(S))
        self.x = self.x + np.dot(K, y)
        P = np.dot(self.I - np.dot(K, self.H), self.P)
        self.width = width
        self.height = height

    def predict(self):
        self.x = np.dot(self.F, self.x) + self.u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def get_box_corners(self):
        box = []
        box.append(self.x[0] - self.width / 2)  # x1
        box.append(self.x[1] - self.height / 2) # y1
        box.append(self.x[0] + self.width / 2)  # x2
        box.append(self.x[1] + self.height / 2) # y2
        return box