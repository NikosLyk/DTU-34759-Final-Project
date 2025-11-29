import numpy as np
from threeD_tracking import kalman_filter
from threeD_tracking import data_association


class Tracker3d:

    def __init__(self, delta_time, depth_threshold=1.5, max_missed_frames=10, max_pixel_dist=200):
        self.dt = delta_time
        self.active_kalmans = []
        self.max_pixel_dist = max_pixel_dist

        self.depth_threshold = depth_threshold
        self.max_missed_frames = max_missed_frames

        self.next_kalman_id = 0

        self.trajectories = {}

        self.frame_count = 0

        self.n_detections = 0

        self.current_state = []

    def update_tracker(self, new_detections):

        self.n_detections = len(new_detections)

        self.frame_count += 1

        for kalman in self.active_kalmans:
            kalman.predict()

        # Populating kalmans array after first detections
        if len(self.active_kalmans) == 0:
            for detection in new_detections:
                self._create_new_kalman(detection)
        else:
            matches, unmatched_kalman_idx, unmatched_detections_idx = data_association.assosiate_detections_to_kalmans(
                self.active_kalmans, new_detections, self.depth_threshold, self.max_pixel_dist
            )

            for k_idx, d_idx in matches:
                kalman = self.active_kalmans[k_idx]
                kalman.missed_frames = 0
                detection = new_detections[d_idx]
                Z = np.array([float(detection[0]), float(detection[1]), float(detection[2])])
                width = float(detection[3])
                height = float(detection[4])
                kalman.update(Z, width, height)

                self._save_trajectory(kalman)

            # Handle potential lost objects
            self._manage_unmatched_kalmans(unmatched_kalman_idx)   # TODO: double check this

            # Handle new detections
            for d_idx in unmatched_detections_idx:
                detection = new_detections[d_idx]
                self._create_new_kalman(detection)

            state = []
            for kalman in self.active_kalmans:
                if kalman.missed_frames <= self.max_missed_frames:
                    raw_box = kalman.get_box_corners()
                    flat_box = np.array(raw_box).flatten()
                    # Clean the box
                    box_clean = [float(x) for x in flat_box]
                    state.append({
                        'id': kalman.id,
                        'bbox': [float(x) for x in box_clean],
                        'depth': float(kalman.x.flatten()[2])
                    })

            self.current_state = state

    def _create_new_kalman(self, detection):
        """Helper function to create and register a new tracker."""
        kalman_id = self.next_kalman_id
        position = [detection[0], detection[1], detection[2]]
        new_kalman = kalman_filter.Kalman(kalman_id, position, detection[3], detection[4], self.dt)

        self.active_kalmans.append(new_kalman)
        self.next_kalman_id += 1

        # Initialize the trajectory history for this new tracker
        self.trajectories[kalman_id] = []
        self._save_trajectory(new_kalman)

    def _manage_unmatched_kalmans(self, unmatched_indices):
        """Helper function to delete old, lost kalmans."""
        # Iterate in reverse order to safely delete items from the list without messing up the indices.
        for k_idx in sorted(unmatched_indices, reverse=True):
            kalman = self.active_kalmans[k_idx]

            kalman.missed_frames += 1

            if kalman.missed_frames > self.max_missed_frames:
                # The trajectory data in self.trajectories is preserved
                self.active_kalmans.pop(k_idx)

    def _save_trajectory(self, kalman):
        """Helper function to save the current state to its trajectory."""

        # Clean the center point (convert from numpy array to list)
        flat_state = np.array(kalman.x).flatten()
        center_clean = [float(flat_state[0]), float(flat_state[1])]
        depth_clean = float(flat_state[2])

        raw_box = kalman.get_box_corners()
        flat_box = np.array(raw_box).flatten()
        # Clean the box
        box_clean = [float(x) for x in flat_box]

        self.trajectories[kalman.id].append({
            'frame': self.frame_count,
            'bbox': box_clean,
            'depth': depth_clean,
            'center': center_clean
        })
