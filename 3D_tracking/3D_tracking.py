import cv2
import numpy as np
import kalman_filter
import data_association


class Tracker3d:

    def __init__(self, delta_time, iou_threshold=0.3, depth_threshold=1.5, max_missed_frames=10):
        self.dt = delta_time
        self.active_kalmans = []

        self.iou_threshold = iou_threshold
        self.depth_threshold = depth_threshold
        self.max_missed_frames = max_missed_frames

        self.next_kalman_id = 0

        self.trajectories = {}

        self.frame_count = 0

        self.current_state = []

    def update_tracker(self, new_detections):

        self.frame_count += 1

        for kalman in self.active_kalmans:
            kalman.predict()

        # Populating kalmans array after first detections
        if len(self.active_kalmans) == 0:
            for detection in new_detections:
                self._create_new_kalman(detection)
        else:
            matches, umatched_kalman_idx, unmatched_detections_idx = data_association.assosiate_detections_to_kalmans(
                self.active_kalmans, new_detections, self.iou_threshold, self.depth_threshold
            )

            for k_idx, d_idx in matches:
                kalman = self.active_kalmans[k_idx]
                detection = new_detections[d_idx]
                kalman.update(detection)

                self._save_trajectory(kalman)

            # Handle potential lost objects
            self._manage_unmatched_kalmans(unmatched_detections_idx)

            # Handle new detections
            for d_idx in unmatched_detections_idx:
                detection = new_detections[d_idx]
                self._create_new_kalman(detection)

            state = []
            for kalman in self.active_kalmans:
                if kalman.missed_frames <= self.max_missed_frames:
                    state.append({
                        'id': kalman.id,
                        'bbox': kalman.get_box_corners(),
                        'depth': kalman.x[2]
                    })

            self.current_state = state

    def _create_new_kalman(self, detection):
        """Helper function to create and register a new tracker."""
        kalman_id = self.next_kalman_id
        position = [detection['pos_x'], detection['pos_y'], detection['depth_m']]
        new_kalman = kalman_filter.Kalman(kalman_id, position, detection['width'], detection['height'], self.dt)

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

            if kalman.missed_frames > self.max_missed_frames:
                # The trajectory data in self.trajectories is preserved
                self.active_kalmans.pop(k_idx)

    def _save_trajectory(self, kalman):
        """Helper function to save the current state to its trajectory."""
        self.trajectories[kalman.id].append({
            'frame': self.frame_count,
            'bbox': kalman.get_box_corners(),
            'depth': kalman.x[2],
            'center': kalman.x[:2]
        })
