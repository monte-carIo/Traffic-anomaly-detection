# tracker.py

import math
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_distance=300, max_lost=5):
        """
        Args:
            max_distance (int): Max distance to associate objects across frames.
            max_lost (int): How many frames an object can be missing before removal.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()      # object_id -> centroid
        self.lost = OrderedDict()         # object_id -> lost counter
        self.max_distance = max_distance
        self.max_lost = max_lost

    def _euclidean(self, pt1, pt2):
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def update(self, detections):
        """
        Updates object tracking with current frame detections.

        Args:
            detections (List[Dict]): Detected objects with 'bbox' key.

        Returns:
            Dict[int, Tuple]: Mapping from object_id to centroid.
        """
        current_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_centroids.append(center)

        updated_ids = {}

        if not self.objects:
            # Initialize new objects
            for centroid in current_centroids:
                self.objects[self.next_object_id] = centroid
                self.lost[self.next_object_id] = 0
                updated_ids[self.next_object_id] = centroid
                self.next_object_id += 1
        else:
            # Match existing objects to new centroids
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            matched = set()
            for centroid in current_centroids:
                distances = [self._euclidean(centroid, c) for c in obj_centroids]
                min_dist = min(distances)
                min_idx = distances.index(min_dist)

                if min_dist < self.max_distance:
                    object_id = obj_ids[min_idx]
                    self.objects[object_id] = centroid
                    self.lost[object_id] = 0
                    updated_ids[object_id] = centroid
                    matched.add(object_id)
                else:
                    # New object
                    self.objects[self.next_object_id] = centroid
                    self.lost[self.next_object_id] = 0
                    updated_ids[self.next_object_id] = centroid
                    self.next_object_id += 1

            # Update lost counters
            for object_id in list(self.objects.keys()):
                if object_id not in updated_ids:
                    self.lost[object_id] += 1
                    if self.lost[object_id] > self.max_lost:
                        del self.objects[object_id]
                        del self.lost[object_id]

        return dict(self.objects)

# Example usage
if __name__ == "__main__":
    tracker = CentroidTracker()

    # Simulated frames of detections
    frame_1 = [{"bbox": [100, 120, 200, 180]}]
    frame_2 = [{"bbox": [105, 122, 205, 182]}]
    frame_3 = [{"bbox": [300, 400, 360, 460]}]  # New object

    for i, detections in enumerate([frame_1, frame_2, frame_3]):
        tracks = tracker.update(detections)
        print(f"Frame {i+1} tracks: {tracks}")
