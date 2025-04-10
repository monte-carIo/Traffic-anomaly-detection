# anomaly_detection.py

import math
from collections import defaultdict

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "van"}

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def detect_anomalies_with_tracking(frames_detections, frames_tracks, movement_thresh=5, max_stationary_frames=3, min_traffic_threshold=1):
    """
    Detect traffic anomalies using tracked object IDs.

    Args:
        frames_detections (List[List[Dict]]): Detections per frame (with bboxes).
        frames_tracks (List[Dict[int, Tuple]]): Track centroids per frame (object_id -> center).
        movement_thresh (int): Distance to count as stationary.
        max_stationary_frames (int): Frames needed to flag as stopped.
        min_traffic_threshold (int): Expected minimum number of vehicles.

    Returns:
        List[str]: Anomalies per frame.
    """
    anomalies = []
    position_history = defaultdict(list)  # object_id -> [center1, center2, ...]

    for frame_idx, (detections, track_dict) in enumerate(zip(frames_detections, frames_tracks)):
        frame_anomalies = []
        vehicle_count = 0
        non_vehicle_objects = []

        # Build lookup for ID-to-class
        id_to_class = {}

        for det in detections:
            class_name = det["class_name"]
            if class_name in VEHICLE_CLASSES:
                vehicle_count += 1
            else:
                non_vehicle_objects.append(class_name)

        for obj_id, center in track_dict.items():
            # Update position history
            position_history[obj_id].append(center)
            if len(position_history[obj_id]) > max_stationary_frames:
                position_history[obj_id] = position_history[obj_id][-max_stationary_frames:]

            # Check movement over time
            if len(position_history[obj_id]) == max_stationary_frames:
                dist = euclidean_dist(position_history[obj_id][0], position_history[obj_id][-1])
                if dist < movement_thresh:
                    frame_anomalies.append(f"Object {obj_id} appears stopped at {center}")

        if non_vehicle_objects:
            frame_anomalies.append(f"Unexpected object(s): {', '.join(set(non_vehicle_objects))}")
        if vehicle_count < min_traffic_threshold:
            frame_anomalies.append("Low traffic volume")

        anomalies.append("; ".join(frame_anomalies) if frame_anomalies else "")

    return anomalies

# Example usage (using outputs from your detection and tracking pipeline)
if __name__ == "__main__":
    from tracker import CentroidTracker

    detections_per_frame = [
        [{"class_name": "car", "bbox": [100, 120, 200, 180]}],
        [{"class_name": "car", "bbox": [105, 122, 205, 182]}],
        [{"class_name": "car", "bbox": [106, 121, 206, 181]}],
        [{"class_name": "dog", "bbox": [220, 150, 260, 200]}]
    ]

    tracker = CentroidTracker()
    tracks_per_frame = []

    for dets in detections_per_frame:
        tracks = tracker.update(dets)
        tracks_per_frame.append(tracks)

    anomalies = detect_anomalies_with_tracking(detections_per_frame, tracks_per_frame)
    for i, a in enumerate(anomalies):
        print(f"Frame {i+1}: {a}")
