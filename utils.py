# utils.py

import cv2

def draw_detections(frame, detections, tracks, anomalies_text=None):
    """
    Draws bounding boxes, class names, and track IDs on the frame.

    Args:
        frame (np.ndarray): The image frame.
        detections (List[Dict]): Detection results for the frame.
        tracks (Dict[int, Tuple[int, int]]): ID → (center_x, center_y).
        anomalies_text (str or None): Optional text to overlay as alert.
    """
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        color = (0, 255, 0) if class_name in {"car", "bus", "truck"} else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for obj_id, center in tracks.items():
        cx, cy = center
        cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
        cv2.putText(frame, f"ID {obj_id}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    if anomalies_text:
        y0 = 30
        for i, line in enumerate(anomalies_text.split("; ")):
            y = y0 + i * 25
            cv2.putText(frame, f"⚠️ {line}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame
