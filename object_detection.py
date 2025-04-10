# object_detection.py

from ultralytics import YOLO
# import cv2

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        """
        Initialize the YOLOv8 object detector.

        Args:
            model_path (str): Path to YOLOv8 weights (e.g., 'yolov8n.pt').
            conf (float): Confidence threshold for detections.
        """
        self.model = YOLO(model_path)
        self.model.conf = conf  # Set confidence threshold

    def detect_objects(self, frames):
        """
        Run detection on a list of frames.

        Args:
            frames (List[np.ndarray]): List of images in BGR format.

        Returns:
            detections (List[List[Dict]]): List of detection results per frame.
        """
        results = self.model.predict(source=frames, imgsz=640, stream=True, verbose=False, conf=self.model.conf)
        all_detections = []

        for result in results:
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
            all_detections.append(detections)

        return all_detections

# Example usage
if __name__ == "__main__":
    from video_loader import load_and_sample_video

    video_file = "sample.mp4"
    frames = load_and_sample_video(video_file, frame_interval=15, max_frames=10)

    detector = YOLODetector()
    detections = detector.detect_objects(frames)

    for i, frame_dets in enumerate(detections):
        print(f"Frame {i+1}:")
        for det in frame_dets:
            print(f"  - {det['class_name']} ({det['confidence']:.2f}) at {det['bbox']}")
