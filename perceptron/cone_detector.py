"""Stop the vehicle when a detected cone occupies enough of the image."""

from pathlib import Path
from ultralytics import YOLO
import cv2
import time


class ConeDetector:
    """Run a YOLO detector after the line follower and annotate its image."""

    def __init__(self, cfg):
        self.model = YOLO(str(cfg.CONE_DETECTOR_MODEL))
        self.stop_height = cfg.CONE_DETECTOR_STOP_HEIGHT
        self.currently_stopped = False
        self.stop_until = -1

    def run(self, steering, throttle, cam_img, cv_img):
        """Preserve steering and stop when any bounding box reaches stop_height."""
        if cam_img is None:
            return steering, throttle, cam_img

        # DonkeyCar camera images are RGB; Ultralytics expects BGR numpy images.
        start = time.perf_counter()
        results = self.model.predict(cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR), verbose=False)
        annotated_img = cv_img.copy()
        should_stop = False
        sections = cam_img.shape[1] / 5
        left_threshold = 2 * sections
        right_threshold = 3 * sections
        guide_color = (255, 255, 255)
        image_bottom = annotated_img.shape[0] - 1
        cv2.line(annotated_img, (int(left_threshold), 0),
                 (int(left_threshold), image_bottom), guide_color, 1)
        cv2.line(annotated_img, (int(right_threshold), 0),
                 (int(right_threshold), image_bottom), guide_color, 1)

        for result in results:
            if result.boxes is None:
                continue
            for x1, y1, x2, y2 in result.boxes.xyxy.cpu().numpy():
                height = y2 - y1
                overlaps_center = x1 < right_threshold and x2 > left_threshold
                should_stop |= overlaps_center and height > self.stop_height

                start = (int(x1), int(y1))
                end = (int(x2), int(y2))
                color = (255, 0, 0) if overlaps_center else (0, 255, 0)
                cv2.rectangle(annotated_img, start, end, color, 2)
                cv2.putText(
                    annotated_img,
                    f"h={height:.0f}",
                    (start[0], max(12, start[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        if self.currently_stopped:
            if time.time() > self.stop_until:
                self.currently_stopped = False
            else:
                throttle = 0
        else:
            if should_stop:
                self.currently_stopped = True
                self.stop_until = time.time() + 3
                throttle = 0

        elapsed = time.perf_counter() - start
        # print(f"Inference time: {elapsed * 1000:.2f} ms")
        return steering, throttle, annotated_img
