import cv2, time
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
from torchvision.models.segmentation import fcn_resnet50
torch.set_num_threads(2)

class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)

class SmallestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3)
        )

    def forward(self, x):
        return self.model(x)

class SegmentationOverlay:

    def __init__(self, pid, cfg):
        self.pid = pid
        self.model_path = "./models/small-sidewalk-segmentation-model.pt"
        # self.model = SegmentationModel()
        self.model = SmallestModel()
        # self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        self.model.eval()

    def run(self, image):
        start = time.perf_counter()
        img = image.astype(np.float32) / 255 # (H, W, 3)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)
        with torch.inference_mode():
            # mask = torch.sigmoid(self.model(img_tensor)["out"]) > 0.5 # (1, 1, H, W)
            mask = torch.sigmoid(self.model(img_tensor)) > 0.5 # (1, 1, H, W)

        mask = mask.squeeze().numpy() # (H, W)
        overlay = image.copy()
        # overlay[mask, 1] = 255

        return 0, 0, overlay

        height, width = mask.shape
        rows = np.linspace(height // 2, height - 1, 4, dtype=int)
        valid_rows = np.flatnonzero(mask[height // 2:].any(axis=1)) + height // 2
        if not valid_rows.size:
            return overlay, 0.0, 0.0

        waypoints = []
        for y in rows:
            source_y = valid_rows[np.abs(valid_rows - y).argmin()]
            road_pixels = np.flatnonzero(mask[source_y])
            waypoints.append(((road_pixels[0] + road_pixels[-1]) / 2, y))

        waypoints = np.asarray(waypoints)
        spline = CubicSpline(waypoints[:, 1], waypoints[:, 0],
                             bc_type='natural')
        curve_y = np.linspace(rows[0], rows[-1], 100)
        curve_x = spline(curve_y)
        curve = np.column_stack((
            np.clip(curve_x, 0, width - 1), curve_y,
        )).round().astype(np.int32)
        cv2.polylines(overlay, [curve], False, (255, 0, 255), 2,
                      lineType=cv2.LINE_AA)
        for waypoint in waypoints.round().astype(np.int32):
            cv2.circle(overlay, tuple(waypoint), 4, (255, 255, 255), -1,
                       lineType=cv2.LINE_AA)

        self.pid.setpoint = width / 2
        lookahead_x = spline(rows[0])
        steering = float(np.clip(self.pid(lookahead_x), -1.0, 1.0))
        elapsed = time.perf_counter() - start
        # print(f"Inference time: {elapsed * 1000:.2f} ms")
        return steering, 0.1, overlay
