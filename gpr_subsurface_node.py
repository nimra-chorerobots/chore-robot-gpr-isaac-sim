#!/usr/bin/env python3
"""
gpr_subsurface_node.py
=======================
Front_GPR_Module subsurface scan node.

Isaac Sim has no built-in GPR sensor (NVIDIA's docs describe Isaac Sim sensors as
extensible/customizable, not GPR as a stock option) -- this follows that extensible-sensor
pattern: gpr_camera (a real downward render pipeline, see gpr_camera_graph_01 in the .usda)
drives a simulated ROS2 topic with plausible subsurface-scan values, the same trick as
faking a sensor with a camera plugin in Gazebo.

Publishes:
    /gpr/subsurface_scan   (std_msgs/String, JSON)
        { soil_density, subsurface_obstacle, depth_estimate, confidence }
    /gpr/markers           (visualization_msgs/MarkerArray)
        one marker at Front_GPR_Module's live position, colored:
            green  = normal ground
            yellow = uncertain soil reading
            red    = buried object / hard surface detected

Subscribes:
    /gpr/depth                  (sensor_msgs/Image, 32FC1)  -- from gpr_camera
    /gpr/semantic_segmentation  (sensor_msgs/Image)          -- from gpr_camera
    /odom                       (nav_msgs/Odometry)          -- robot pose, for marker placement

Deploy alongside your other perception nodes in chore_stable:
    docker cp gpr_subsurface_node.py chore_stable:/perception/
    docker exec -it chore_stable bash
    python3 /perception/gpr_subsurface_node.py

NOTE ON MESSAGE TYPE: /gpr/subsurface_scan is published as JSON on std_msgs/String so
it deploys with zero package-build overhead -- consistent with how /gpr/detections was
done earlier. If you want a typed message for a more "real" ROS2 interface later
(float32 soil_density, bool subsurface_obstacle, float32 depth_estimate, float32
confidence), that's a small custom .msg + colcon build away -- worth doing after the
demo, not before it.
"""

import json
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge

# ---------------------------------------------------------------------------
MANIFEST_PATH = os.environ.get("GPR_MANIFEST_PATH", "/perception/gpr_ground_truth.json")

# Semantic label -> manifest class (matches the surface_marker labels in the .usda)
SEMANTIC_LABEL_TO_MANIFEST_CLASS = {
    "gpr_pipe": "pipe",
    "gpr_rock": "rock",
    "gpr_root": "root",
    "gpr_cable": "cable",
}

MIN_PIXELS_FOR_DETECTION = 15
BASELINE_SOIL_DENSITY = 1.60      # g/cm^3, typical loam baseline -- tune to your soil type
SOIL_DENSITY_NOISE_STD = 0.03
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.4   # below this with no clear hit -> yellow
TICK_HZ = 5.0


class GPRSubsurfaceNode(Node):
    def __init__(self):
        super().__init__("gpr_subsurface_node")

        self.bridge = CvBridge()
        self.manifest_by_class = self._load_manifest(MANIFEST_PATH)
        self.get_logger().info(f"GPR manifest loaded: {list(self.manifest_by_class.keys())}")

        self.latest_depth = None
        self.latest_segmentation = None
        self.robot_xy = np.array([0.0, 0.0])
        self.robot_z = 0.0
        self.have_pose = False

        self.depth_sub = self.create_subscription(Image, "/gpr/depth", self._depth_cb, 5)
        self.seg_sub = self.create_subscription(Image, "/gpr/semantic_segmentation", self._seg_cb, 5)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        self.scan_pub = self.create_publisher(String, "/gpr/subsurface_scan", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/gpr/markers", 10)

        self.timer = self.create_timer(1.0 / TICK_HZ, self._tick)

        self.get_logger().info("Front_GPR_Module up -- publishing /gpr/subsurface_scan + /gpr/markers")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_manifest(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return {obj["class"]: obj for obj in data["objects"]}
        except Exception as e:  # noqa: BLE001
            print(f"[gpr_subsurface_node] WARNING: could not load manifest at {path}: {e}")
            return {}

    def _odom_cb(self, msg: Odometry):
        self.robot_xy = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.robot_z = msg.pose.pose.position.z
        self.have_pose = True

    def _depth_cb(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _seg_cb(self, msg: Image):
        try:
            self.latest_segmentation = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Could not decode segmentation image: {e}")

    # ------------------------------------------------------------------ #
    def _best_detection(self):
        """Return the strongest current buried-object detection, or None."""
        if self.latest_segmentation is None or self.latest_depth is None:
            return None

        seg = self.latest_segmentation
        best = None
        best_pixels = 0

        # NOTE: swap in the real id<->label table your ROS2CameraHelper segmentation
        # config assigns at runtime -- this "class_{id}" placeholder mirrors the
        # earlier gpr_camera_node.py and needs the same one-time confirmation.
        for class_id in np.unique(seg):
            if class_id == 0:
                continue
            mask = seg == class_id
            pixel_count = int(mask.sum())
            if pixel_count < MIN_PIXELS_FOR_DETECTION or pixel_count <= best_pixels:
                continue

            label = f"class_{int(class_id)}"
            manifest_class = SEMANTIC_LABEL_TO_MANIFEST_CLASS.get(label)
            if manifest_class is None:
                continue
            obj = self.manifest_by_class.get(manifest_class)
            if obj is None:
                continue

            best_pixels = pixel_count
            best = (obj, pixel_count)

        return best

    def _compute_scan(self):
        """Build the {soil_density, subsurface_obstacle, depth_estimate, confidence} reading."""
        detection = self._best_detection()

        # baseline "normal ground" reading with sensor noise
        soil_density = BASELINE_SOIL_DENSITY + np.random.normal(0.0, SOIL_DENSITY_NOISE_STD)
        subsurface_obstacle = False
        depth_estimate = None
        confidence = max(0.0, min(1.0, np.random.normal(0.75, 0.08)))  # nominal confidence, normal ground

        if detection is not None:
            obj, pixel_count = detection
            subsurface_obstacle = True
            depth_estimate = obj["depth_m"]
            confidence = min(1.0, 0.5 + pixel_count / 400.0)
            # denser/higher-permittivity materials read as a density anomaly relative to baseline
            er = obj.get("relative_permittivity", 6.0)
            soil_density = BASELINE_SOIL_DENSITY + (er - 6.0) * 0.05 + np.random.normal(0.0, SOIL_DENSITY_NOISE_STD)

        return {
            "soil_density": round(float(soil_density), 3),
            "subsurface_obstacle": bool(subsurface_obstacle),
            "depth_estimate": round(float(depth_estimate), 3) if depth_estimate is not None else None,
            "confidence": round(float(confidence), 3),
        }

    def _publish_scan(self, scan, stamp):
        payload = {"stamp_sec": stamp.sec + stamp.nanosec * 1e-9, **scan}
        msg = String()
        msg.data = json.dumps(payload)
        self.scan_pub.publish(msg)

    def _marker_color(self, scan):
        """green = normal ground, yellow = uncertain soil, red = buried object / hard surface."""
        if scan["subsurface_obstacle"]:
            return (0.85, 0.15, 0.15, 0.9)  # red
        if scan["confidence"] < UNCERTAIN_CONFIDENCE_THRESHOLD:
            return (0.95, 0.75, 0.1, 0.9)   # yellow
        return (0.2, 0.75, 0.25, 0.9)        # green

    def _publish_marker(self, scan, stamp):
        if not self.have_pose:
            return

        r, g, b, a = self._marker_color(scan)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = "gpr_subsurface"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.robot_xy[0]) + 0.35  # Front_GPR_Module's forward offset
        marker.pose.position.y = float(self.robot_xy[1])
        marker.pose.position.z = self.robot_z - 0.15
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        marker.lifetime = Duration(seconds=0.5).to_msg()

        array = MarkerArray()
        array.markers.append(marker)
        self.marker_pub.publish(array)

    # ------------------------------------------------------------------ #
    def _tick(self):
        now = self.get_clock().now().to_msg()
        scan = self._compute_scan()
        self._publish_scan(scan, now)
        self._publish_marker(scan, now)

        if scan["subsurface_obstacle"]:
            self.get_logger().info(
                f"RED -- obstacle at depth {scan['depth_estimate']}m, "
                f"density {scan['soil_density']}, confidence {scan['confidence']}"
            )


def main():
    rclpy.init()
    node = GPRSubsurfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
