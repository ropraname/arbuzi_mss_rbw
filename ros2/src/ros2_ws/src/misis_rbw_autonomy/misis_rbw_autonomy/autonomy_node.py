from __future__ import annotations

import math
from typing import Optional

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

from misis_rbw_autonomy.field_detector import FieldDetector, FieldObservation
from misis_rbw_autonomy.strategy import CompetitionStrategy, Pose2D


class MisisRbwAutonomyNode(Node):
    def __init__(self) -> None:
        super().__init__("misis_rbw_autonomy")

        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("debug_topic", "/misis_rbw/debug_image")
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("use_odom_pose", False)
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("match_duration_s", 90.0)
        self.declare_parameter("max_linear_speed", 0.24)
        self.declare_parameter("max_angular_speed", 1.4)
        self.declare_parameter("home_x", 0.18)
        self.declare_parameter("home_y", 0.18)
        self.declare_parameter("home_yaw", 0.0)

        self.bridge = CvBridge()
        self.detector = FieldDetector()
        self.latest_observation: Optional[FieldObservation] = None
        self.latest_pose: Optional[Pose2D] = None
        self.last_reason = ""

        home_pose = Pose2D(
            float(self.get_parameter("home_x").value),
            float(self.get_parameter("home_y").value),
            float(self.get_parameter("home_yaw").value),
        )
        self.strategy = CompetitionStrategy(
            max_linear=float(self.get_parameter("max_linear_speed").value),
            max_angular=float(self.get_parameter("max_angular_speed").value),
            match_duration_s=float(self.get_parameter("match_duration_s").value),
            home_pose=home_pose,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            10,
        )
        self.debug_pub = self.create_publisher(
            Image,
            str(self.get_parameter("debug_topic").value),
            10,
        )
        self.image_sub = self.create_subscription(
            Image,
            str(self.get_parameter("camera_topic").value),
            self.image_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_callback,
            10,
        )

        period = 1.0 / float(self.get_parameter("control_rate_hz").value)
        self.control_timer = self.create_timer(period, self.control_loop)
        self.get_logger().info("MISIS RBW autonomy node started")

    def image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        now = self.get_clock().now().nanoseconds / 1e9
        observation = self.detector.process_frame(frame, timestamp=now)
        self.latest_observation = observation

        if bool(self.get_parameter("publish_debug").value):
            debug_msg = self.bridge.cv2_to_imgmsg(observation.debug_frame, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

    def odom_callback(self, msg: Odometry) -> None:
        orientation = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.latest_pose = Pose2D(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
        )

    def control_loop(self) -> None:
        if self.latest_observation is None:
            self.publish_stop()
            return

        now = self.get_clock().now().nanoseconds / 1e9
        use_odom = bool(self.get_parameter("use_odom_pose").value)
        pose = self.latest_pose if use_odom else None
        command = self.strategy.compute_command(
            toys=self.latest_observation.toys,
            robots=self.latest_observation.robots,
            now_s=now,
            own_pose=pose,
        )

        twist = Twist()
        twist.linear.x = float(command.linear)
        twist.angular.z = float(command.angular)
        self.cmd_pub.publish(twist)

        if command.reason != self.last_reason:
            self.get_logger().info(
                f"{command.reason}: v={command.linear:.2f}, w={command.angular:.2f}, target={command.target}"
            )
            self.last_reason = command.reason

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MisisRbwAutonomyNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
