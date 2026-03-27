import time
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

from ament_index_python.packages import get_package_share_directory

from .utils import quat_to_yaw, yaw_to_quat, print_config_rich
from .transform import transform_points

@dataclass
class OdomData:
    # position
    x: float
    y: float
    z: float
    yaw: float
    
    # velocity
    v: float
    w: float

@dataclass
class ROSParameters:
    # topics
    scan_topic:str = "/scan"
    odom_topic:str = "/odom"
    cmd_vel_topic:str = "/cmd_vel"
    
    # services
    set_state_service:str  = "/set_entity_state"
    reset_world_service:str  = "/set_entity_state"
    
    # simulation
    use_sim_time:bool = True
    
    # reinforcement learning parameters
    env_config:str  = "params/env_config.yaml"
    
class RosInterface:
    """
      - subscribes to /robot/scan (LaserScan)
      - subscribes to /robot/odom (Odometry)
      - publishes     /robot/cmd_vel (Twist)
      - uses Gazebo services (set_entity_state; reset_world)
    """

    def __init__(self, args=None):        
        # node
        if not rclpy.ok():
            rclpy.init(args=args)

        self.node = Node("robot_navigation_node")
        
        # parameters
        self.params:ROSParameters = self._load_parameters()        
        
        # messages
        self._scan_msg: Optional[LaserScan] = None
        self._odom_msg: Optional[Odometry] = None

        self._last_scan_time = None
        self._last_odom_time = None

        # subscribers
        self._scan_sub = self.node.create_subscription(LaserScan, self.params.scan_topic, self._on_scan, 10)
        self._odom_sub = self.node.create_subscription(Odometry,  self.params.odom_topic, self._on_odom, 10)
        
        # publishers
        self._cmd_pub = self.node.create_publisher(Twist, self.params.cmd_vel_topic, 10)

        # services
        self._set_state_cli = self.node.create_client(SetEntityState, self.params.set_state_service)
        self._reset_world_cli = self.node.create_client(Empty, self.params.reset_world_service)
        
        # tf2 transformation
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

    def _load_parameters(self)->ROSParameters:
        
        if not self.node.has_parameter("scan_topic"):
            self.node.declare_parameter("scan_topic", "/scan")
        if not self.node.has_parameter("odom_topic"):
            self.node.declare_parameter("odom_topic", "/odom")
        if not self.node.has_parameter("cmd_vel_topic"):
            self.node.declare_parameter("cmd_vel_topic", "/cmd_vel")
        if not self.node.has_parameter("set_state_service"):
            self.node.declare_parameter("set_state_service", "/set_entity_state")
        if not self.node.has_parameter("reset_world_service"):
            self.node.declare_parameter("reset_world_service", "/reset_world")
        if not self.node.has_parameter("use_sim_time"):
            self.node.declare_parameter("use_sim_time", True)
        if not self.node.has_parameter("env_config"):
            self.node.declare_parameter("env_config", "params/env_config.yaml")
        
        params = ROSParameters(
            scan_topic=self.node.get_parameter("scan_topic").value,
            odom_topic=self.node.get_parameter("odom_topic").value,
            cmd_vel_topic=self.node.get_parameter("cmd_vel_topic").value,
            set_state_service=self.node.get_parameter("set_state_service").value,
            reset_world_service=self.node.get_parameter("reset_world_service").value,
            use_sim_time=self.node.get_parameter("use_sim_time").value,
            env_config=self.node.get_parameter("env_config").value,
        )
        
        print_config_rich(params)
        
        params.env_config = os.path.join(
            self.share_directory,
            params.env_config
        )
        
        return params
    
    @property
    def share_directory(self)->str:
        return get_package_share_directory("robot_navigation")

    def _on_scan(self, msg: LaserScan) -> None:
        """
            /robot/scan callback
        
        - subscribes to sensor observation from LiDAR    

        Args:
            msg (LaserScan): scan message
        """
        self._scan_msg = msg
        self._last_scan_time = self.node.get_clock().now()

    def _on_odom(self, msg: Odometry) -> None:
        """
            /robot/odom callback
        
        - subscribes to odometry observation from simulator    

        Args:
            msg (Odometry): odometry message
        """
        self._odom_msg = msg
        self._last_odom_time = self.node.get_clock().now()


    def spin_once(self, timeout_sec: float = 0.0) -> None:
        """
            ROS spin (ros loop step)
        Args:
            timeout_sec (float, optional): timeout for spin. Defaults to 0.0.
        """
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)


    def wait_for_topics(self, timeout_sec: float = 10.0) -> None:
        """
            busy wait for messages
        
        - used when start new episode after restart the gazebo

        Args:
            timeout_sec (float, optional): timeout for the busy wait. Defaults to 10.0.

        Raises:
            TimeoutError: when it reach the timeout and does not receive a new message from sensors
        """
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            self.spin_once(timeout_sec=0.05)
            if self._scan_msg is not None and self._odom_msg is not None:
                return
        raise TimeoutError("Timed out waiting for /robot/scan and /robot/odom.")

    def publish_cmd(self, v: float, w: float) -> None:
        """
            publish action
            
        - v (float) : linear velocity
        - w (float) : angular velocity
        
        Args:
            v (float): linear velocity
            w (float): angular velocity
        """
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self._cmd_pub.publish(msg)

    def stop_robot(self, n: int = 3) -> None:
        """
            stop the robot by publishing velocity 0.0
        Args:
            n (int, optional): number of messages with zeroed values. Defaults to 3.
        """
        for _ in range(n):
            self.publish_cmd(0.0, 0.0)
            self.spin_once(timeout_sec=0.02)

    def get_scan(self, transform:bool=True) -> np.ndarray:
        """
            sanity check for scan messages       
        Raises:
            RuntimeError: when you dont have new /robot/scan messages

        Returns:
            LaserScan: scan messages
        """
        if self._scan_msg is None:
            raise RuntimeError("No scan received yet.")
        
        return np.array(self._scan_msg.ranges, dtype=np.float32)

    def get_odom(self) -> OdomData:
        """
            convert odom message to odom data
            
        - useful to convert quaternion orientation to euler (roll, pitch, yaw)

        Returns:
            OdomData: odom data (x, y, z, yaw, linear_vel, angular_vel)
        """
        if self._odom_msg is None:
            raise RuntimeError("No odom received yet.")
        
        odom = self._odom_msg
        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        q = odom.pose.pose.orientation
        yaw = float(quat_to_yaw(q.x, q.y, q.z, q.w))
        v = float(odom.twist.twist.linear.x)
        w = float(odom.twist.twist.angular.z)
        return OdomData(x=x, y=y, z=0.05, yaw=yaw, v=v, w=w)

    def _wait_for_service(self, client, timeout_sec: float) -> bool:
        """
            timeout for gazebo services
            
        - restart simulation or change agent state
        Args:
            client (Node): ROS node
            timeout_sec (float): timeout 

        Returns:
            bool: true if the service is on, false otherwise
        """
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if client.wait_for_service(timeout_sec=0.2):
                return True
            self.spin_once(timeout_sec=0.01)
        return False

    def try_reset_world(self, timeout_sec: float = 1.0) -> bool:
        """
            try to reset the simulation
            
        Args:
            timeout_sec (float, optional): _description_. Defaults to 1.0

        Returns:
            bool: _description_
        """
        if not self._wait_for_service(self._reset_world_cli, timeout_sec=timeout_sec):
            return False
        req = Empty.Request()
        fut = self._reset_world_cli.call_async(req)
        while rclpy.ok() and not fut.done():
            self.spin_once(timeout_sec=0.05)
        return fut.done() and (fut.result() is not None)

    def set_entity_state(
        self,
        entity_name: str,
        x: float,
        y: float,
        yaw: float,
        z: float = 0.05,
        reference_frame: str = "world",
        timeout_sec: float = 3.0,
    ) -> None:
        """
            Set the position of the robot at the given (x, y, z, yaw)

        Args:
            entity_name (str): name of the robot in gazebo world (robot)
            x (float): x position
            y (float): y position
            yaw (float): orientation
            z (float, optional): z position (elevation). Defaults to 0.05
            reference_frame (str, optional): reference frame in gazebo world. Defaults to "world"
            timeout_sec (float, optional): timeout for service call. Defaults to 3.0 (seconds)

        Raises:
            TimeoutError: if service call fails due timeout
            RuntimeError: if service returns a void response
        """
        if not self._wait_for_service(self._set_state_cli, timeout_sec=timeout_sec):
            raise TimeoutError(f"SetEntityState service not available: {self.params.set_state_service}")

        state = EntityState()
        state.name = entity_name
        state.reference_frame = reference_frame

        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        
        qx, qy, qz, qw = yaw_to_quat(yaw)
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw

        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        req = SetEntityState.Request()
        req.state = state

        fut = self._set_state_cli.call_async(req)
        while rclpy.ok() and not fut.done():
            self.spin_once(timeout_sec=0.05)

        if fut.result() is None:
            raise RuntimeError("SetEntityState failed (no response).")

    def wait_sim_time(self, dt_sim: float, timeout_sec: float = 2.0) -> None:
        """
            Busy wait dt time in simulation reference (simulation time)
        
        Args:
            dt_sim (float): delay to wait (simulation time)
            timeout_sec (float): timeout (system time)
        """
        start = self.node.get_clock().now() # /clock 
        t0 = time.time() # system time
        target_ns = int(dt_sim * 1e9) # sec to nanosec

        while (time.time() - t0) < timeout_sec:
            self.spin_once(timeout_sec=0.01)
            now = self.node.get_clock().now()
            if (now - start).nanoseconds >= target_ns:
                return

    def close(self) -> None:
        """
            stop robot and destroy ROS node
        """
        try:
            self.stop_robot()
        except Exception:
            pass
        self.node.destroy_node()