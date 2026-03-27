import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import xacro


def generate_launch_description():
    gui = LaunchConfiguration("gui")
    world = LaunchConfiguration("world")
    robot_name = LaunchConfiguration("robot_name")

    declare_gui = DeclareLaunchArgument("gui", default_value="true", description="Run gzclient (GUI)")
    declare_world = DeclareLaunchArgument(
        "world",
        default_value=os.path.join(
            get_package_share_directory("robot_gazebo"),
            "worlds",
            "bookstore.world",
        ),
        description="World file",
    )
    declare_robot = DeclareLaunchArgument("robot_name", default_value="robot", description="Spawned robot entity name")

    # --- Robot description ---
    desc_share = get_package_share_directory("robot_description")
    xacro_path = os.path.join(desc_share, "urdf", "robot.urdf.xacro")
    robot_description = xacro.process_file(xacro_path).toxml()

    # --- AWS model paths   ---
    aws_bookstore_share = get_package_share_directory("aws_robomaker_bookstore_world")

    set_gazebo_model_path = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=os.pathsep.join([
            os.environ.get("GAZEBO_MODEL_PATH", ""),
            os.path.join(aws_bookstore_share, "models")
        ])
    )

    set_gazebo_resource_path = SetEnvironmentVariable(
        name="GAZEBO_RESOURCE_PATH",
        value=os.pathsep.join([
            os.environ.get("GAZEBO_RESOURCE_PATH", ""),
            aws_bookstore_share
        ])
    )

    # --- Gazebo server with ROS init ---
    gzserver = ExecuteProcess(
        cmd=[
            "gzserver",
            "--verbose",
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
            world,
        ],
        output="screen",
    )

    gzclient = ExecuteProcess(
        condition=IfCondition(gui),
        cmd=["gzclient", "--verbose"],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace="robot",
        output="screen",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
    )
    
    goal_marker_spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity", "goal_marker",
            "-file", os.path.join(get_package_share_directory("robot_gazebo"), "models", "goal_marker.sdf"),
            "-x", "0.0", "-y", "0.0", "-z", "0.0",
        ],
    )

    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        namespace="robot",
        arguments=[
            "-entity", robot_name,
            "-topic", "robot_description",
            "-x", "-4.0",
            "-y", "-4.0",
            "-z", "0.05",
            "-Y", "0.0",
        ],
    )

    delayed_spawn = TimerAction(period=2.0, actions=[spawn,goal_marker_spawn])

    return LaunchDescription(
        [
            declare_gui,
            declare_world,
            declare_robot,
            set_gazebo_model_path,
            set_gazebo_resource_path,
            gzserver,
            gzclient,
            robot_state_publisher,
            delayed_spawn,
        ]
    )