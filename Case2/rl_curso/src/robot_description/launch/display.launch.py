
import os
import launch
import launch_ros
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, Command, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    pkgPath = launch_ros.substitutions.FindPackageShare(package='robot_description').find('robot_description')
    urdfModelPath= os.path.join(pkgPath, 'urdf', 'robot.urdf.xacro')

    namespace = 'robot'

    params = {
        'robot_description': ParameterValue(
            Command(['xacro ', str(urdfModelPath)]), value_type=str
        )
    }
    
    robot_state_publisher_node =launch_ros.actions.Node(
        package='robot_state_publisher',
    	executable='robot_state_publisher',
        output='screen',
        namespace=namespace,
        parameters=[params]
    )   
    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=namespace,
        parameters=[params],
        condition=launch.conditions.UnlessCondition(LaunchConfiguration('gui'))
    )
    joint_state_publisher_gui_node = launch_ros.actions.Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        namespace=namespace,
        condition=launch.conditions.IfCondition(LaunchConfiguration('gui'))
    )
    
    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='gui', default_value='True',
                                            description='This is a flag for joint_state_publisher_gui'),
        launch.actions.DeclareLaunchArgument(name='model', default_value=urdfModelPath,
                                            description='Path to the urdf model file'),
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node
    ]) 
