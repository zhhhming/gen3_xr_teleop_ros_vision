#!/usr/bin/env python3
"""Full system launch: start sensors + XR publisher, then gate Gen3 teleop until topics are ready."""

import os
import time
from pathlib import Path
import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    LogInfo,
    TimerAction,
    IncludeLaunchDescription,
)
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# ----------------------------
# Helpers (kept from your file)
# ----------------------------
def _resolve_path(path_str: str, base_directory: Path) -> str:
    """Resolve a potentially relative path using the package share directory."""
    if not path_str:
        return ''
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((base_directory / path).resolve())


def _parse_bool(value: str, default: bool) -> bool:
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        return value
    return value.lower() in ('true', '1', 'yes', 'on')


# ----------------------------
# Core launch assembly
# ----------------------------
def launch_setup(context, *args, **kwargs):
    pkg_share = Path(get_package_share_directory('gen3_xr_teleop'))
    config_path = Path(LaunchConfiguration('config_file').perform(context))

    if not config_path.is_absolute():
        config_path = (pkg_share / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    teleop_cfg = config.get('teleop', {})
    xr_cfg = config.get('xr_publisher', {})
    rviz_cfg = config.get('rviz', {})

    # Overrides
    robot_ip = LaunchConfiguration('robot_ip').perform(context) or teleop_cfg.get('robot_ip', '192.168.1.10')
    urdf_path = _resolve_path(LaunchConfiguration('urdf_path').perform(context) or teleop_cfg.get('urdf_path', ''), pkg_share)
    log_file = _resolve_path(LaunchConfiguration('log_file').perform(context) or teleop_cfg.get('log_file', 'joint_tracking_data.csv'), pkg_share)

    publish_rate = xr_cfg.get('publish_rate', 200.0)
    if LaunchConfiguration('publish_rate').perform(context):
        publish_rate = float(LaunchConfiguration('publish_rate').perform(context))
    frame_id = LaunchConfiguration('frame_id').perform(context) or xr_cfg.get('frame_id', 'vr_origin')

    rviz_enabled = _parse_bool(LaunchConfiguration('rviz_enabled').perform(context), rviz_cfg.get('enabled', True))
    rviz_config_path = _resolve_path(LaunchConfiguration('rviz_config').perform(context) or rviz_cfg.get('config', ''), pkg_share)

    # Nodes to start first
    actions = [
        LogInfo(msg=f'[Launch] Using config: {config_path}')
    ]

    # 1) RealSense
    actions.append(
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense2_camera_node',
            output='screen',
        )
    )

    # 2) Orbbec Astra (include the vendor launch)
    orbbec_share = get_package_share_directory('orbbec_camera')
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(orbbec_share, 'launch', 'astra.launch.py')),
        )
    )

    # 3) DIGIT (assume one node publishes both /digit/D20583/image_raw and /digit/D20584/image_raw)
    actions.append(
        Node(
            package='digit_ros2',
            executable='digit_publisher_node',
            name='digit_publisher_node',
            output='screen',
        )
    )

    # 4) XR Publisher
    xr_parameters = {
        'publish_rate': publish_rate,
        'frame_id': frame_id,
    }
    actions.append(
        Node(
            package='gen3_xr_teleop',
            executable='xr_publisher_node.py',
            name='xr_publisher_node',
            output='screen',
            parameters=[xr_parameters],
        )
    )

    # Optional: RViz
    if rviz_enabled:
        rviz_args = []
        if rviz_config_path:
            rviz_args.extend(['-d', rviz_config_path])
        actions.append(
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=rviz_args,
            )
        )

    # 5) Wait until all are ready, then (after 3s) start Gen3 Teleop
    actions.append(OpaqueFunction(function=_gate_and_launch_teleop))

    return actions


def _gate_and_launch_teleop(context, *args, **kwargs):
    """
    Block until:
      - RealSense publishes /camera/camera/color/image_raw
      - Orbbec publishes /camera/color/image_raw
      - DIGIT publishes both /digit/D20583/image_raw and /digit/D20584/image_raw
      - XR publisher node 'xr_publisher_node' is publishing on >= 5 topics
    Then wait 3s and start the teleop node.
    If timeout (60s), warn and still start after 3s to avoid deadlock.
    """
    import rclpy
    from rclpy.node import Node as RclpyNode

    # Topics to check for specific devices
    required_topics_any_publisher = [
        '/camera/camera/color/image_raw',     # RealSense (your exact path)
        '/camera/color/image_raw',            # Orbbec Astra
        '/digit/D20583/image_raw',
        '/digit/D20584/image_raw',
    ]

    xr_node_name = 'xr_publisher_node'
    xr_min_topics = 5
    timeout_sec = 60.0
    poll_dt = 0.5

    rclpy.init(args=None)
    mon = RclpyNode('startup_monitor')

    start_ts = time.time()
    ready = False

    mon.get_logger().info('[Gate] Waiting for sensors & XR publisher to be ready...')

    try:
        while True:
            # Check specific topics have at least one publisher
            devices_ok = True
            for t in required_topics_any_publisher:
                pubs = mon.get_publishers_info_by_topic(t)
                if len(pubs) == 0:
                    devices_ok = False
                    break

            # Count how many topics are currently published by the XR publisher node
            xr_topic_count = 0
            for topic_name, _ in mon.get_topic_names_and_types():
                infos = mon.get_publishers_info_by_topic(topic_name)
                # If any publisher for this topic is from xr_publisher_node, count it
                if any(info.node_name == xr_node_name for info in infos):
                    xr_topic_count += 1

            xr_ok = xr_topic_count >= xr_min_topics

            if devices_ok and xr_ok:
                mon.get_logger().info(f'[Gate] Topics ready. XR publisher topics={xr_topic_count} (>= {xr_min_topics}).')
                ready = True
                break

            if (time.time() - start_ts) > timeout_sec:
                mon.get_logger().warn(
                    f'[Gate][TIMEOUT] Not all requirements met in {timeout_sec:.0f}s. '
                    f'XR topics seen={xr_topic_count}. Will start teleop anyway after 3s.'
                )
                break

            time.sleep(poll_dt)

    finally:
        mon.destroy_node()
        rclpy.shutdown()

    # After readiness (or timeout), wait extra 3 seconds before launching teleop
    delay_and_launch = TimerAction(
        period=3.0,
        actions=[
            LogInfo(
                msg='[Gate] Starting Gen3 teleop node '
                    f'{"(ready confirmed)" if ready else "(FORCED AFTER TIMEOUT)"}...'
            ),
            _make_teleop_node(context),
        ],
    )
    return [delay_and_launch]


def _make_teleop_node(context):
    """Create the final teleop node using the same argument handling as your original file."""
    pkg_share = Path(get_package_share_directory('gen3_xr_teleop'))
    # pull from launch configurations (same names as earlier)
    robot_ip = LaunchConfiguration('robot_ip').perform(context) or '192.168.1.10'
    urdf_path = LaunchConfiguration('urdf_path').perform(context)
    log_file = LaunchConfiguration('log_file').perform(context)

    # If relative, resolve from package share
    if urdf_path:
        urdf_path = _resolve_path(urdf_path, pkg_share)
    if log_file:
        log_file = _resolve_path(log_file, pkg_share)

    teleop_args = [robot_ip]
    if urdf_path:
        teleop_args.append(urdf_path)
    if log_file:
        teleop_args.append(log_file)

    return Node(
        package='gen3_xr_teleop',
        executable='gen3_xr_teleop_node',
        name='gen3_xr_teleop_node',
        output='screen',
        arguments=teleop_args,
    )


# ----------------------------
# Entry point
# ----------------------------
def generate_launch_description():
    pkg_share = Path(get_package_share_directory('gen3_xr_teleop'))
    default_config = str(pkg_share / 'config' / 'teleop_config.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to YAML configuration file for the teleop launch.'
        ),
        DeclareLaunchArgument(
            'robot_ip',
            default_value='',
            description='Override robot IP address defined in config file.'
        ),
        DeclareLaunchArgument(
            'urdf_path',
            default_value='',
            description='Override URDF path defined in config file.'
        ),
        DeclareLaunchArgument(
            'log_file',
            default_value='',
            description='Override log file path defined in config file.'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='',
            description='Override XR publisher rate (Hz) from config file.'
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='',
            description='Override XR pose frame id from config file.'
        ),
        DeclareLaunchArgument(
            'rviz_enabled',
            default_value='',
            description='Override RViz enabled flag from config file.'
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value='',
            description='Override RViz config path from config file.'
        ),
        OpaqueFunction(function=launch_setup),
    ])