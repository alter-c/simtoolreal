#!/usr/bin/env python3
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import rospy
from geometry_msgs.msg import PoseStamped


LATEST_LOCK = threading.Lock()
LATEST_PAYLOAD: Optional[Dict[str, Any]] = None
PUBLISHERS: Dict[str, rospy.Publisher] = {}
LOG_LOCK = threading.Lock()
ALLOWED_TOPICS = {
    "/camera_frame/current_object_pose",
    "/robot_frame/current_object_pose",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Receive HTTP pose JSON and republish it as local ROS PoseStamped topics."
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8088, help="HTTP bind port.")
    parser.add_argument(
        "--endpoint",
        default="/api/current_object_pose",
        help="HTTP POST endpoint.",
    )
    parser.add_argument(
        "--publish_camera_pose",
        action="store_true",
        help="Also publish /camera_frame/current_object_pose if present.",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=1,
        help="ROS publisher queue size.",
    )
    parser.add_argument(
        "--save_log",
        default="/home/unitree/posejsoncurrent/published_poses.jsonl",
        help="JSONL file to save poses after they are published. Set empty string to disable.",
    )
    return parser.parse_args()


def get_publisher(topic: str, queue_size: int) -> rospy.Publisher:
    if topic not in PUBLISHERS:
        PUBLISHERS[topic] = rospy.Publisher(topic, PoseStamped, queue_size=queue_size)
    return PUBLISHERS[topic]


def pose_stamped_from_dict(msg_dict: Dict[str, Any]) -> PoseStamped:
    msg = PoseStamped()
    header = msg_dict.get("header", {})
    stamp = header.get("stamp", {})

    secs = stamp.get("secs")
    nsecs = stamp.get("nsecs")
    if secs is None or nsecs is None:
        msg.header.stamp = rospy.Time.now()
    else:
        msg.header.stamp = rospy.Time(int(secs), int(nsecs))

    msg.header.frame_id = str(header.get("frame_id", "robot_frame"))

    pose = msg_dict["pose"]
    position = pose["position"]
    orientation = pose["orientation"]

    msg.pose.position.x = float(position["x"])
    msg.pose.position.y = float(position["y"])
    msg.pose.position.z = float(position["z"])
    msg.pose.orientation.x = float(orientation["x"])
    msg.pose.orientation.y = float(orientation["y"])
    msg.pose.orientation.z = float(orientation["z"])
    msg.pose.orientation.w = float(orientation["w"])
    return msg


def pose_stamped_to_dict(topic: str, msg: PoseStamped) -> Dict[str, Any]:
    return {
        "topic": topic,
        "type": "geometry_msgs/PoseStamped",
        "msg": {
            "header": {
                "stamp": {
                    "secs": int(msg.header.stamp.secs),
                    "nsecs": int(msg.header.stamp.nsecs),
                },
                "frame_id": msg.header.frame_id,
            },
            "pose": {
                "position": {
                    "x": float(msg.pose.position.x),
                    "y": float(msg.pose.position.y),
                    "z": float(msg.pose.position.z),
                },
                "orientation": {
                    "x": float(msg.pose.orientation.x),
                    "y": float(msg.pose.orientation.y),
                    "z": float(msg.pose.orientation.z),
                    "w": float(msg.pose.orientation.w),
                },
            },
        },
    }


def write_pose_log(
    log_f: Optional[TextIO],
    payload: Dict[str, Any],
    published_messages: List[Dict[str, Any]],
) -> None:
    if log_f is None:
        return
    record = {
        "saved_at": time.time(),
        "payload_timestamp": payload.get("timestamp"),
        "frame_index": payload.get("frame_index"),
        "messages": published_messages,
    }
    with LOG_LOCK:
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()


def publish_payload(
    payload: Dict[str, Any],
    queue_size: int,
    publish_camera_pose: bool,
) -> Tuple[int, List[Dict[str, Any]]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("payload must contain a list field: messages")

    published_messages = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        topic = item.get("topic")
        msg_type = item.get("type")
        msg_dict = item.get("msg")

        if topic not in ALLOWED_TOPICS:
            continue
        if topic == "/camera_frame/current_object_pose" and not publish_camera_pose:
            continue
        if msg_type != "geometry_msgs/PoseStamped":
            raise ValueError(f"unsupported message type for {topic}: {msg_type}")
        if not isinstance(msg_dict, dict):
            raise ValueError(f"missing msg object for {topic}")

        ros_msg = pose_stamped_from_dict(msg_dict)
        get_publisher(topic, queue_size).publish(ros_msg)
        published_messages.append(pose_stamped_to_dict(topic, ros_msg))

    if not published_messages:
        raise ValueError("no publishable PoseStamped message found")
    return len(published_messages), published_messages


class PoseHandler(BaseHTTPRequestHandler):
    endpoint = "/api/current_object_pose"
    queue_size = 1
    publish_camera_pose = False
    log_f: Optional[TextIO] = None

    def _send_json(self, status: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True, "time": time.time()})
            return
        if self.path == self.endpoint:
            with LATEST_LOCK:
                payload = LATEST_PAYLOAD
            if payload is None:
                self._send_json(404, {"ok": False, "error": "no pose received yet"})
            else:
                self._send_json(200, {"ok": True, "payload": payload})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != self.endpoint:
            self._send_json(404, {"ok": False, "error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            published, published_messages = publish_payload(
                payload,
                queue_size=self.queue_size,
                publish_camera_pose=self.publish_camera_pose,
            )
            try:
                write_pose_log(self.log_f, payload, published_messages)
            except Exception as exc:
                rospy.logwarn("Published pose but failed to save log: %s", exc)
            global LATEST_PAYLOAD
            with LATEST_LOCK:
                LATEST_PAYLOAD = payload
            self._send_json(200, {"ok": True, "published": published})
        except Exception as exc:
            rospy.logwarn("Failed to handle pose POST: %s", exc)
            self._send_json(400, {"ok": False, "error": str(exc)})

    def log_message(self, fmt, *args):
        rospy.loginfo("%s - %s", self.address_string(), fmt % args)


def main():
    args = parse_args()
    rospy.init_node("pose_http_to_ros_bridge", anonymous=False)

    PoseHandler.endpoint = args.endpoint
    PoseHandler.queue_size = args.queue_size
    PoseHandler.publish_camera_pose = args.publish_camera_pose
    if args.save_log:
        log_path = Path(args.save_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        PoseHandler.log_f = log_path.open("a", encoding="utf-8")
        rospy.loginfo("Saving published poses to %s", log_path)

    server = ThreadingHTTPServer((args.host, args.port), PoseHandler)
    rospy.loginfo(
        "Pose HTTP to ROS bridge listening on http://%s:%d%s",
        args.host,
        args.port,
        args.endpoint,
    )
    rospy.loginfo("Publishing /robot_frame/current_object_pose as geometry_msgs/PoseStamped")
    if args.publish_camera_pose:
        rospy.loginfo("Also publishing /camera_frame/current_object_pose")

    server.timeout = 0.1
    try:
        while not rospy.is_shutdown():
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if PoseHandler.log_f is not None:
            PoseHandler.log_f.close()


if __name__ == "__main__":
    main()
