#!/usr/bin/env python3
# MIT License
# ... (license header unchanged)

import wave
import pyaudio
import threading
import collections
import numpy as np
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from audio_tts_msgs.msg import AudioStamped
from audio_tts_msgs.action import TTS
from tts_ros.utils import array_to_msg, get_msg_chunk
from std_msgs.msg import Bool
import time
import subprocess
from std_srvs.srv import Trigger
import requests
from io import BytesIO
from audio_tts_msgs.srv import SetVoice
from rcl_interfaces.msg import ParameterEvent, SetParametersResult 
import re
from std_srvs.srv import Trigger
import os

class TtsNode(Node):
    def __init__(self) -> None:
        super().__init__("tts_node")

        self.declare_parameters(
            "",
            [
                ("frame_id", ""),
                ("tts_ip", os.getenv("TTS_IP", "10.147.1.1")),
            ],
        )

        self._tts_ip = (
            self.get_parameter("tts_ip").get_parameter_value().string_value
        )
        self.get_logger().info(f"TTS backend IP: {self._tts_ip}")

        # Register runtime parameter change callback
        self.add_on_set_parameters_callback(self._on_set_parameters_callback)

        self.robot_speaks = False
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.stop_audio_client = self.create_client(Trigger, "/stop_audio_playback")
        self.create_service(Trigger, "/tts/get_voice", self.get_voice_callback)
        self._goal_queue = collections.deque()
        self._goal_queue_lock = threading.Lock()
        self._current_goal = None

        self._pub_rate = None
        self._pub_lock = threading.Lock()
        self.__player_pub = self.create_publisher(
            AudioStamped, "/tts/audio", qos_profile_sensor_data
        )
        self.declare_parameter("current_voice", "infantil")
        self.voice_detected_sub = self.create_subscription(
            Bool, "/robot_speaking", self.on_robot_speaking, 1
        )
        self.create_service(SetVoice, "/tts/change_voice", self.change_voice_callback)
        self._current_voice = self.get_parameter("current_voice").value

        self._action_server = ActionServer(
            self,
            TTS,
            "say",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info("TTS node started")

    # ------------------------------------------------------------------
    # Parameter change handler
    # ------------------------------------------------------------------
    def _on_set_parameters_callback(self, params):
        result = SetParametersResult(successful=True)
        for param in params:
            if param.name == "tts_ip":
                new_ip = param.value.strip()
                # Basic sanity check — reject empty strings
                if not new_ip:
                    result.successful = False
                    result.reason = "tts_ip cannot be empty"
                    return result
                self._tts_ip = new_ip
                self.get_logger().info(f"TTS backend IP updated to: {self._tts_ip}")
        return result

    # ------------------------------------------------------------------
    # Helper: build base URL from current IP
    # ------------------------------------------------------------------
    def _tts_base_url(self) -> str:
        return f"http://{self._tts_ip}"

    # ------------------------------------------------------------------
    def get_voice_callback(self, request, response):
        response.success = True
        response.message = self._current_voice
        return response
    
    def change_voice_callback(self, request, response):
        new_voice = request.voice.strip()
        if not new_voice:
            response.success = False
            response.message = "Empty voice name not allowed"
            return response
        try:
            url = f"{self._tts_base_url()}/tts/change_voice" 
            payload = {"voice": new_voice}
            r = requests.post(url, json=payload, timeout=3.0)
            if r.status_code == 200:
                self._current_voice = new_voice
                self.set_parameters([
                    rclpy.parameter.Parameter(
                        "current_voice",
                        rclpy.Parameter.Type.STRING,
                        new_voice
                    )
                ])
                response.success = True
                response.message = f"Voice changed to '{new_voice}'"
                self.get_logger().info(response.message)
            else:
                response.success = False
                response.message = f"TTS server returned {r.status_code}: {r.text}"
                self.get_logger().error(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Failed to change voice: {e}"
            self.get_logger().error(response.message)
        return response

    def generate_speech(self, text, emotion="neutral", language="es", temperature=0.9, rate=1.0):
        log_str = f"Emotion: {emotion}\t Generated text: {text}\t"
        print(log_str)

        json_payload = {
            "text": text,
            "emotion": emotion,
            "temperature": temperature,
            "language": language,
            "speed": 1.0,
        }

        url = f"{self._tts_base_url()}/tts/expressive_read" 
        t0 = time.time()

        try:
            response = requests.post(url, json=json_payload, timeout=10.0)
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"TTS request failed: {e}")
            return None, None

        audio_array = None
        sample_rate = None

        if response.status_code == 200:
            try:
                with wave.open(BytesIO(response.content), "rb") as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    nchannels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()

                if sampwidth == 1:
                    raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                    audio_array = (raw - 128.0) / 128.0
                elif sampwidth == 2:
                    raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    audio_array = raw / 32767.0
                elif sampwidth == 4:
                    raw = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
                    audio_array = raw / 2147483647.0
                else:
                    self.get_logger().error(f"Unsupported sample width: {sampwidth}")
                    return None, None

                if nchannels > 1:
                    audio_array = audio_array.reshape(-1, nchannels).mean(axis=1)

            except Exception as e:
                self.get_logger().error(f"Failed to parse WAV response: {e}")
                return None, None
        else:
            self.get_logger().error(f"Error {response.status_code}: {response.text}")

        log_str += f"Inference time: {round(time.time() - t0, 2)}"
        self.get_logger().info(log_str)
        return audio_array, sample_rate

    # ------------------------------------------------------------------
    # Remaining methods unchanged
    # ------------------------------------------------------------------
    def wait_for_audio_playback(self, goal_handle, timeout_sec=10.0) -> bool:
        self.get_logger().info("waiting for audio to finish")
        start_time = time.time()
        canceled = False
        while self.robot_speaks:
            if goal_handle.is_cancel_requested and not canceled:
                try:
                    goal_handle.canceled()
                    self.get_logger().info("Goal canceled, stopping playback")
                    canceled = True
                    if self.stop_audio_client.wait_for_service(timeout_sec=1.0):
                        req = Trigger.Request()
                        self.stop_audio_client.call_async(req)
                    else:
                        self.get_logger().warn("stop_audio_playback service not available")
                except Exception as e:
                    self.get_logger().warn(f"Could not cancel: {e}")
                self.robot_speaks = False
                return False
            if time.time() - start_time > timeout_sec:
                self.get_logger().warn("Timeout waiting for audio playback to finish.")
                return False
            time.sleep(0.1)
        return True

    def on_robot_speaking(self, msg):
        self.robot_speaks = msg.data

    def destroy_node(self) -> bool:
        self._action_server.destroy()
        return super().destroy_node()

    def goal_callback(self, goal_request: ServerGoalHandle) -> int:
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle: ServerGoalHandle) -> None:
        goal_handle.execute()

    def cancel_callback(self, goal_handle: ServerGoalHandle) -> int:
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle: ServerGoalHandle) -> TTS.Result:
        request: TTS.Goal = goal_handle.request
        text = request.text
        language = request.language or "es"
        emotion = request.emotion or "neutral"
        temperature = request.temperature
        rate = request.rate if request.rate > 0.0 else 1.0

        if not text.strip():
            goal_handle.succeed()
            return TTS.Result()

        return self.execute_xtts(goal_handle, text, language, emotion, temperature, rate)

    def execute_xtts(self, goal_handle, text, language, emotion, temperature, rate) -> TTS.Result:
        self.get_logger().info(
            f"Generating audio | emotion={emotion}, temp={temperature}, rate={rate}"
        )
        try:
            out, sample_rate = self.generate_speech(
                text=text, emotion=emotion, language=language,
                temperature=temperature, rate=rate,
            )
            if out is None or sample_rate is None:
                self.get_logger().error("No audio returned from TTS server")
                if goal_handle.is_active:
                    goal_handle.abort()
                result = TTS.Result()
                result.text = text
                return result

            data = np.clip(out, -1.0, 1.0)
            data = (data * 32767.0).astype(np.int16)
            audio_msg = array_to_msg(data)

            if audio_msg is None:
                self.get_logger().error("Failed to convert audio array to ROS message")
                if goal_handle.is_active:
                    goal_handle.abort()
                return TTS.Result()

            msg = AudioStamped()
            msg.header.frame_id = self.frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.audio = audio_msg
            msg.audio.info.channels = 1
            msg.audio.info.chunk = get_msg_chunk(audio_msg)
            msg.audio.info.rate = sample_rate

            with self._pub_lock:
                self.__player_pub.publish(msg)
                self.robot_speaks = True
                feedback = TTS.Feedback()
                feedback.audio = msg
                goal_handle.publish_feedback(feedback)

            success = self.wait_for_audio_playback(goal_handle, timeout_sec=30.0)
            if not success:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                elif goal_handle.is_active:
                    goal_handle.abort()
                return TTS.Result()

            if goal_handle.is_active:
                goal_handle.succeed()

            result = TTS.Result()
            result.text = text
            return result

        except Exception as e:
            self.get_logger().error(f"Exception in execute_xtts: {e}")
            if goal_handle.is_active and not goal_handle.is_cancel_requested:
                try:
                    goal_handle.abort()
                except Exception:
                    pass
            result = TTS.Result()
            result.text = text
            return result


def main():
    rclpy.init()
    node = TtsNode()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()