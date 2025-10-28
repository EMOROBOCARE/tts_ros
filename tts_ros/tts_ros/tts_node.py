#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023 Miguel Ángel González Santamarta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import wave
import pyaudio
import threading
import collections
import numpy as np
#from TTS.tts.models.xtts import Xtts
#from TTS.tts.configs.xtts_config import XttsConfig
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
from tts_ros.utils import array_to_msg, get_msg_chunk, concat_audios_with_silence
from std_msgs.msg import Bool
import time
import numpy as np
import subprocess
from std_srvs.srv import Trigger
import io
import tempfile
import requests
from io import BytesIO

import re

# regex to parse tags like: [happy] Hello world
EMOTION_TAG_RE =  re.compile(r"<expression\((\w+)\)>(.*?)</expression>", re.IGNORECASE | re.DOTALL)

EXPRESSION_MAP = {
    "happy": "happiness",
    "fear": "fear",
    "neutral": "neutral",
    "surprised": "surprise"
}

class TtsNode(Node):

    def __init__(self) -> None:
        super().__init__("tts_node")

        # Declare parameters with defaults
        self.declare_parameters(
            "",
            [
                ("frame_id", ""),
            ],
        )
        self.robot_speaks = False
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        self.stop_audio_client = self.create_client(Trigger, '/stop_audio_playback')

        # Goal queue for handling action server requests
        self._goal_queue = collections.deque()
        self._goal_queue_lock = threading.Lock()
        self._current_goal = None

        # Publisher for audio output
        self._pub_rate = None
        self._pub_lock = threading.Lock()
        self.__player_pub = self.create_publisher(AudioStamped, "/tts/audio", qos_profile_sensor_data)
        self.voice_detected_sub = self.create_subscription(
            Bool, "/robot_speaking", self.on_robot_speaking, 1)
        # Action server setup
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


    def generate_speech(self, text, emotion="neutral", language="es", temperature=0.9):
        log_str = f"Emotion: {emotion}\t Generated text: {text}\t"
        print(log_str)
        #if emotion not in self.emotion_embeddings:
        #    raise ValueError(f"Emotion '{emotion}' not found in the embeddings.")
        json_payload = {
            "text": text,
            "emotion": emotion,
            "temperature": temperature,
            "language": language
        }
        url = "http://10.147.19.11/tts/read"
        t0 = time.time()
        # Call local TTS HTTP server (/tts/read) with a sensible timeout
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

                # Map sampwidth to numpy dtype
                if sampwidth == 1:
                    dtype = np.uint8
                    # uint8 WAV is unsigned with 128 offset
                    raw = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                    audio_array = (raw - 128.0) / 128.0
                elif sampwidth == 2:
                    dtype = np.int16
                    raw = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                    audio_array = raw / 32767.0
                elif sampwidth == 4:
                    # assume 32-bit PCM
                    dtype = np.int32
                    raw = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                    audio_array = raw / 2147483647.0
                else:
                    self.get_logger().error(f"Unsupported sample width: {sampwidth}")
                    return None, None

                # If stereo, convert to mono by averaging channels
                if nchannels > 1:
                    audio_array = audio_array.reshape(-1, nchannels).mean(axis=1)

            except Exception as e:
                self.get_logger().error(f"Failed to parse WAV response: {e}")
                return None, None
        else:
            self.get_logger().error(f"Error {response.status_code}: {response.text}")

        log_str += f"Inference time: {round(time.time()-t0, 2)}"
        self.get_logger().info(log_str)
        return audio_array, sample_rate
        

    def parse_expressions(self, text: str):
        """
        Returns a list of tuples: (tts_emotion, segment_text)
        """
        segments = []
        for match in EMOTION_TAG_RE.finditer(text):
            emotion_tag = match.group(1).lower()
            segment_text = match.group(2).strip()
            tts_emotion = EXPRESSION_MAP.get(emotion_tag, "neutral")
            segments.append((tts_emotion, segment_text))
        return segments


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
                    # Call the stop_audio_playback service
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

        self.robot_speaks= msg.data

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
        language = request.language
        temperature = request.temperature

        if not text.strip():
            goal_handle.succeed()
            return TTS.Result()

        return self.execute_xtts(goal_handle, text, language, temperature)

    def execute_xtts(self, goal_handle: ServerGoalHandle, text: str, language: str, temperature: float) -> TTS.Result:

        self.get_logger().info("Generating Audio")

        try:
            segments = self.parse_expressions(text)
            audios = []

            if segments:
                for emotion, segment_text in segments:
                    out, rate = self.generate_speech(segment_text, emotion=emotion, language=language, temperature=temperature)
                    audios.append(out)
                out = concat_audios_with_silence(audios, rate)
            else:
                # If no expression tags, default to happy/happiness
                out, rate = self.generate_speech(text, emotion="happiness", language=language, temperature=temperature)

            if out is None:
                self.get_logger().error("No audio returned from TTS server")
                try:
                    goal_handle.abort()
                except Exception:
                    pass
                result = TTS.Result()
                result.text = text
                return result

            # 'out' is expected to be a float32 array in range [-1,1]
            data = np.clip(out, -1.0, 1.0)
            data = (data * 32767.0).astype(np.int16)

            # Create and publish full audio message at once
            audio_msg = array_to_msg(data)
            #if audio_msg is None:
            #    self.get_logger().error(f"Format {audio_format} unknown")
            #    goal_handle.abort()
            #    self.run_next_goal()
            #    return TTS.Result()

            msg = AudioStamped()
            msg.header.frame_id = self.frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.audio = audio_msg
            # set channels and rate from the source (default to mono/24000 if unknown)
            msg.audio.info.channels = 1
            msg.audio.info.chunk = get_msg_chunk(audio_msg)
            msg.audio.info.rate = rate if rate is not None else 24000
            self.get_logger().info("publish audio")
            with self._pub_lock:
                self.__player_pub.publish(msg)
                self.get_logger().info("published audio")
                self.robot_speaks = True #or can be subscribed from topic

                success = self.wait_for_audio_playback(goal_handle, timeout_sec=30.0)

                if not success:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        return TTS.Result()
                    else:
                        try:
                            if goal_handle.is_active:
                                goal_handle.canceled()
                        except Exception as e:
                            self.get_logger().warn(f"Could not abort: {e}")
                        return TTS.Result()

                if goal_handle.is_active:
                    goal_handle.succeed()
                    
                result = TTS.Result()
                result.text = text
                return result
        

        except Exception as e:
            self.get_logger().error(f"Exception: {e}")
            if not goal_handle.is_cancel_requested:
                try:
                    goal_handle.abort()
                except Exception as inner_e:
                    self.get_logger().warn(f"Failed to abort goal: {inner_e}")
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
