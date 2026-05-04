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
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from audio_tts_msgs.msg import AudioStamped
from audio_tts_msgs.action import TTS
from tts_ros.utils import array_to_msg, get_msg_chunk
from std_msgs.msg import Bool
import time
import numpy as np
import subprocess
from std_srvs.srv import Trigger
import io
import tempfile
import requests
from io import BytesIO
from audio_tts_msgs.srv import SetVoice
import re

AUDIO_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
)

class TtsNode(Node):

    def __init__(self) -> None:
        super().__init__("tts_node")

        # Declare parameters with defaults
        self.declare_parameters(
            "",
            [
                ("frame_id", ""),
                ("workstation_url", "http://10.147.1.1"),
            ],
        )
        self.robot_speaks = False
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self._workstation_url = self.get_parameter("workstation_url").get_parameter_value().string_value.rstrip("/")

        self.stop_audio_client = self.create_client(Trigger, '/stop_audio_playback')

        # Goal queue for handling action server requests
        self._goal_queue = collections.deque()
        self._goal_queue_lock = threading.Lock()
        self._current_goal = None
        self._tts_execution_lock = threading.Lock()
        self._tts_goal_seq = 0

        # Publisher for audio output
        self._pub_rate = None
        self._pub_lock = threading.Lock()
        self.__player_pub = self.create_publisher(AudioStamped, "/tts/audio", AUDIO_QOS)
        self._robot_speaking_pub = self.create_publisher(Bool, "/robot_speaking", 1)
        self.voice_detected_sub = self.create_subscription(
            Bool, "/robot_speaking", self.on_robot_speaking, 1)
        self.create_service(SetVoice, "/tts/change_voice", self.change_voice_callback)
        self._current_voice = "infantil"
        self._robot_speaking_cv = threading.Condition()
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

    def change_voice_callback(self, request, response):
        new_voice = request.voice.strip()
        if not new_voice:
            response.success = False
            response.message = "Empty voice name not allowed"
            return response

        try:
            url = f"{self._workstation_url}/tts/change_voice"
            payload = {"voice": new_voice}
            r = requests.post(url, json=payload, timeout=3.0)
            if r.status_code == 200:
                self._current_voice = new_voice
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
        #if emotion not in self.emotion_embeddings:
        #    raise ValueError(f"Emotion '{emotion}' not found in the embeddings.")
        json_payload = {
            "text": text,
            "emotion": emotion,
            "temperature": temperature,
            "language": language#,
            #"rate": rate,
        }
        url = f"{self._workstation_url}/tts/read"
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

    def wait_for_audio_playback(self, goal_handle, timeout_sec=10.0) -> bool:
        self.get_logger().info("waiting for audio to finish")
        # Wait for audio_player to confirm it started playing
        time.sleep(0.3)
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
        with self._robot_speaking_cv:
            self.robot_speaks = bool(msg.data)
            self._robot_speaking_cv.notify_all()

    def _wait_for_player_cycle(
        self,
        goal_handle: ServerGoalHandle,
        *,
        goal_id: int,
        audio_duration_sec: float,
        start_timeout_sec: float = 2.0,
    ) -> bool:
        """Wait for audio_player to start and finish the just-published audio."""
        self.get_logger().info(f"[TTS:{goal_id}] waiting for audio to finish (~{audio_duration_sec:.1f}s)")
        wait_started_at = time.time()
        start_deadline = time.time() + max(0.1, start_timeout_sec)
        finish_deadline = time.time() + max(3.0, audio_duration_sec + start_timeout_sec + 2.0)
        saw_start = False

        with self._robot_speaking_cv:
            while not self.robot_speaks and time.time() < start_deadline:
                if goal_handle.is_cancel_requested:
                    if goal_handle.is_active:
                        goal_handle.canceled()
                    return False
                self._robot_speaking_cv.wait(timeout=0.05)

            saw_start = bool(self.robot_speaks)
            if not saw_start:
                self.get_logger().warn(
                    f"[TTS:{goal_id}] audio playback did not confirm start"
                )
            else:
                self.get_logger().info(
                    f"[TTS:{goal_id}] audio playback confirmed start after {time.time() - wait_started_at:.2f}s"
                )

        if not saw_start:
            self._publish_robot_speaking_fallback(goal_id, audio_duration_sec)
            return False

        with self._robot_speaking_cv:
            while self.robot_speaks and time.time() < finish_deadline:
                if goal_handle.is_cancel_requested:
                    if goal_handle.is_active:
                        goal_handle.canceled()
                    self.robot_speaks = False
                    return False
                self._robot_speaking_cv.wait(timeout=0.05)

            if self.robot_speaks:
                self.get_logger().warn(f"[TTS:{goal_id}] timeout waiting for audio playback to finish")
                return False

        self.get_logger().info(f"[TTS:{goal_id}] audio playback finished after {time.time() - wait_started_at:.2f}s")
        return True

    def _publish_robot_speaking_fallback(self, goal_id: int, duration_sec: float) -> None:
        """Simulate the robot_speaking True→False cycle when audio_player doesn't respond."""
        self.get_logger().warn(
            f"[TTS:{goal_id}] audio_player silent — publishing robot_speaking=False fallback"
        )
        msg_true = Bool()
        msg_true.data = True
        self._robot_speaking_pub.publish(msg_true)
        time.sleep(max(0.1, duration_sec))
        msg_false = Bool()
        msg_false.data = False
        self._robot_speaking_pub.publish(msg_false)

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
        with self._goal_queue_lock:
            self._tts_goal_seq += 1
            goal_id = self._tts_goal_seq

        if self._tts_execution_lock.locked():
            self.get_logger().info(f"[TTS:{goal_id}] waiting for previous TTS goal to finish")

        with self._tts_execution_lock:
            return self._execute_callback_locked(goal_handle, goal_id)

    def _execute_callback_locked(self, goal_handle: ServerGoalHandle, goal_id: int) -> TTS.Result:
        request: TTS.Goal = goal_handle.request
        text = request.text
        language = request.language or "es"
        emotion = request.emotion or "neutral"
        temperature = request.temperature
        rate = request.rate if request.rate > 0.0 else 1.0

        if not text.strip():
            self.get_logger().info(f"[TTS:{goal_id}] empty text goal; succeeding without playback")
            goal_handle.succeed()
            return TTS.Result()

        preview = text.replace("\n", " ").strip()
        if len(preview) > 100:
            preview = preview[:97] + "..."
        self.get_logger().info(
            f"[TTS:{goal_id}] accepted text_len={len(text)} language={language!r} "
            f"emotion={emotion!r} rate={rate:.2f} text={preview!r}"
        )

        return self.execute_xtts(goal_handle, goal_id, text, language, emotion, temperature, rate)

    def execute_xtts(
        self,
        goal_handle: ServerGoalHandle,
        goal_id: int,
        text: str,
        language: str,
        emotion: str,
        temperature: float,
        rate: float,
    ) -> TTS.Result:

        self.get_logger().info(
            f"[TTS:{goal_id}] generating audio | emotion={emotion}, temp={temperature}, rate={rate}"
        )

        try:
            # Call TTS backend
            out, sample_rate = self.generate_speech(
                text=text,
                emotion=emotion,
                language=language,
                temperature=temperature,
                rate=rate,
            )

            if out is None or sample_rate is None:
                self.get_logger().error("No audio returned from TTS server")
                if goal_handle.is_active:
                    goal_handle.abort()
                result = TTS.Result()
                result.text = text
                return result

            # Convert float32 [-1,1] to int16 PCM
            data = np.clip(out, -1.0, 1.0)
            data = (data * 32767.0).astype(np.int16)
            audio_duration_sec = len(data) / float(sample_rate)
            self.get_logger().info(
                f"[TTS:{goal_id}] generated samples={len(data)} sample_rate={sample_rate} "
                f"duration={audio_duration_sec:.2f}s"
            )

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

            # Publish audio
            with self._pub_lock:
                self.__player_pub.publish(msg)
                self.get_logger().info(f"[TTS:{goal_id}] published /tts/audio")

                # Publish feedback
                feedback = TTS.Feedback()
                feedback.audio = msg
                goal_handle.publish_feedback(feedback)

            # Wait for audio_player's actual /robot_speaking cycle when possible.
            success = self._wait_for_player_cycle(
                goal_handle,
                goal_id=goal_id,
                audio_duration_sec=audio_duration_sec,
            )

            if not success:
                if goal_handle.is_cancel_requested:
                    self.get_logger().info(f"[TTS:{goal_id}] canceled while waiting for playback")
                    goal_handle.canceled()
                elif goal_handle.is_active:
                    self.get_logger().warn(f"[TTS:{goal_id}] aborting because playback did not complete")
                    goal_handle.abort()
                return TTS.Result()

            if goal_handle.is_active:
                goal_handle.succeed()
                self.get_logger().info(f"[TTS:{goal_id}] goal succeeded")

            result = TTS.Result()
            result.text = text
            return result

        except Exception as e:
            self.get_logger().error(f"[TTS:{goal_id}] exception in execute_xtts: {e}")
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
