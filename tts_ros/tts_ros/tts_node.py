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
import numpy as np
import torch
import torchaudio
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from audio_tts_msgs.msg import AudioStamped
from audio_tts_msgs.action import TTS
from tts_ros.utils import data_to_msg, get_msg_chunk
from std_msgs.msg import Bool
import time
import io
import subprocess
from gtts import gTTS
from std_srvs.srv import Trigger


class TtsNode(Node):

    def __init__(self) -> None:
        super().__init__("tts_node")

        self.declare_parameters(
            "",
            [
                ("chunk", 4096),
                ("frame_id", ""),
                ("model", "tts_models/en/ljspeech/vits"),
                ("model_path", ""),
                ("config_path", ""),
                ("vocoder_path", ""),
                ("vocoder_config_path", ""),
                ("device", "cpu"),
                ("speaker_wav", ""),
                ("speaker", ""),
                ("stream", False),
                ("tts_engine", "gtts"),
            ],
        )

        self.robot_speaks = False
        self.chunk = self.get_parameter("chunk").get_parameter_value().integer_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.stream = self.get_parameter("stream").get_parameter_value().bool_value
        self.tts_engine = self.get_parameter("tts_engine").get_parameter_value().string_value
        self.speaker_wav = self.get_parameter("speaker_wav").get_parameter_value().string_value
        self.speaker = self.get_parameter("speaker").get_parameter_value().string_value
        self.stop_audio_client = self.create_client(Trigger, '/stop_audio_playback')

        if self.tts_engine == "xtts":
            model_dir = Path(get_package_share_directory("tts_ros")) / "config"
            checkpoint_path = model_dir / "model.pth"
            vocab_path = model_dir / "vocab.json"
            speakers_path = model_dir / "speakers_xtts.pth"
            config_path = model_dir / "config.json"

            self.get_logger().info("Loading XTTS model...")
            config = XttsConfig()
            config.load_json(str(config_path))
            self.tts = Xtts.init_from_config(config)
            self.tts.load_checkpoint(
                config,
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(vocab_path),
                speaker_file_path=str(speakers_path),
                use_deepspeed=False,
            )
            if not self.speaker_wav or not (Path(self.speaker_wav).exists() and Path(self.speaker_wav).is_file()):
                self.speaker_wav = None

            self.gpt_cond_latent, self.speaker_embedding = self.tts.get_conditioning_latents(
                audio_path=[self.speaker_wav]
            )

            if self.device == "cuda":
                self.tts.cuda()
            else:
                self.tts.cpu()

        self.__player_pub = self.create_publisher(AudioStamped, "/tts/audio", qos_profile_sensor_data)
        self.voice_detected_sub = self.create_subscription(Bool, "/robot_speaking", self.on_robot_speaking, 1)

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

    def wait_for_audio_playback(self, goal_handle, timeout_sec=10.0) -> bool:
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
        self.robot_speaks = msg.data

    def destroy_node(self) -> bool:
        self._action_server.destroy()
        return super().destroy_node()

    def goal_callback(self, goal_request: TTS.Goal) -> int:
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle: ServerGoalHandle) -> None:
        goal_handle.execute()

    def cancel_callback(self, goal_handle: ServerGoalHandle) -> int:
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle: ServerGoalHandle) -> TTS.Result:
        request = goal_handle.request
        text = request.text
        language = request.language

        if not text.strip():
            goal_handle.succeed()
            return TTS.Result()

        if self.tts_engine == "gtts":
            return self.execute_gtts(goal_handle, text, language)
        elif self.tts_engine == "xtts":
            return self.execute_xtts(goal_handle, text, language)

    def execute_gtts(self, goal_handle: ServerGoalHandle, text: str, language: str) -> TTS.Result:
        try:
            mp3_fp = io.BytesIO()
            tts = gTTS(text=text, lang=language or "en")
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)

            wav_bytes = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-ar", "24000", "-ac", "1", "-f", "wav", "pipe:1"],
                input=mp3_fp.read(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            ).stdout

            wav_fp = io.BytesIO(wav_bytes)
            wf = wave.open(wav_fp, "rb")
            channels = wf.getnchannels()
            rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            data = np.frombuffer(audio_data, dtype=np.int16)

            audio_msg = data_to_msg(data, pyaudio.paInt16)
            msg = AudioStamped()
            msg.header.frame_id = self.frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.audio = audio_msg
            msg.audio.info.channels = channels
            msg.audio.info.chunk = get_msg_chunk(audio_msg)
            msg.audio.info.rate = rate

            self.__player_pub.publish(msg)
            goal_handle.publish_feedback(TTS.Feedback(audio=msg))
            self.robot_speaks = True

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


    def execute_xtts(self, goal_handle: ServerGoalHandle, text: str, language: str) -> TTS.Result:
        try:
            out = self.tts.inference(
                text,
                language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                temperature=0.7,
            )
            wav = torch.tensor(out["wav"]).unsqueeze(0)
            data = wav.numpy().flatten()
            data = np.clip(data, -1, 1)
            data = (data * 32767).astype(np.int16)

            audio_msg = data_to_msg(data, pyaudio.paInt16)
            msg = AudioStamped()
            msg.header.frame_id = self.frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.audio = audio_msg
            msg.audio.info.channels = 1
            msg.audio.info.chunk = get_msg_chunk(audio_msg)
            msg.audio.info.rate = 24000

            self.__player_pub.publish(msg)
            goal_handle.publish_feedback(TTS.Feedback(audio=msg))
            self.robot_speaks = True

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
            self.get_logger().error(f"[gTTS] Exception: {e}")
            if not goal_handle.is_cancel_requested:
                try:
                    goal_handle.canceled()
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
