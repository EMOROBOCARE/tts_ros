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
import tempfile
import threading
import collections
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
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from audio_tts_msgs.msg import AudioStamped
from audio_tts_msgs.action import TTS
from tts_ros.utils import data_to_msg, get_msg_chunk
from std_msgs.msg import Bool
import time
import os
import re 
import numpy as np

EMOTION_TAGS = ["fear", "happiness", "neutral", "surprise"]


class TtsNode(Node):

    def __init__(self) -> None:
        super().__init__("tts_node")

        # Declare parameters with defaults
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
            ],
        )
        self.robot_speaks = False
        self.chunk = self.get_parameter("chunk").get_parameter_value().integer_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        model = self.get_parameter("model").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        config_path_param = self.get_parameter("config_path").get_parameter_value().string_value
        vocoder_path = self.get_parameter("vocoder_path").get_parameter_value().string_value
        vocoder_config_path = self.get_parameter("vocoder_config_path").get_parameter_value().string_value

        self.speaker_wav = self.get_parameter("speaker_wav").get_parameter_value().string_value
        self.speaker = self.get_parameter("speaker").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.stream = self.get_parameter("stream").get_parameter_value().bool_value

        # Load model files from package share directory + config folder
        model_dir = Path(get_package_share_directory("tts_ros")) / "config"
        checkpoint_path = model_dir / "model.pth"
        vocab_path = model_dir / "vocab.json"
        speakers_path = model_dir / "speakers_xtts.pth"
        config_path = model_dir / "config.json"
        self.embedding_path = model_dir / "speaker_embeddings/41"
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
        self.emotion_embeddings = self.get_emotion_embeddings()
        self.EMOTION_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
        # Set device
        if self.device == "cuda":
            self.tts.cuda()
        else:
            self.tts.cpu()

        # Validate speaker WAV path if provided
        if not self.speaker_wav or not (Path(self.speaker_wav).exists() and Path(self.speaker_wav).is_file()):
            self.speaker_wav = None

        # Validate speaker param for multi-speaker model
        #if not self.speaker or not self.tts.is_multi_speaker:
        #    self.speaker = None

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

        self.get_logger().debug(f"Stream: {self.stream} Speaker WAV: {self.speaker_wav}")

        self.gpt_cond_latent, self.speaker_embedding = self.tts.get_conditioning_latents(
                audio_path=[self.speaker_wav]
        )
        self.get_logger().info("got embeeddings")
        
        self.get_logger().info("TTS node started")

    def get_emotion_embeddings(self):
        emotion_embeddings = {}
        for emotion in os.listdir(self.embedding_path):
            emotion_embeddings[emotion] = {
                "speaker_embedding": torch.load(os.path.join(self.embedding_path, emotion, f"speaker_embedding_{emotion}.pth")),
                "gpt_cond_latent": torch.load(os.path.join(self.embedding_path, emotion, f"gpt_cond_latent_{emotion}.pth")),
            }
        print("Loaded with emotions", [em for em in emotion_embeddings.keys()])
        return emotion_embeddings

    def generate_speech(self, text, emotion="neutral", parse_emotions=False):
        self.get_logger().info("Text to say")
        self.get_logger().info(text)
        if emotion not in self.emotion_embeddings:
            raise ValueError(f"Emotion '{emotion}' not found in the embeddings.")

        if not parse_emotions:
            out = self.tts.inference(
                text,
                "es",
                self.emotion_embeddings[emotion]["gpt_cond_latent"],
                self.emotion_embeddings[emotion]["speaker_embedding"],
                temperature=0.9,
            )
            return out
        else:
            chunks = self.parse_emotions(text)
            self.get_logger().info(f"{chunks}")
            if len(chunks) == 0:
                out = self.tts.inference(
                    text,
                    "es",
                    self.emotion_embeddings[emotion]["gpt_cond_latent"],
                    self.emotion_embeddings[emotion]["speaker_embedding"],
                    temperature=0.9,
                )
                return out
            else:
                audio_arrays = []

                for chk in chunks:
                    emotion = chk[0]
                    txt = chk[1]
                    out = self.tts.inference(
                        txt,
                        "es",
                        self.emotion_embeddings[emotion]["gpt_cond_latent"],
                        self.emotion_embeddings[emotion]["speaker_embedding"],
                        temperature=0.9,
                    )
                    wav = out.get("wav")
                    if wav is not None and len(wav) > 0:
                        audio_arrays.append(wav)

                # Concatenate all audio chunks
                if len(audio_arrays) > 0:
                    full_audio = np.concatenate(audio_arrays)
                else:
                    full_audio = out

                # Return the same structure as a single inference output
                return {"wav": full_audio}

    def parse_emotions(self, text: str):
            return self.EMOTION_TAG_RE.findall(text)

    def wait_for_audio_playback(self, timeout_sec=10.0) -> bool:
        """Waits for the robot_speaks flag to become False with a timeout."""
        start_time = time.time()
        while self.robot_speaks:
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
        with self._goal_queue_lock:
            if self._current_goal is not None:
                self._goal_queue.append(goal_handle)
            else:
                self._current_goal = goal_handle
                self._current_goal.execute()

    def cancel_callback(self, goal_handle: ServerGoalHandle) -> None:
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle: ServerGoalHandle) -> TTS.Result:
        request: TTS.Goal = goal_handle.request
        text = request.text
        language = request.language

        if not text.strip():
            goal_handle.succeed()
            return TTS.Result()

        audio_format = pyaudio.paInt16
        channels = 1
        rate = 24000

        #if not self.tts.is_multilingual:
        #    language = None

        self.get_logger().info("Generating Audio")

        try:
            if not self.stream:
                out = self.generate_speech(text, emotion="neutral", parse_emotions=True)
                wav = torch.tensor(out["wav"]).unsqueeze(0)
                data = wav.numpy().flatten()
                data = np.clip(data, -1, 1)
                data = (data * 32767).astype(np.int16)


                # Create and publish full audio message at once
                audio_msg = data_to_msg(data, audio_format)
                if audio_msg is None:
                    self.get_logger().error(f"Format {audio_format} unknown")
                    goal_handle.abort()
                    self.run_next_goal()
                    return TTS.Result()

                msg = AudioStamped()
                msg.header.frame_id = self.frame_id
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.audio = audio_msg
                msg.audio.info.channels = channels
                msg.audio.info.chunk = get_msg_chunk(audio_msg)
                msg.audio.info.rate = rate

                with self._pub_lock:
                    self.__player_pub.publish(msg)
                    self.get_logger().info("published audio")
                    self.robot_speaks = True #or can be subscribed from topic

                    goal_handle.publish_feedback(TTS.Feedback(audio=msg))
                    if not self.wait_for_audio_playback(timeout_sec=50.0):
                        goal_handle.abort()
                        self.get_logger().error("Audio was not played (timeout).")
                    else:
                        goal_handle.succeed()

                    self.run_next_goal()
                    return TTS.Result()


            else:
                # streaming mode (keep as-is)
                chunks = self.tts.synthesizer.tts_model.inference_stream(
                    text,
                    language,
                    self.gpt_cond_latent,
                    self.speaker_embedding,
                )

        except Exception as e:
            self.get_logger().error(f"Exception '{e}' when processing text '{text}'")
            goal_handle.abort()
            self.run_next_goal()
            return TTS.Result()


        with self._pub_lock:
            self.run_next_goal()

            self.get_logger().info("Publishing Audio")
            frequency = rate / self.chunk

            if self._pub_rate is None:
                self._pub_rate = self.create_rate(frequency)

            for chunk_data in chunks:
                for j in range(0, len(chunk_data), self.chunk):

                    if self.stream:
                        data = chunk_data[j : j + self.chunk]
                        data = data.clone().detach().cpu().numpy()
                        data = data[None, : int(data.shape[0])]
                        data = np.clip(data, -1, 1)
                        data = (data * 32767).astype(np.int16)
                    else:
                        data = chunk_data

                    audio_msg = data_to_msg(data, audio_format)
                    if audio_msg is None:
                        self.get_logger().error(f"Format {audio_format} unknown")
                        goal_handle.abort()
                        return TTS.Result()

                    msg = AudioStamped()
                    msg.header.frame_id = self.frame_id
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.audio = audio_msg
                    msg.audio.info.channels = channels
                    msg.audio.info.chunk = get_msg_chunk(audio_msg)
                    msg.audio.info.rate = rate

                    feedback = TTS.Feedback()
                    feedback.audio = msg

                    if not goal_handle.is_active:
                        return TTS.Result()

                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        return TTS.Result()

                    self.get_logger().debug("Publishing Audio Chunk")
                    self.__player_pub.publish(msg)
                    goal_handle.publish_feedback(feedback)
                    self._pub_rate.sleep()

                    #if not self.stream:
                    #    break
                    self.get_logger().info("Waiting for robot to finish speaking...")
                    if not self.wait_for_audio_playback(timeout_sec=10.0):
                        goal_handle.abort()
                        self.get_logger().error("Audio was not played (timeout).")
                        return TTS.Result()

                    self.get_logger().info("Finished speaking.")
                    result = TTS.Result()
                    result.text = text
                    goal_handle.succeed()
                    return result


    def run_next_goal(self) -> bool:
        with self._goal_queue_lock:
            try:
                self._current_goal = self._goal_queue.popleft()
                t = threading.Thread(target=self._current_goal.execute)
                t.start()
                return True
            except IndexError:
                self._current_goal = None
                return False


def main():
    rclpy.init()
    node = TtsNode()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
