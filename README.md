# tts_ros

EMOROBCARE updates. 
This code includes a wrapper of ROS2 with COQUI-TTS, based on the existing solution by [tts_ros](https://github.com/mgonzs13/tts_ros/tree/main). It has then integrated the [robot_tts](https://github.com/EMOROBOCARE/robot_tts)'s example to use our own TTS models. 

In order to test it, first you need to download an example WAV file that will be used to clone the voice. For example, create it with  from [ttsmp3.com](https://ttsmp3.com/), convert it to WAV format with vith this [converter](https://www.viewpdf.com/result.html?e=0&c=mp3-to-wav#).

You also need to download the models in the tts_ros/config/ folder. 
Then execute the TTS ROS2 node, specifying the path of the WAV file. NOTE this is currently working on Arnau's computer docker "emorobcare": 

```shell
 ros2 run tts_ros tts_node --ros-args   -p stream:=false  -p speaker_wav:=/home/user/ws/src/audios/voice.wav
```

Note that usually we want to set stream:=false, as this will publish the whole TTS together on the ROS2 topic 'audio'. We can investigate how to publish it progressively. 

Then test by calling the action: 
 
 
```shell
 ros2 action send_goal /say audio_tts_msgs/action/TTS "{'text': 'Hola, soy un robot muy alegre que quiere jugar contigo', 'language': 'es', 'volume': 0.8, 'rate': 1.2}"
```

The output will be publishes on `audio` topic (to check naming). To play it directly, use the [audio_player](https://github.com/EMOROBOCARE/audio_player) package, see the repository on how to set it up. The action monitors the /robot_speaking topic (of type std_msgs/msg/Bool), and only returns success once it is set back to False (the robot has finished playing the audio). Note for these you need to set-up ROS2 multicasting between both computers with CYCLONE (see the audio_player repo). To test it without the audio player, you can manually set it to True with:

```shell
ros2 topic pub /robot_speaking std_msgs/msg/Bool "{data: true}" --once
```


## ROS interfaces

Publishers:

**/tts/audio** of type **audio_tts_msgs/msg/AudioStamped**: publishes the audio stream

Subscription:

**/robot_speaking** of type **std_msgs/msg/Bool**: sets to True when the audio starts playing, and back to False when it finishes playing. 

Actions:

**/say** of type audio_tts_msgs/action/TTS: gets as input the text, language, volume and rate to use, and returns success when audio has finished
playig. Note right now in our current models we are not managing volume and rate (WIP). 



----
This repositiory integrates the Python [TTS](https://pypi.org/project/TTS/) (Text-to-Speech) package into ROS 2 using [audio_common](https://github.com/mgonzs13/audio_common) [4.0.5](https://github.com/mgonzs13/audio_common/releases/tag/4.0.5).

<div align="center">

[![License: MIT](https://img.shields.io/badge/GitHub-MIT-informational)](https://opensource.org/license/mit) [![GitHub release](https://img.shields.io/github/release/mgonzs13/tts_ros.svg)](https://github.com/mgonzs13/tts_ros/releases) [![Code Size](https://img.shields.io/github/languages/code-size/mgonzs13/tts_ros.svg?branch=main)](https://github.com/mgonzs13/tts_ros?branch=main) [![Last Commit](https://img.shields.io/github/last-commit/mgonzs13/tts_ros.svg)](https://github.com/mgonzs13/tts_ros/commits/main) [![GitHub issues](https://img.shields.io/github/issues/mgonzs13/tts_ros)](https://github.com/mgonzs13/tts_ros/issues) [![GitHub pull requests](https://img.shields.io/github/issues-pr/mgonzs13/tts_ros)](https://github.com/mgonzs13/tts_ros/pulls) [![Contributors](https://img.shields.io/github/contributors/mgonzs13/tts_ros.svg)](https://github.com/mgonzs13/tts_ros/graphs/contributors) [![Python Formatter Check](https://github.com/mgonzs13/tts_ros/actions/workflows/python-formatter.yml/badge.svg?branch=main)](https://github.com/mgonzs13/tts_ros/actions/workflows/python-formatter.yml?branch=main)

| ROS 2 Distro |                         Branch                          |                                                                                                     Build status                                                                                                     |                                                               Docker Image                                                               | Documentation                                                                                                                                              |
| :----------: | :-----------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  **Humble**  | [`main`](https://github.com/mgonzs13/tts_ros/tree/main) |  [![Humble Build](https://github.com/mgonzs13/tts_ros/actions/workflows/humble-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/tts_ros/actions/workflows/humble-docker-build.yml?branch=main)   |  [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-humble-blue)](https://hub.docker.com/r/mgons/tts_ros/tags?name=humble)  | [![Doxygen Deployment](https://github.com/mgonzs13/tts_ros/actions/workflows/doxygen-deployment.yml/badge.svg)](https://mgonzs13.github.io/tts_ros/latest) |
|   **Iron**   | [`main`](https://github.com/mgonzs13/tts_ros/tree/main) |     [![Iron Build](https://github.com/mgonzs13/tts_ros/actions/workflows/iron-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/tts_ros/actions/workflows/iron-docker-build.yml?branch=main)      |    [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-iron-blue)](https://hub.docker.com/r/mgons/tts_ros/tags?name=iron)    | [![Doxygen Deployment](https://github.com/mgonzs13/tts_ros/actions/workflows/doxygen-deployment.yml/badge.svg)](https://mgonzs13.github.io/tts_ros/latest) |
|  **Jazzy**   | [`main`](https://github.com/mgonzs13/tts_ros/tree/main) |    [![Jazzy Build](https://github.com/mgonzs13/tts_ros/actions/workflows/jazzy-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/tts_ros/actions/workflows/jazzy-docker-build.yml?branch=main)    |   [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-jazzy-blue)](https://hub.docker.com/r/mgons/tts_ros/tags?name=jazzy)   | [![Doxygen Deployment](https://github.com/mgonzs13/tts_ros/actions/workflows/doxygen-deployment.yml/badge.svg)](https://mgonzs13.github.io/tts_ros/latest) |
| **Rolling**  | [`main`](https://github.com/mgonzs13/tts_ros/tree/main) | [![Rolling Build](https://github.com/mgonzs13/tts_ros/actions/workflows/rolling-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/tts_ros/actions/workflows/rolling-docker-build.yml?branch=main) | [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-rolling-blue)](https://hub.docker.com/r/mgons/tts_ros/tags?name=rolling) | [![Doxygen Deployment](https://github.com/mgonzs13/tts_ros/actions/workflows/doxygen-deployment.yml/badge.svg)](https://mgonzs13.github.io/tts_ros/latest) |

</div>

## Table of Contents

1. [Installation](#installation)
2. [Docker](#docker)
3. [Usage](#usage)

## Installation

```shell
cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/audio_common.git
git clone https://github.com/mgonzs13/tts_ros.git
pip3 install -r tts_ros/requirements.txt
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

## Docker

You can build the tts_ros docker:

```shell
docker build -t tts_ros .
```

Then, you can run the docker container:

```shell
docker run -it --rm --device /dev/snd tts_ros
```

## Usage

To use this tool you have to run the tts_node. It has the following parameters:

- `chunk`: Size of audio chunks to be sent to the audio player.
- `frame_id`: Frame of for the tts.
- `model`: The tts model. You can check the available models with `tts --list_models`.
- `model_path`: Path to a local model file.
- `config_path`: Path to a config file.
- `vocoder_path`: Path to a vocoder model file.
- `vocoder_config_path`: Path to a config file.
- `device`: The device to run the model same as in torch.
- `speaker_wav`: The wav file to perform voice cloning.
- `speaker`: Which speaker voice to use for multi-speaker models. Check with `tts --model_name <model> --list_language_idx`.
- `stream`: Whether to stream the audio data.

### Parameters Format

```shell
ros2 run tts_ros tts_node --ros-args -p chunk:=4096 -p frame_id:="your-frame" -p model:="your-model" -p device:="cpu/cuda" -p speaker_wav:="/path/to/wav/file" -p stream:=False
```

## Demo

```shell
ros2 launch tts_bringup tts.launch.py device:="cuda"
```

```shell
ros2 action send_goal /say audio_common_msgs/action/TTS "{'text': 'Hello World'}"
```

## Streaming Demo

```shell
ros2 launch tts_bringup tts.launch.py stream:=True model:="tts_models/multilingual/multi-dataset/xtts_v2" speaker_wav:="/home/miguel/Downloads/question_1.wav"  device:="cuda"
```

```shell
ros2 action send_goal /say audio_common_msgs/action/TTS "{'text': 'Hello World, How are you? Can you hear me? What is your favorite color? Do you know the Robot Operating System?'}"
```
