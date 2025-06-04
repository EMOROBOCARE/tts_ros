import os
import re

import torch
import torchaudio
from collections import deque

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class EmoRobCareTTS:
    EMOTION_TAGS = ["fear", "happiness", "neutral", "surprise"]
    def __init__(self, root_path, voice_actor=41):
        self.root_path = root_path
        self.model_path = os.path.join(self.root_path, "model")
        self.config = XttsConfig()
        self.config.load_json(os.path.join(self.root_path, "config.json"))
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config,
                                   checkpoint_path=os.path.join(self.model_path, "model.pth"),
                                   vocab_path=os.path.join(self.model_path, "vocab.json"),
                                   speaker_file_path=os.path.join(self.model_path, "speakers_xtts.pth"),
                                   use_deepspeed=False
        )


        self.voice_actor = voice_actor
        self.emotion_embeddings_path = os.path.join(self.root_path, f"speaker_embeddings/{voice_actor}")
        self.emotion_embeddings = self.get_emotion_embeddings()

        self.EMOTION_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)

        self.buffer = deque([], maxlen=20)  # Buffer to store recent text chunks
        self.out_queue = deque([], maxlen=20)  # Output queue for generated audio

        if torch.cuda.is_available():
            print("Running on GPU")
            self.model.cuda()

    def get_emotion_embeddings(self):
        emotion_embeddings = {}
        for emotion in os.listdir(self.emotion_embeddings_path):
            print(os.path.join(self.emotion_embeddings_path, emotion, f"speaker_embedding_{emotion}.pth"))
            emotion_embeddings[emotion] = {
                "speaker_embedding": torch.load(os.path.join(self.emotion_embeddings_path, emotion, f"speaker_embedding_{emotion}.pth"), map_location=torch.device('cpu')),
                "gpt_cond_latent": torch.load(os.path.join(self.emotion_embeddings_path, emotion, f"gpt_cond_latent_{emotion}.pth"), map_location=torch.device('cpu')),
            }
        print("Loaded with emotions", [em for em in emotion_embeddings.keys()])
        return emotion_embeddings

    def generate_speech(self, text, emotion="neutral", save_path=None, parse_emotions=False):
        if emotion not in self.emotion_embeddings:
            raise ValueError(f"Emotion '{emotion}' not found in the embeddings.")
        if not parse_emotions:
            out = self.model.inference(
                text,
                "es",
                self.emotion_embeddings[emotion]["gpt_cond_latent"],
                self.emotion_embeddings[emotion]["speaker_embedding"],
                temperature=0.9,  # Add custom parameters here
            )
            wav = torch.tensor(out["wav"]).unsqueeze(0)
            if save_path is None:
                torchaudio.save(save_path, wav, 24000)
            return wav
        else:
            chunks = self.parse_emotions(text)
            audios = []
            for i, chk in enumerate(chunks):
                emotion = chk[0]
                txt = chk[1]
                audios.append(self.model.inference(
                    txt,
                    "es",
                    self.emotion_embeddings[emotion]["gpt_cond_latent"],
                    self.emotion_embeddings[emotion]["speaker_embedding"],
                    temperature=0.9,  # Add custom parameters here
                ))
                if save_path is not None:
                    torchaudio.save(f"{save_path}_{i}.wav", torch.tensor(audios[-1]["wav"]).unsqueeze(0), 24000) #its only saving last one

            return audios

    def process_text(self, text: str):
        for chk in self.parse_emotions(text):
            self.buffer.append(chk)

    def generate_next(self):
        while True:
            emotion, text = self.buffer.popleft() # TODO manage empty buffer, and unexpected emotions
            out = self.model.inference(
                text,
                "es",
                self.emotion_embeddings[emotion]["gpt_cond_latent"],
                self.emotion_embeddings[emotion]["speaker_embedding"],
                temperature=0.9,  # Add custom parameters here
            )
            yield out["wav"]


    def parse_emotions(self, text: str):
        return self.EMOTION_TAG_RE.findall(text)


if __name__ == "__main__":
    tts = EmoRobCareTTS("config.json")
    tts.generate_speech("Hola! vamos a jugar a un juego!", emotion="neutral", save_path="neutral.wav")
    tts.generate_speech("Hola! vamos a jugar a un juego!", emotion="happiness", save_path="happiness.wav")
    tts.generate_speech("Hola! vamos a jugar a un juego!", emotion="fear", save_path="fear.wav")
    tts.generate_speech("Hola! vamos a jugar a un juego!", emotion="surprise", save_path="surprise.wav")
    tts.generate_speech("<happiness>Qué ruido hace la vaca...</happiness><surprise>¡Muy bien!</surprise>", parse_emotions=True, save_path="vaca.wav")


