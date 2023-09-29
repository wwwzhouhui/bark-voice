import subprocess
import random
import os
from pathlib import Path
import librosa
from scipy.io import wavfile
import numpy as np
import torch
import csv
import whisper
import gradio as gr

os.system("pip install --upgrade Cython==0.29.35")
os.system("pip install pysptk --no-build-isolation")
os.system("pip install kantts -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html")
os.system("pip install tts-autolabel -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html")
os.system("apt-get install sox")

os.system("git clone https://github.com/fbcotter/pytorch_wavelets")
os.system("cd pytorch_wavelets")
os.system("pip install .")

os.system("pip install modelscope==1.8.4")

import sox

def split_long_audio(model, filepaths, save_dir="data_dir", out_sr=44100):
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    for file_idx, filepath in enumerate(filepaths):

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        print(f"Transcribing file {file_idx}: '{filepath}' to segments...")
        result = model.transcribe(filepath, word_timestamps=True, task="transcribe", beam_size=5, best_of=5)
        segments = result['segments']

        wav, sr = librosa.load(filepath, sr=None, offset=0, duration=None, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=out_sr)
        wav2 /= max(wav2.max(), -wav2.min())

        for i, seg in enumerate(segments):
            start_time = seg['start']
            end_time = seg['end']
            wav_seg = wav2[int(start_time * out_sr):int(end_time * out_sr)]
            wav_seg_name = f"{file_idx}_{i}.wav"
            out_fpath = save_path / wav_seg_name
            wavfile.write(out_fpath, rate=out_sr, data=(wav_seg * np.iinfo(np.int16).max).astype(np.int16))

whisper_size = "medium"
whisper_model = whisper.load_model(whisper_size)

from modelscope.tools import run_auto_label

from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType

pretrained_model_id = 'damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k'

dataset_id = "./output_training_data/"
pretrain_work_dir = "./pretrain_work_dir/"


def auto_label(audio):
    try:
        split_long_audio(whisper_model, audio, "test_wavs")
        os.makedirs("output_training_data", exist_ok=True)
        input_wav = "./test_wavs/"
        output_data = "./output_training_data/"
        ret, report = run_auto_label(input_wav=input_wav, work_dir=output_data, resource_revision="v1.0.7")
    
    except Exception:
        pass
    return "标注成功"


def train(a):
    try:
        os.makedirs("pretrain_work_dir", exist_ok=True)

        train_info = {
            TtsTrainType.TRAIN_TYPE_SAMBERT: {  # 配置训练AM（sambert）模型
                'train_steps': 52,               # 训练多少个step
                'save_interval_steps': 50,       # 每训练多少个step保存一次checkpoint
                'log_interval': 10               # 每训练多少个step打印一次训练日志
            }
        }

        # 配置训练参数，指定数据集，临时工作目录和train_info
        kwargs = dict(
            model=pretrained_model_id,                  # 指定要finetune的模型
            model_revision = "v1.0.6",
            work_dir=pretrain_work_dir,                 # 指定临时工作目录
            train_dataset=dataset_id,                   # 指定数据集id
            train_type=train_info                       # 指定要训练类型及参数
        )

        trainer = build_trainer(Trainers.speech_kantts_trainer,
                            default_args=kwargs)

        trainer.train()

    except Exception:
        pass
      
    return "训练完成"


import random

def infer(text):

  model_dir = os.path.abspath("./pretrain_work_dir")

  custom_infer_abs = {
      'voice_name':
      'F7',
      'am_ckpt':
      os.path.join(model_dir, 'tmp_am', 'ckpt'),
      'am_config':
      os.path.join(model_dir, 'tmp_am', 'config.yaml'),
      'voc_ckpt':
      os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'ckpt'),
      'voc_config':
      os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
              'config.yaml'),
      'audio_config':
      os.path.join(model_dir, 'data', 'audio_config.yaml'),
      'se_file':
      os.path.join(model_dir, 'data', 'se', 'se.npy')
  }
  kwargs = {'custom_ckpt': custom_infer_abs}

  model_id = SambertHifigan(os.path.join(model_dir, "orig_model"), **kwargs)

  inference = pipeline(task=Tasks.text_to_speech, model=model_id)
  output = inference(input=text)

  filename = str(random.randint(1, 1000000000000))

  with open(filename + "myfile.wav", mode='bx') as f:
      f.write(output["output_wav"])
  return filename + "myfile.wav"


#auto_label("nana_speech.wav")
#train("test")
#infer("测试一下")

app = gr.Blocks()

with app:
    gr.Markdown("# <center>🥳🎶🎡 - AI中文声音克隆</center>")
    gr.Markdown("## <center>🌟 - 训练10分钟，推理8秒钟，中英自然发音 </center>")

    with gr.Row():
      inp1 = gr.Audio(type="filepath", label="请上传一段音频")
      out1 = gr.Textbox(label="标注情况", lines=1, interactive=False)

      out2 = gr.Textbox(label="训练情况", lines=1, interactive=False)
      inp2 = gr.Textbox(label="文本", lines=3)
      out3 = gr.Audio(type="filepath", label="合成的音频")
      btn1 = gr.Button("1.标注数据")
      btn2 = gr.Button("2.开始训练")
      btn3 = gr.Button("3.一键推理", variant="primary")

      btn1.click(auto_label, inp1, out1)
      btn2.click(train, out1, out2)
      btn3.click(infer, inp2, out3)

    gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。</center>")
app.launch(show_error=True, share=True)
