import os
import torch
import numpy as np
from scipy.io.wavfile import write
import pandas as pd
from TTS.api import TTS

# 检查设备是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化 TTS 模型
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Excel 文件路径
excel_file = "danciall.xlsx"  # 替换为你的 Excel 文件路径

# 采样率和空白音频参数
rate = 24000
silence_duration = 1.5  # 1.5 秒的空白
silence_audio = np.zeros(int(rate * silence_duration), dtype=np.int16)  # 空白音频

# 定义交替使用的两个音频克隆文件
speaker_wavs = [
    "my/cloning/sample-answer-Eileen.wav",
    "my/cloning/活动-Sample answer copy.wav"
]

# 读取 Excel 文件
xls = pd.ExcelFile(excel_file)

# 遍历每个 sheet，并交替使用人声文件
for idx, sheet_name in enumerate(xls.sheet_names):
    # 选择当前的音频克隆文件
    speaker_wav = speaker_wavs[idx % 2]  # 使用交替的逻辑选择人声文件

    # 创建对应的文件夹
    output_dir = os.path.join("output_audio", sheet_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取当前 sheet
    df = pd.read_excel(xls, sheet_name=sheet_name,header=None, dtype=str)
    words = df.iloc[:, 0].dropna().tolist()  # 获取 A 列的所有单词并去除空值

    combined_audio = []  # 存储合并后的音频

    # 生成每个单词的音频
    for word in words:
        # 设置音频文件路径
        output_file = os.path.join(output_dir, f"{word}.wav")

        # 如果音频文件已经存在，则跳过
        if os.path.exists(output_file):
            print(f"Audio file for '{word}' already exists. Skipping...")
            continue
        if word and isinstance(word, str):
            tts.tts_to_file(
                text=word,
                speaker_wav=speaker_wav,
                language="en",
                file_path=output_file
            )
        else:
            print
            print(f"Skipping empty or invalid word: {word}")
        # 合成单词的音频并保存
        # tts.tts_to_file(
        #     text=word, 
        #     speaker_wav=speaker_wav,  # 使用当前的音频克隆文件
        #     language="en", 
        #     file_path=output_file
        # )
        print(f"Audio file saved as {output_file}")

        # 加载音频数据并添加到合并音频列表中
        wav_data = np.memmap(output_file, dtype='h', mode='r').copy()
        combined_audio.extend(wav_data)
        combined_audio.extend(silence_audio)  # 插入空白音频

    # 保存合并后的音频文件
    combined_output_file = os.path.join(output_dir, f"{sheet_name}_combined.wav")
    combined_audio = np.array(combined_audio, dtype=np.int16)
    write(combined_output_file, rate, combined_audio)
    print(f"Combined audio file saved as {combined_output_file}")
