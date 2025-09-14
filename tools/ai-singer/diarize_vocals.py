#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentence-level diarization with speaker differentiation (带音色区分)

依赖安装：
    pip install "typing-extensions<4.6.0"
    pip install openai-whisper soundfile resemblyzer numpy

说明：
1. 使用 Whisper large-v2 模型进行语音转录
2. 使用 Resemblyzer 提取音色 embedding 并判断不同歌手
3. 每句话生成单独音频段，并在 JSONL 中记录 speaker/text/start/end/file
4. 可以通过 --threshold 参数调整音色相似度判断灵敏度

使用命令示例：
    python diarize_vocals.py vocals.wav ./diarized
    python diarize_vocals.py vocals.wav ./diarized --threshold 0.7

    python3 /usr/local/src/ai-singer/diarize_vocals.py /usr/local/src/ai-singer/results/song_44k/vocals.wav /usr/local/src/ai-singer/results/diarized --threshold 0.7
输出：
    ./diarized/seg_000.wav, seg_001.wav ...
    ./diarized/segments.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import whisper
from numpy.linalg import norm
from resemblyzer import VoiceEncoder, preprocess_wav


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def main():
    parser = argparse.ArgumentParser(
        description="Sentence-level diarization with speaker differentiation")
    parser.add_argument("vocals_file", help="输入人声音频文件")
    parser.add_argument("out_dir", help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="音色相似度阈值，越大越严格（默认0.75）")
    args = parser.parse_args()

    vocals_file = args.vocals_file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    SIM_THRESHOLD = args.threshold

    print(f"加载 Whisper large-v2 模型...")
    model = whisper.load_model("large-v2")
    result = model.transcribe(
        vocals_file, language="zh", word_timestamps=False)

    data, sr = sf.read(vocals_file)
    encoder = VoiceEncoder()

    speakers = []  # [(speaker_id, embedding)]
    segments = []
    spk_count = 0

    for i, seg in enumerate(result["segments"], 1):
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        seg_file = out_dir / f"seg_{i:03d}.wav"
        sf.write(seg_file, data[start_sample:end_sample], sr)

        wav = preprocess_wav(seg_file)
        emb = encoder.embed_utterance(wav)

        # 判断属于哪个 speaker
        assigned = False
        for spk_id, spk_emb in speakers:
            if cosine_similarity(emb, spk_emb) >= SIM_THRESHOLD:
                speaker = spk_id
                # 更新平均 embedding
                new_emb = (spk_emb + emb) / 2
                speakers = [(sid, new_emb if sid == spk_id else e)
                            for sid, e in speakers]
                assigned = True
                break

        if not assigned:
            speaker = f"spk{spk_count}"
            speakers.append((speaker, emb))
            spk_count += 1

        segments.append({
            "speaker": speaker,
            "file": str(seg_file.resolve()),
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg.get("text", "")
        })

    # 输出 JSONL
    segments_file = out_dir / "segments.jsonl"
    with open(segments_file, "w", encoding="utf-8") as f:
        for seg in segments:
            json.dump(seg, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ 完成: 共 {len(segments)} 段, 检测到 {len(speakers)} 个不同音色")
    print(f"使用相似度阈值: {SIM_THRESHOLD}")


if __name__ == "__main__":
    main()
