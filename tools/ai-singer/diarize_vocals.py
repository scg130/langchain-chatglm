#!/usr/bin/env python3
"""
pip install resemblyzer librosa soundfile numpy scikit-learn

用法:
python diarize_vocals.py vocals.wav ./diarized [--force-single]

功能:
- 支持单人和多人歌唱音轨
- 静音切分 + 可选聚类
- 输出每段信息到 ./diarized/segments.jsonl
"""
import sys, os, json, numpy as np, librosa, soundfile as sf
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# ========== 参数 ==========
vocals_file = sys.argv[1]
outdir = sys.argv[2]
force_single = "--force-single" in sys.argv

os.makedirs(outdir, exist_ok=True)

# 1️⃣ 读取音频
y, sr = librosa.load(vocals_file, sr=16000)
encoder = VoiceEncoder()

# 2️⃣ 静音切分（避免切太碎，调大 top_db）
intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
segments = [(start, end) for start, end in intervals if (end-start)/sr >= 0.5]  # 丢掉<0.5s片段

if not segments:
    print("❌ 未检测到有效人声片段")
    sys.exit(1)

# 3️⃣ 提取嵌入
embeddings = []
for start, end in segments:
    seg = y[start:end]
    if np.max(np.abs(seg)) < 1e-4:
        embeddings.append(np.zeros(256))  # 空段填零向量
    else:
        embeddings.append(encoder.embed_utterance(seg))
X = np.vstack(embeddings)

# 4️⃣ 判断单人或多人
labels = None
num_speakers = 1
avg_sim = 1.0

if not force_single:
    sim_matrix = cosine_similarity(X)
    avg_sim = (np.sum(sim_matrix) - len(sim_matrix)) / (len(sim_matrix)**2 - len(sim_matrix))
    if avg_sim > 0.85 or len(segments) == 1:
        labels = [0] * len(segments)
        num_speakers = 1
        print(f"⚠️ 相似度 {avg_sim:.3f}，识别为单人音色，所有段统一为 speaker0")
    else:
        best_score = -1
        best_labels = None
        max_speakers = min(len(segments), 3)
        for n in range(2, max_speakers+1):
            clustering = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
            lbls = clustering.fit_predict(X)
            try:
                score = silhouette_score(X, lbls, metric="cosine")
            except:
                continue
            if score > best_score:
                best_score = score
                best_labels = lbls
                num_speakers = n
        labels = best_labels
        print(f"✅ 相似度 {avg_sim:.3f}，识别为 {num_speakers} 个说话人 (silhouette={best_score:.3f})")
else:
    labels = [0] * len(segments)
    num_speakers = 1
    print("⚡ 强制单人模式 (--force-single)，所有段统一为 speaker0")

# 5️⃣ 输出 JSONL + 切片音频（确保无 BOM，UTF-8 编码）
segments_file = os.path.join(outdir, "segments.jsonl")
# 在 diarize_vocals.py 中修改写入部分
with open(segments_file, "w", encoding="utf-8") as f:
    for i, ((start, end), label) in enumerate(zip(segments, labels), 1):
        start_sec = start / sr
        end_sec = end / sr
        seg_file = os.path.join(outdir, f"segment_{i}.wav")
        sf.write(seg_file, y[start:end], sr)
        info = {
            "speaker": f"speaker{label}",
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "file": seg_file
        }
        # 确保每次写入后立即刷新，避免缓冲区问题
        f.write(json.dumps(info, ensure_ascii=False) + "\n")
        f.flush()  # 立即写入磁盘

print(f"✅ 分段信息已保存: {segments_file}, 总段数: {len(segments)}, 识别人数: {num_speakers}")