#!/bin/bash
# 一键生成 AI 多人轮唱音频（中间过程都在 tmp 目录处理，结束统一清理）
# python 环境 python 3.11
# 安装依赖：
# sudo apt install jq -y
# pip install demucs
# 使用 conda 环境隔离 seed-vc
# 截取 50-80 秒
# ffmpeg -i xxx.mp3 -ss 50 -t 30 -c copy song.mp3
set -e

BASEDIR="/usr/local/src"
SONG="${BASEDIR}/ai-singer/source/song.mp3"
OUTDIR="${BASEDIR}/ai-singer/results"
TMP_DIR="${OUTDIR}/tmp"

# 参考音频列表（每个演唱者对应一个参考音色）
REFS=(
  "${BASEDIR}/ai-singer/source/dsm.mp3"
  "${BASEDIR}/ai-singer/source/lyl.mp3"
  "${BASEDIR}/ai-singer/source/zsy.mp3"
  "${BASEDIR}/ai-singer/source/xmz.mp3"
)

mkdir -p "$OUTDIR" "$TMP_DIR"

echo "===> Step 1. 分离人声和伴奏"
# 转为 44.1kHz 双声道 WAV
ffmpeg -i "$SONG" -ar 44100 -ac 2 "${TMP_DIR}/song_44k.wav" -y
# 使用 demucs 分离人声和伴奏
demucs --two-stems=vocals -o "$TMP_DIR" "${TMP_DIR}/song_44k.wav"
VOCALS="${TMP_DIR}/htdemucs/song_44k/vocals.wav"
ACCOMP="${TMP_DIR}/htdemucs/song_44k/no_vocals.wav"

echo "===> Step 2. 离线人声分段 (diarize_vocals.py)"
DIARIZE_DIR="${TMP_DIR}/diarized"
mkdir -p "$DIARIZE_DIR"
python3 "${BASEDIR}/ai-singer/diarize_vocals.py" "$VOCALS" "$DIARIZE_DIR" --threshold 0.7
SEGMENTS_FILE="${DIARIZE_DIR}/segments.jsonl"
[[ ! -f "$SEGMENTS_FILE" ]] && { echo "❌ 分段文件不存在"; exit 1; }

# 暂停手动调整 SEGMENTS_FILE 调整完按y 继续
read -p "是否继续？(y/n)" -n 1 -r
echo
[[ ! $REPLY =~ ^[Yy]$ ]] && exit 0

echo "===> Step 3. SeedVC 音色转换 & 分段标准化"
SEG_AUDIO_LIST=()  # 保存标准化后的分段文件路径
START_LIST=()      # 保存每段起始时间（ms）
i=0

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    i=$((i+1))

    # 修复 JSON 奇怪开头
    # [[ "$line" == *'speaker'* ]] && [[ "$line" != \{* ]] && line="{\"${line}"
    # [[ "$line" == *'peaker'* ]] && [[ "$line" != \{* ]] && line="{\"s${line}"

    # 解析分段信息
    SPEAKER=$(echo "$line" | jq -r '.speaker' 2>/dev/null)
    SEG_FILE=$(echo "$line" | jq -r '.file' 2>/dev/null)
    START_SEC=$(echo "$line" | jq -r '.start // 0')
    START_MS=$(awk -v s="$START_SEC" 'BEGIN{printf "%d", s*1000}')

    [[ ! -f "$SEG_FILE" ]] && { echo "⚠️ 文件不存在: $SEG_FILE"; continue; }

    # 选择参考音色
    REF_INDEX=$(echo "$SPEAKER" | grep -Eo '[0-9]+' | head -n 1)
    REF="${REFS[$REF_INDEX]}"
    [[ ! -f "$REF" ]] && REF="${BASEDIR}/ai-singer/source/dsm.mp3"

    OUT_CONVERT="${TMP_DIR}/segment_${i}"
    mkdir -p "$OUT_CONVERT"

    echo "===> Step 3.${i} SeedVC ($SPEAKER)"
    cd "${BASEDIR}/seed-vc"
    conda run --no-capture-output -n seed-vc python inference.py \
        --source "$SEG_FILE" \
        --target "$REF" \
        --output "$OUT_CONVERT" \
        --diffusion-steps 60 \
        --length-adjust 1.0 \
        --inference-cfg-rate 0.9 \
        --f0-condition True \
        --auto-f0-adjust True \
        --semi-tone-shift 0 \
        --fp16 False || { echo "⚠️ SeedVC 执行失败"; continue; }

    # 找到最新生成的 wav
    FINAL_SEG=$(ls -t "${OUT_CONVERT}"/*.wav 2>/dev/null | head -n 1)
    [[ -z "$FINAL_SEG" ]] && { echo "❌ SeedVC 未生成 wav"; continue; }

    # 分段音量标准化
    NORM_SEG="${TMP_DIR}/segment_${i}_norm.wav"
    ffmpeg -nostdin -hide_banner -loglevel error -i "$FINAL_SEG" \
        -af "loudnorm=I=-16:TP=-1.5:LRA=11" \
        -ar 44100 -ac 1 -c:a pcm_s16le "$NORM_SEG" -y

    # 保存信息用于后续叠加
    START_LIST+=("$START_MS")
    SEG_AUDIO_LIST+=("$NORM_SEG")
done < "$SEGMENTS_FILE"

[[ ${#SEG_AUDIO_LIST[@]} -eq 0 ]] && { echo "❌ 没有可用片段"; exit 1; }

echo "===> Step 4. 按时间戳逐段叠加合并"
cp "${SEG_AUDIO_LIST[0]}" "${TMP_DIR}/current.wav"
for ((i=1; i<${#SEG_AUDIO_LIST[@]}; i++)); do
    DELAY_MS="${START_LIST[i]}"
    SEG_FILE="${SEG_AUDIO_LIST[i]}"
    ffmpeg -hide_banner -loglevel error -i "${TMP_DIR}/current.wav" -i "$SEG_FILE" \
        -filter_complex "[1:a]adelay=${DELAY_MS}|${DELAY_MS}[delayed]; \
                         [0:a][delayed]amix=inputs=2:duration=longest:weights=1 1[aout]" \
        -map "[aout]" -c:a pcm_s16le -ar 44100 "${TMP_DIR}/current_new.wav" -y
    mv "${TMP_DIR}/current_new.wav" "${TMP_DIR}/current.wav"
done

# 整体音量标准化
ffmpeg -hide_banner -loglevel error -i "${TMP_DIR}/current.wav" \
    -af "loudnorm=I=-16:TP=-1.5:LRA=11" \
    -ar 44100 -ac 2 "${TMP_DIR}/vocals_seq_norm.wav" -y

echo "===> Step 5. 混合伴奏并保持立体声"
# 提升伴奏和人声音量
ffmpeg -i "$ACCOMP" -af "volume=+3dB" "${TMP_DIR}/accomp_boosted.wav" -y
ffmpeg -i "${TMP_DIR}/vocals_seq_norm.wav" -af "volume=+6dB" "${TMP_DIR}/vocals_boosted.wav" -y

# 立体声混音
ffmpeg -i "${TMP_DIR}/vocals_boosted.wav" -i "${TMP_DIR}/accomp_boosted.wav" \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0[aout]" \
  -map "[aout]" -c:a libmp3lame -b:a 192k -ac 2 "${OUTDIR}/final_audio.mp3" -y

echo "===> Step 6. 清理临时文件"
rm -rf "$TMP_DIR"

echo "✅ 完成: ${OUTDIR}/final_audio.mp3"
