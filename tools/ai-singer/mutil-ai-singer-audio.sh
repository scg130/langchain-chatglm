#!/bin/bash
# 一键生成 AI 多人轮唱音频（按时间顺序拼接 + 音量均衡）
# 使用 conda 环境隔离：seed-vc
# 支持修复偶数行 JSONL 问题

set -e

BASEDIR="/usr/local/src"
SONG="${BASEDIR}/ai-singer/song.mp3"
OUTDIR="${BASEDIR}/ai-singer/results"

REFS=(
  "${BASEDIR}/ai-singer/ref1.mp3"
#   "${BASEDIR}/ai-singer/ref2.mp3"
)

mkdir -p "$OUTDIR"

echo "===> Step 1. 分离人声和伴奏"
ffmpeg -i "$SONG" -ar 44100 -ac 2 "${OUTDIR}/song_44k.wav" -y
spleeter separate -p spleeter:2stems -o "$OUTDIR" "${OUTDIR}/song_44k.wav"
VOCALS="${OUTDIR}/song_44k/vocals.wav"
ACCOMP="${OUTDIR}/song_44k/accompaniment.wav"

echo "===> Step 2. 调用离线 diarize_vocals.py"
DIARIZE_DIR="${OUTDIR}/diarized"
mkdir -p "$DIARIZE_DIR"
python3 "${BASEDIR}/ai-singer/diarize_vocals.py" "$VOCALS" "$DIARIZE_DIR" --force-single

SEGMENTS_FILE="${DIARIZE_DIR}/segments.jsonl"
if [[ ! -f "$SEGMENTS_FILE" ]]; then
    echo "❌ Error: 分段文件不存在，退出"
    exit 1
fi

echo "===> Step 3. SeedVC 音色转换 + 音量均衡"
SEG_AUDIO_LIST=()
i=0

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    i=$((i+1))
    # 修复偶数行 前面缺少{"s
    if [[ $line == peaker\"* ]]; then
        line="{\"s$line"
    fi


    SPEAKER=$(echo "$line" | jq -r '.speaker' 2>/dev/null) || { echo "❌ 无法解析 JSON: $line"; continue; }
    SEG_FILE=$(echo "$line" | jq -r '.file' 2>/dev/null) || { echo "❌ 无法提取 file: $line"; continue; }
    [[ ! -f "$SEG_FILE" ]] && { echo "❌ 文件不存在: $SEG_FILE"; continue; }

    line=""

    echo "Processing segment $i: Speaker=$SPEAKER, File=$SEG_FILE"

    REF="${REFS[$(( (i-1) % ${#REFS[@]} ))]}"
    [[ ! -f "$REF" ]] && { echo "❌ 参考音频不存在: $REF"; exit 1; }

    OUT_CONVERT="${OUTDIR}/segment_${i}"
    mkdir -p "$OUT_CONVERT"

    echo "===> Step 3.${i} SeedVC 音色转换 ($SPEAKER)"
    cd "${BASEDIR}/seed-vc"
    conda run -n seed-vc python inference.py \
        --source "$SEG_FILE" \
        --target "$REF" \
        --output "$OUT_CONVERT" \
        --diffusion-steps 60 \
        --length-adjust 1.0 \
        --inference-cfg-rate 0.9 \
        --f0-condition True \
        --auto-f0-adjust True \
        --semi-tone-shift 0 \
        --fp16 False

    FINAL_SEG=$(ls -t "${OUT_CONVERT}"/*.wav 2>/dev/null | head -n 1)
    [[ -z "$FINAL_SEG" ]] && { echo "❌ SeedVC 未生成 wav"; exit 1; }

    NORM_SEG="${OUTDIR}/segment_${i}_final.wav"
    ffmpeg -i "$FINAL_SEG" -ar 44100 -ac 1 -c:a pcm_s16le "$NORM_SEG" -y

    AVG_DB=$(ffmpeg -i "$NORM_SEG" -af volumedetect -f null /dev/null 2>&1 \
             | grep "mean_volume" | grep -Eo "[-0-9.]+ dB" | head -1 | sed 's/ dB//')
    if [[ -n "$AVG_DB" ]]; then
        TARGET_DB=-20
        GAIN=$(echo "$TARGET_DB - $AVG_DB" | bc)
        NORM_SEG_GAIN="${OUTDIR}/segment_${i}_norm.wav"
        ffmpeg -i "$NORM_SEG" -af "volume=${GAIN}dB" -ar 44100 -ac 1 -c:a pcm_s16le "$NORM_SEG_GAIN" -y
        SEG_AUDIO_LIST+=("$NORM_SEG_GAIN")
    else
        echo "⚠️ 警告: 未检测到音量信息，直接使用 $NORM_SEG"
        SEG_AUDIO_LIST+=("$NORM_SEG")
    fi
done < "$SEGMENTS_FILE"

[[ ${#SEG_AUDIO_LIST[@]} -eq 0 ]] && { echo "❌ 没有处理任何片段"; exit 1; }

echo "===> Step 4. 拼接段落"
TXT_LIST="${OUTDIR}/concat_list.txt"
> "$TXT_LIST"
for f in "${SEG_AUDIO_LIST[@]}"; do
    echo "file '$(realpath "$f")'" >> "$TXT_LIST"
done
ffmpeg -f concat -safe 0 -i "$TXT_LIST" -c copy "${OUTDIR}/vocals_seq.wav" -y

echo "===> Step 5. 混合伴奏"
ffmpeg -i "${OUTDIR}/vocals_seq.wav" -i "$ACCOMP" \
    -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=2[aout]" \
    -map "[aout]" -c:a aac -b:a 192k "${OUTDIR}/final_audio.mp3" -y

echo "===> Step 6. 清理临时文件"
rm -f "${OUTDIR}/song_44k.wav" "$TXT_LIST"
rm -rf "${OUTDIR}/segment_"* "${OUTDIR}/diarized"

echo "✅ 完成: ${OUTDIR}/final_audio.mp3"
