#!/bin/bash
# 一键生成 AI 多人轮唱视频 (InfiniteTalk)
BASEDIR="/usr/local/src"
SONG="${BASEDIR}/ai-singer/song.mp3"
OUTDIR="${BASEDIR}/ai-singer/results"

REFS=(
  "${BASEDIR}/ai-singer/ref1.mp3"
  "${BASEDIR}/ai-singer/ref2.mp3"
)
FACES=(
  "${BASEDIR}/ai-singer/face1.jpeg"
  "${BASEDIR}/ai-singer/face2.jpeg"
)

mkdir -p "$OUTDIR"

echo "===> Step 1. 分离人声和伴奏"
ffmpeg -i "$SONG" -ar 44100 -ac 2 "${OUTDIR}/song_44k.wav" -y
spleeter separate -p spleeter:2stems -o "$OUTDIR" "${OUTDIR}/song_44k.wav"
VOCALS="${OUTDIR}/song_44k/vocals.wav"
ACCOMP="${OUTDIR}/song_44k/accompaniment.wav"

echo "===> Step 2. 自动分段识别不同人声段"
python "${BASEDIR}/diarize_vocals.py" "$VOCALS" "${OUTDIR}/diarized"

# Step 3. SeedVC 转换 + 生成 JSON 输入
JSON_FILE="${OUTDIR}/infinitetalk_input.json"
echo "[" > $JSON_FILE

SEGMENTS_FILE="${OUTDIR}/segments.txt"
python - <<EOF
import json
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline("${VOCALS}")
with open("${SEGMENTS_FILE}", "w") as f:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        f.write(json.dumps({"speaker": speaker, "start": turn.start, "end": turn.end})+"\n")
EOF

i=0
while read line; do
    i=$((i+1))
    SPEAKER=$(echo $line | python -c "import sys,json; print(json.load(sys.stdin)['speaker'])")
    START=$(echo $line | python -c "import sys,json; print(json.load(sys.stdin)['start'])")
    END=$(echo $line | python -c "import sys,json; print(json.load(sys.stdin)['end'])")
    SEG_FILE="${OUTDIR}/segment_${i}.wav"

    # 切出该段
    ffmpeg -i "$VOCALS" -ss $START -to $END "$SEG_FILE" -y

    # SeedVC 转换
    REF="${REFS[$(( (i-1) % ${#REFS[@]} ))]}"
    OUT_CONVERT="${OUTDIR}/segment_${i}_converted.wav"
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

    # 转换为 16k 单声道 (InfiniteTalk 要求)
    FINAL_SEG="${OUTDIR}/segment_${i}_16k.wav"
    ffmpeg -i "$OUT_CONVERT" -ar 16000 -ac 1 "$FINAL_SEG" -y

    # 生成 JSON 条目
    FACE="${FACES[$(( (i-1) % ${#FACES[@]} ))]}"
    if [ $i -eq $(wc -l < $SEGMENTS_FILE) ]; then
        echo "  {\"audio\": \"$FINAL_SEG\", \"image\": \"$FACE\"}" >> $JSON_FILE
    else
        echo "  {\"audio\": \"$FINAL_SEG\", \"image\": \"$FACE\"}," >> $JSON_FILE
    fi

done < "$SEGMENTS_FILE"
echo "]" >> $JSON_FILE

echo "===> Step 4. 调用 InfiniteTalk 生成多人视频"
cd "${BASEDIR}/InfiniteTalk"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 conda run -n infinitetalk python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json "$JSON_FILE" \
    --size infinitetalk-720 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file "${OUTDIR}/multi_infinitetalk_res_720p"

echo "✅ 完成: ${OUTDIR}/multi_infinitetalk_res_720p.mp4"
