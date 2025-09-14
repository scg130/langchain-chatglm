#!/bin/bash
# 一键生成 AI 歌手唱歌视频
# 使用 conda 环境隔离：seed-vc + sadtalker
BASEDIR="/usr/local/src"

# ========== 参数 ==========
SONG="${BASEDIR}/ai-singer/source/song.mp3"            # 输入歌曲 (带伴奏)
REF="${BASEDIR}/ai-singer/source/ref.mp3"              # 参考人声音色
FACE="${BASEDIR}/ai-singer/source/face.jpeg"           # 人脸图片
OUTDIR="${BASEDIR}/ai-singer/results"           # 输出目录

mkdir -p "$OUTDIR"

echo "===> Step 1. 分离人声伴奏 (spleeter)"
ffmpeg -i "$SONG" -ar 44100 -ac 2 "${OUTDIR}/song_44k.wav" -y
# python 3.11
# pip install spleeter   "click<8.2"    "typer<0.10"  tensorflow==2.12.1
spleeter separate -p spleeter:2stems -o "$OUTDIR" "${OUTDIR}/song_44k.wav"
VOCALS="${OUTDIR}/song_44k/vocals.wav"
ACCOMP="${OUTDIR}/song_44k/accompaniment.wav"

echo "===> Step 2. SeedVC 音色转换"
cd "${BASEDIR}/seed-vc"
conda run -n seed-vc python "${BASEDIR}/seed-vc/inference.py" \
  --source "$VOCALS" \
  --target "$REF" \
  --output "${OUTDIR}/converted.wav" \
  --diffusion-steps 60 \
  --length-adjust 1.0 \
  --inference-cfg-rate 0.9 \
  --f0-condition True \
  --auto-f0-adjust True \
  --semi-tone-shift 0 \
  --fp16 False

echo "===> Step 3. 转换为 16k 单声道 (供 SadTalker)"
cd "${BASEDIR}/ai-singer"
ffmpeg -i ${OUTDIR}/converted.wav/*.wav -ar 16000 -ac 1 "${OUTDIR}/converted_16k.wav" -y

# cp "${BASEDIR}/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors" "${BASEDIR}/SadTalker/checkpoints/epoch_20.pth"

echo "===> Step 4. SadTalker 生成视频"
cd "${BASEDIR}/SadTalker"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 conda run -n sadtalker python "${BASEDIR}/SadTalker/inference.py" \
  --driven_audio "${OUTDIR}/converted_16k.wav" \
  --source_image "$FACE" \
  --result_dir "${OUTDIR}/output" \
  --still \
  --expression_scale 0.8 \
  --preprocess full \
  --enhancer gfpgan 

VIDEO=${OUTDIR}/output/*.mp4

echo "===> Step 5. 合并伴奏与视频"
cd "${BASEDIR}/ai-singer"
# 合并 只保留accompaniment.wav声音
# ffmpeg -i $VIDEO -i "${OUTDIR}/song_44k/accompaniment.wav" -c:v copy -c:a aac -shortest "${OUTDIR}/final_mv.mp4" -y
# 合并音频 同时保留视频和伴奏的声音
ffmpeg -i $VIDEO -i "${OUTDIR}/song_44k/accompaniment.wav" -filter_complex "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=2[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -shortest "${OUTDIR}/final_mv.mp4" -y

echo "===> Step 6. 清理临时文件"
rm -rf "${OUTDIR}/song_44k.wav"
rm -rf "${OUTDIR}/song_44k"
rm -rf "${OUTDIR}/converted.wav"
rm -rf "${OUTDIR}/converted_16k.wav"
rm -rf "${VIDEO}"
rm -fr "${OUTDIR}/output"

echo "✅ 完成: ${OUTDIR}/final_mv.mp4"
