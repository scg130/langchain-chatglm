#!/bin/bash
# 一键生成 AI 歌手唱歌视频
# 使用 conda 环境隔离：seed-vc + sadtalker
BASEDIR="/usr/local/src"

# ========== 参数 ==========
SONG="${BASEDIR}/ai-singer/song.mp3"            # 输入歌曲 (带伴奏)
REF="${BASEDIR}/ai-singer/ref.mp3"              # 参考人声音色
FACE="${BASEDIR}/ai-singer/face.jpeg"           # 人脸图片
OUTDIR="${BASEDIR}/ai-singer/results"           # 输出目录
CKPT="${BASEDIR}/seed-vc/checkpoints/singing_44k.pth"
CFG="${BASEDIR}/seed-vc/configs/singing_44k.yaml"

mkdir -p "$OUTDIR"

echo "===> Step 1. 分离人声伴奏 (spleeter)"
ffmpeg -i "$SONG" -ar 44100 -ac 2 "${OUTDIR}/song_44k.wav" -y
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
  --diffusion-steps 40 \
  --length-adjust 1.0 \
  --inference-cfg-rate 0.7 \
  --f0-condition True \
  --auto-f0-adjust False \
  --semi-tone-shift 0 \
  --fp16 True

echo "===> Step 3. 转换为 16k 单声道 (供 SadTalker)"
cd "${BASEDIR}/ai-singer"
ffmpeg -i "${OUTDIR}/converted.wav/*.wav" -ar 16000 -ac 1 "${OUTDIR}/converted_16k.wav" -y

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

VIDEO="${OUTDIR}/output/*.mp4"

echo "===> Step 5. 合并伴奏与视频"
cd "${BASEDIR}/ai-singer"
ffmpeg -i "$VIDEO" -i "$SONG" -c:v copy -c:a aac -shortest "${OUTDIR}/final_mv.mp4" -y

echo "===> Step 6. 清理临时文件"
rm -rf "${OUTDIR}/song_44k.wav"
rm -rf "${OUTDIR}/song_44k"
rm -rf "${OUTDIR}/converted.wav"
rm -rf "${OUTDIR}/converted_16k.wav"
rm -rf "${VIDEO}"
rm -fr "${OUTDIR}/output"

echo "✅ 完成: ${OUTDIR}/final_mv.mp4"
