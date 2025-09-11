#!/bin/bash
# 一键生成 AI 歌手唱歌视频
# 使用 conda 环境隔离：seed-vc + sadtalker

# ========== 参数 ==========
SONG="song.mp3"            # 输入歌曲 (带伴奏)
REF="ref.wav"              # 参考人声音色
FACE="face.png"            # 人脸图片
OUTDIR="results"           # 输出目录
CKPT="/usr/local/src/seed-vc/checkpoints/singing_44k.pth"
CFG="/usr/local/src/seed-vc/configs/singing_44k.yaml"

mkdir -p $OUTDIR

echo "===> Step 1. 分离人声伴奏 (spleeter)"
ffmpeg -i $SONG -ar 44100 -ac 2 ${OUTDIR}/song_44k.wav -y
spleeter separate -p spleeter:2stems -o $OUTDIR ${OUTDIR}/song_44k.wav
VOCALS=${OUTDIR}/song_44k/vocals.wav
ACCOMP=${OUTDIR}/song_44k/accompaniment.wav

echo "===> Step 2. SeedVC 音色转换"
conda run -n seed-vc python /usr/local/src/seed-vc/inference.py \
  --source $VOCALS \
  --target $REF \
  --output ${OUTDIR}/converted.wav \
  --checkpoint $CKPT \
  --config $CFG \
  --diffusion-steps 40 \
  --length-adjust 1.0 \
  --inference-cfg-rate 0.7 \
  --f0-condition True \
  --auto-f0-adjust False \
  --semi-tone-shift 0 \
  --fp16 True

echo "===> Step 3. 转换为 16k 单声道 (供 SadTalker)"
ffmpeg -i ${OUTDIR}/converted.wav -ar 16000 -ac 1 ${OUTDIR}/converted_16k.wav -y

echo "===> Step 4. SadTalker 生成视频"
conda run -n sadtalker python /usr/local/src/SadTalker/inference.py \
  --driven_audio ${OUTDIR}/converted_16k.wav \
  --source_image $FACE \
  --result_dir $OUTDIR \
  --preprocess full \
  --still \
  --enhancer gfpgan \
  --fps 25

VIDEO=${OUTDIR}/result.mp4

echo "===> Step 5. 合并伴奏与视频"
ffmpeg -i $VIDEO -i $SONG -c:v copy -c:a aac -shortest ${OUTDIR}/final_mv.mp4 -y

echo "===> Step 6. 清理临时文件"
rm -rf ${OUTDIR}/song_44k.wav
rm -rf ${OUTDIR}/song_44k
rm -rf ${OUTDIR}/converted.wav
rm -rf ${OUTDIR}/converted_16k.wav
rm -rf ${VIDEO}

echo "✅ 完成: ${OUTDIR}/final_mv.mp4"
