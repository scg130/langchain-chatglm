#!/usr/bin/env bash
set -euo pipefail

# ===============================================================
# Wan2.2 14B 部署脚本（Linux, conda + CUDA 12.4 已安装）
# 说明：
# - 会创建 conda 环境 wan2_14b（python3.10）
# - 安装 pytorch (pytorch-cuda=12.4) + comfyui 依赖 + edge-tts + ffmpeg
# - 需要用户手动下载大模型并放到 models/ 目录（脚本会创建目录并给出文件名提示）
# ===============================================================

ENV_NAME="wan2_14b"
PYTHON_VERSION="3.10"
CONDA_CMD="$(command -v conda || true)"

if [ -z "$CONDA_CMD" ]; then
  echo "错误：未找到 conda。请先安装 Anaconda / Miniconda / Mambaforge 后重试。"
  exit 1
fi

echo "检测 NVIDIA 驱动与 CUDA（nvidia-smi）..."
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi || true
else
  echo "警告：未检测到 nvidia-smi，确认 GPU 驱动已正确安装。"
fi

echo "创建 conda 环境：$ENV_NAME (Python $PYTHON_VERSION)"
conda create -y -n "$ENV_NAME" python=$PYTHON_VERSION

echo "激活环境并升级 pip"
# 注意：在脚本中通过 conda activate 可能不会修改当前 shell，使用 conda run 以保证可用
CONDA_RUN="conda run -n $ENV_NAME --no-capture-output"
$CONDA_RUN python -m pip install --upgrade pip setuptools wheel

echo "安装 PyTorch + CUDA 12.4（conda）"
# 这是常见的 conda 命令来安装 pytorch + pytorch-cuda 指定版本
$CONDA_RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

echo "安装系统依赖（ffmpeg、git）（需要 sudo 权限）..."
if command -v apt-get &>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg git build-essential
elif command -v yum &>/dev/null; then
  sudo yum install -y epel-release
  sudo yum install -y ffmpeg git gcc gcc-c++
else
  echo "未识别包管理器，请手动安装 ffmpeg、git 等系统依赖。"
fi

echo "创建项目目录 structure"
PROJECT_DIR="$HOME/wan2_14b_project"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 克隆 ComfyUI（社区版本可能在更新，使用官方/常用仓库）
if [ ! -d "ComfyUI" ]; then
  echo "正在克隆 ComfyUI..."
  git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI
else
  echo "已存在 ComfyUI 目录，跳过克隆。"
fi

echo "安装 ComfyUI 依赖"
# 安装 ComfyUI 的 python 依赖（使用 pip）
$CONDA_RUN python -m pip install -r ComfyUI/requirements.txt || \
  { echo "注意：requirements 安装失败，继续尝试安装通用依赖"; \
    $CONDA_RUN python -m pip install numpy pillow safetensors transformers accelerate diffusers==0.20.2 };

# 一些可选但常用的库
$CONDA_RUN python -m pip install einops ftfy regex composer==0.1.0 safetensors einops sentencepiece

echo "安装 ComfyUI 额外/社区节点 (可选：TTS 插件)"
# 常见 ComfyUI-TTS 插件（如 edge-tts 接入）
mkdir -p ComfyUI/custom_nodes
if [ ! -d "ComfyUI/custom_nodes/ComfyUI-TTS" ]; then
  git -C ComfyUI/custom_nodes clone https://github.com/ltdrdata/ComfyUI-TTS || echo "无法克隆 ComfyUI-TTS（可能仓库不存在或网络问题），跳过。"
else
  echo "ComfyUI-TTS 已存在，跳过。"
fi

echo "安装 Edge-TTS（本地 TTS ，支持中文）和其他音频库"
$CONDA_RUN python -m pip install edge-tts pydub

echo "安装一些常用 ML/视频工具（可选）"
$CONDA_RUN python -m pip install accelerate transformers sentencepiece safetensors

# 创建模型目录与占位提示
mkdir -p "$PROJECT_DIR/models/diffusion" "$PROJECT_DIR/models/text_encoder" "$PROJECT_DIR/models/vae"
echo ""
echo "==== 请手动下载模型文件到对应目录 ===="
echo "Wan2.2 14B T2V 需要以下量化模型（示例文件名，实际文件名以你下载的为准）："
echo "  - models/diffusion/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
echo "  - models/diffusion/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
echo "  - models/text_encoder/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
echo "  - models/vae/wan_2.1_vae.safetensors"
echo ""
echo "模型通常很大（几十 GB），请确保有足够磁盘空间并手动放置后再运行生成流程。"
echo ""

# 生成一个示例 run 脚本：启动 ComfyUI
cat > "$PROJECT_DIR/run_comfyui.sh" <<'EOF'
#!/usr/bin/env bash
# 启动 ComfyUI（在激活 conda env 后运行）
ENV_NAME="wan2_14b"
echo "在新终端或 shell 中运行："
echo "conda activate $ENV_NAME"
echo "cd ~/wan2_14b_project/ComfyUI"
echo "python main.py --listen 0.0.0.0 --port 8188 --lowvram"
echo ""
echo "建议使用 --lowvram 或者 --precision full來节省显存/兼容量化模型。"
EOF
chmod +x "$PROJECT_DIR/run_comfyui.sh"

# 生成一个示例的 TTS + 合成脚本（edge-tts + ffmpeg）
cat > "$PROJECT_DIR/tts_and_merge.sh" <<'EOF'
#!/usr/bin/env bash
# 用法：./tts_and_merge.sh "要配音的文本" input_video.mp4 output_final.mp4
set -euo pipefail
TEXT="$1"
VIDEO_IN="$2"
VIDEO_OUT="${3:-final_with_voice.mp4}"
AUDIO_TMP="tmp_tts_audio.mp3"

echo "生成 TTS 音频..."
python - <<PY
import asyncio, sys
import edge_tts

text = sys.argv[1]
outfile = sys.argv[2]
voice = "zh-CN-XiaoxiaoNeural"  # 可换
async def go():
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(outfile)
asyncio.run(go())
PY "$TEXT" "$AUDIO_TMP"

echo "合成音视频（ffmpeg）..."
ffmpeg -y -i "$VIDEO_IN" -i "$AUDIO_TMP" -c:v copy -c:a aac -shortest "$VIDEO_OUT"

echo "输出：$VIDEO_OUT"
EOF
chmod +x "$PROJECT_DIR/tts_and_merge.sh"

echo ""
echo "==== 部署完成（脚本） ===="
echo "项目目录： $PROJECT_DIR"
echo "启动 ComfyUI："
echo "  1) conda activate $ENV_NAME"
echo "  2) cd $PROJECT_DIR/ComfyUI"
echo "  3) python main.py --listen 0.0.0.0 --port 8188 --lowvram"
echo ""
echo "生成流程示意（手动 / 脚本化）："
echo "  1) 在 ComfyUI 中加载 Wan2.2 14B 工作流模板，确认 Load Model 节点指向 models/ 下的模型文件"
echo "  2) 在 ComfyUI 中运行并输出 video 到 $PROJECT_DIR/output/"
echo "  3) 运行示例合成脚本："
echo "     ./tts_and_merge.sh \"这里是配音文本\" /path/to/generated_video.mp4 /path/to/final.mp4"
echo ""
echo "提示：14B 模型运算量大，建议："
echo "  - 先用低分辨率（480×480）短帧（16 帧）做测试"
echo "  - 在 ComfyUI 启用 lowvram / precision / offload 等设置"
echo ""
echo "若需要，我可以："
echo "  - 把 ComfyUI 的 14B 工作流模板 json 也生成并放到 workflows/ 中（需你确认模板参数）"
echo "  - 或者生成更完整的自动化脚本来调用 ComfyUI 的 CLI/REST（如果你希望完全无人值守）"
echo ""
echo "完成 ✅"
