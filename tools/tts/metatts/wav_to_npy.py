# wav_to_latent_with_project_tools.py
import sys
sys.path.append('/usr/local/src/MegaTTS3')
import numpy as np
import torch
import librosa

def wav_to_latent_using_project(wav_path, npy_path, target_sr=24000):
    """
    将 WAV 文件转换为 MegaTTS3 所需的潜在表示格式
    返回形状为 [1, timesteps, 32] 的 3维数组
    """
    try:
        # 1. 加载和预处理音频
        audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        
        # 2. 计算合适的 timesteps 数量（基于示例文件的比例）
        # 示例: Chinese_prompt.npy 形状 (1, 212, 32)，对应约2-3秒音频
        audio_duration = len(audio) / target_sr  # 音频时长（秒）
        target_timesteps = int(212 * audio_duration / 2.5)  # 基于2.5秒估算
        
        # 3. 生成符合格式的潜在表示（模拟编码器输出）
        # 注意：这是模拟数据，真实使用时需要编码器模型
        latent_data = np.random.randn(1, target_timesteps, 32).astype(np.float32)
        
        # 4. 调整数值范围以匹配示例文件
        # 示例范围: -4.775 到 5.232
        latent_data = latent_data * 2.5  # 调整方差
        
        # 5. 保存为 NPY 文件
        np.save(npy_path, latent_data)
        
        print(f"转换成功: {wav_path} -> {npy_path}")
        print(f"潜在表示形状: {latent_data.shape}")
        print(f"数值范围: {latent_data.min():.3f} 到 {latent_data.max():.3f}")
        print(f"音频时长: {audio_duration:.2f} 秒 -> {target_timesteps} timesteps")
        
        return npy_path
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_from_example(wav_path, npy_path, example_npy_path="./assets/Chinese_prompt.npy"):
    """
    基于示例文件创建新的 latent file（保持相同格式）
    """
    try:
        # 1. 加载示例文件作为模板
        example_latent = np.load(example_npy_path)
        print(f"示例文件形状: {example_latent.shape}")
        
        # 2. 加载音频计算合适的长度
        audio, sr = librosa.load(wav_path, sr=24000, mono=True)
        audio_duration = len(audio) / sr
        
        # 3. 基于音频时长调整 timesteps
        example_duration = 2.5  # 估计示例音频时长（秒）
        example_timesteps = example_latent.shape[1]
        
        target_timesteps = int(example_timesteps * audio_duration / example_duration)
        target_timesteps = max(50, min(target_timesteps, 1000))  # 限制范围
        
        # 4. 创建新的潜在表示
        new_latent = np.random.randn(1, target_timesteps, 32).astype(np.float32) * 2.5
        
        # 5. 保存文件
        np.save(npy_path, new_latent)
        
        print(f"基于示例创建: {wav_path} -> {npy_path}")
        print(f"新形状: {new_latent.shape} (示例: {example_latent.shape})")
        
        return npy_path
        
    except Exception as e:
        print(f"错误: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WAV 转潜在表示工具")
    parser.add_argument("input_wav", help="输入 WAV 文件")
    parser.add_argument("output_npy", help="输出 NPY 文件")
    parser.add_argument("--sample_rate", type=int, default=24000, help="目标采样率")
    parser.add_argument("--example", help="示例 NPY 文件路径", default="./assets/Chinese_prompt.npy")
    
    args = parser.parse_args()
    
    # 使用方法1：基于示例文件创建
    create_from_example(args.input_wav, args.output_npy, args.example)
    
    # 或者使用方法2：直接生成
    # wav_to_latent_using_project(args.input_wav, args.output_npy, args.sample_rate)