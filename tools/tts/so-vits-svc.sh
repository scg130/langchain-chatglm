#!/bin/bash
# svc5.0 https://openi.pcl.ac.cn/neuro/so-vits-svc-5.0/src/branch/bigvgan-mix-v2/README_ZH.md
echo '是否重新安装so-vits。y/n'
read sovits

if [ "$sovits" = 'y' ];then
    cd /usr/local/src
    git clone -b 4.1-Stable https://github.com/svc-develop-team/so-vits-svc.git
    cd /usr/local/src/so-vits-svc/pretrain/
    wget  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -O checkpoint_best_legacy_500.pt

    cd /usr/local/src/so-vits-svc/logs/44k
    # wget  https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/D_0.pth
    wget -O D_0.pth  https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/D_0.pth?download=true
    # wget  https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/G_0.pth
    wget -O G_0.pth  https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/G_0.pth?download=true
    cd /usr/local/src/so-vits-svc/logs/44k/diffusion
    # wget  https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/model_0.pt
    wget -O model_0.pt https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/model_0.pt?download=true
 
    cd /usr/local/src/so-vits-svc

    #如果使用rmvpeF0预测器的话，需要下载预训练的 RMVPE 模型
    curl -L https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt -o pretrain/rmvpe.pt
    curl -L https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt -o pretrain/fcpe.pt

    curl -L https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip -o nsf_hifigan_20221211.zip
    md5sum nsf_hifigan_20221211.zip
    unzip nsf_hifigan_20221211.zip
    rm -rf pretrain/nsf_hifigan
    mv -v nsf_hifigan pretrain

    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    chmod -R 755 ./venv/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2
    sed -i 's/app.launch()/app.launch(share=True)/g' /usr/local/src/so-vits-svc/webUI.py
    pip install gradio==3.50.2
elif [ "$sovits" = 'n' ];then
    echo 'pass'
else
    exit
fi


echo '是否练丹。提前准备数据集(dataset/)y/n'
read flag

if [ "$flag" = 'y' ];then
    cd /usr/local/src/so-vits-svc/
    echo '输入模型名称'
    read model
    source venv/bin/activate
    python resample.py
    # 4.0
    # python preprocess_flist_config.py --speech_encoder vec256l9
    # python preprocess_hubert_f0.py --f0_predictor dio
    # 4.1
    python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug 
    python preprocess_hubert_f0.py --f0_predictor crepe --use_diff

    # sed -i 's/"n_speakers": 1/"n_speakers": 1,\n        "speech_encoder":"vec256l9"/g' /usr/local/src/so-vits-svc/configs/config.json
    python train.py -c configs/config.json -m $model
    
    # 扩散模型（可选） 扩散模型在logs/44k/diffusion下
    python train_diff.py -c configs/diffusion.yaml
    # 聚类模型训练（可选）
    python cluster/train_cluster.py --dataset "./dataset_raw/$model" --output "./trained/$model" --gpu
elif [ "$flag" = 'n' ];then
    echo 'pass'
else
    exit
fi

echo '结束练丹保存模型y/n'
read flag

if [ "$flag" = 'y' ];then
    cd /usr/local/src/so-vits-svc/
    echo '输入模型名称'
    read model
    mkdir -p trained/tmp
    source venv/bin/activate
    cp logs/$model/config.json trained/tmp/
    file_name=`ls -al logs/$model/G_*.pth | grep -v grep | awk 'END{print $9}'`
    echo $file_name
    cp $file_name trained/tmp/
    mkdir -p trained/$model/diffusion/
    cp logs/44k/diffusion/config.yaml trained/$model/diffusion/
    cp logs/44k/diffusion/model_10000.pt trained/$model/diffusion/
    cp trained/tmp/config.json trained/$model/
    python compress_model.py -c="trained/$model/config.json" -i="$file_name" -o="trained/$model/$model.pth"
    rm -fr trained/tmp
elif [ "$flag" = 'n' ];then
    echo 'pass'
else
    exit
fi



# pip install spleeter --user
# 分离人声 伴奏
# spleeter separate -o ./output/ -p spleeter:2stems ./input/遥远的歌.mp3
# ./output/vocals.wav    人声
# ./output/accompaniment.wav 伴奏


# 合变人声 伴奏 
# ffmpeg -i bz.wav -i gc.wav -filter_complex amix=inputs=2:duration=first:dropout_transition=3 output.wav
# 合成视频
# ffmpeg -i video.mp4 -i output.mp3 -c:v copy -c:a aac -strict experimental output.mp4



# ll *.mp3 | awk 'BEGIN{i=2}{print "ffmpeg -i " $9" " i ".wav";i=i+2 }'


# 提取视频里的声音
# ffmpeg -i 1.mp4 -vn  output.mp3
 
# 提取视频里的画面，过滤掉声音
# ffmpeg -i 1.mp4 -an  output.mp4
 
 
# 同时分离视频流和音频流
# ffmpeg -i 1.mp4 -vn -c:v copy audio.mp3 -an -c:v copy video.mp4