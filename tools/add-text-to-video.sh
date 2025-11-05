ffmpeg -loop 1 -i "infoflow 2025-09-17 16-25-58.png" -t 6 \
-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,\
drawtext=fontfile=/System/Library/Fonts/PingFang.ttc:fontsize=18:fontcolor=white:x=(w-text_w)/2:y=h-80:text='狗':enable='between(t,0,0.5)',\
drawtext=fontfile=/System/Library/Fonts/PingFang.ttc:fontsize=18:fontcolor=white:x=(w-text_w)/2:y=h-80:text='狗东':enable='between(t,0.5,1)',\
drawtext=fontfile=/System/Library/Fonts/PingFang.ttc:fontsize=18:fontcolor=white:x=(w-text_w)/2:y=h-80:text='狗东西':enable='between(t,1,1.5)',\
drawtext=fontfile=/System/Library/Fonts/PingFang.ttc:fontsize=18:fontcolor=white:x=(w-text_w)/2:y=h-90:text='狗东西 该':enable='between(t,1.5,2)',\
drawtext=fontfile=/System/Library/Fonts/PingFang.ttc:fontsize=18:fontcolor=white:x=(w-text_w)/2:y=h-90:text='狗东西 该下':enable='between(t,2,2.5)',\
drawtext=fontfile=/System/Library/Fonts/PingFang.ttc:fontsize=18:fontcolor=white:x=(w-text_w)/2:y=h-90:text='狗东西 该下班':enable='between(t,2.5,3)',\
drawtext=fontfile=/System/Library/Fonts/PingFang.ttc:fontsize=18:fontcolor=white:x=(w-text_w)/2:y=h-90:text='狗东西 该下班吃':enable='between(t,3,3.5)',\
drawtext=fontfile=/System/Library/Fonts/PingFang.ttc:fontsize=18:fontcolor=white:x=(w-text_w)/2:y=h-90:text='狗东西 该下班吃饭':enable='between(t,3.5,4)',\
drawtext=fontfile=/System/Library/Fonts/PingFang.ttc:fontsize=18:fontcolor=white:x=(w-text_w)/2:y=h-90:text='狗东西 该下班吃饭了':enable='between(t,4,6)'" \
-pix_fmt yuv420p -r 30 output_typewriter.mp4
