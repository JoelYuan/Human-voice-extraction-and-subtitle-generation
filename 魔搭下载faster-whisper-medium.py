# 下载模型到当前目录，约1.42GB，完成后注意将faster-whisper-medium剪切放到根目录
# https://www.modelscope.cn/
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    model_id='pengzhendong/faster-whisper-medium',
    cache_dir='.'            # 下载到当前目录
)