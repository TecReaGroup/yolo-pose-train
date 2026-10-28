# Pen Studio

使用 PySide6 构建的笔标注预览工具，读取 `pen_dataset/images` 与 `labels` 中的 YOLO Pose 标签。

```sh
make install
make app
```

已经安装 PySide6 时直接运行 `make app`。也可使用 `python app.py`。
指定其他目录：`make app DATASET="路径"` 或 `python app.py --dataset "路径"`。
可通过 `make app PYTHON="路径/python.exe"` 指定 Python（路径有空格时直接使用该解释器运行 app.py）。

界面提供文件名搜索、视频筛选、图片切换、独立框与关键点图层、像素坐标及可见性信息。
滚轮缩放，鼠标拖动画布；左右方向键翻页，F 适应画布，P 聚焦笔。
绿色为笔框，珊瑚红为 tip，蓝色为 tail；空心圆代表遮挡估计。
应用直接读取 TXT 标签叠加显示，预览操作不会修改图片和标签。

`make check` 会离屏加载全部图片与标签，检查筛选、翻页、图层和聚焦，并输出 `app-preview.png`。

抽帧与标注数据说明见 [pen_dataset/README.md](pen_dataset/README.md)。

## 训练 YOLO26 Pose

```sh
make install-train
make train
```

默认微调 `yolo26n-pose.pt`，输入 640，batch 8，最多 100 轮，25 轮无改善时早停。
使用已安装的 CUDA 版 PyTorch，默认 GPU 0；可用 `make train DEVICE=cpu` 选择 CPU。
训练入口为 `train_pose.py`，`make train EPOCHS=100 BATCH=8` 可调整参数。
最终产物为 `models/yolo26-pose.pt`，已有同名产物时会停止，避免覆盖。
原始训练权重和曲线位于 `runs/pose/`，训练状态保存在 `training_status.json`。

模型 checkpoint 和内部网络对象均保存 `object_class`、`object_point`，完整信息保存于 `metadata`。
另外输出 `.metadata.json` 便于直接查看；该 JSON 是副本，PT 模型本身包含元数据。

```python
from ultralytics import YOLO

model = YOLO('models/yolo26-pose.pt')
print(model.model.object_class)       # pen
print(model.model.object_point)       # ['tip', 'tail']
print(model.names)                    # {0: 'pen'}
print(model.model.kpt_names)          # {0: ['tip', 'tail']}
print(model.ckpt['metadata'])         # 版本、关键点定义、数据指纹等

result = model('pen_dataset/images/v2_065.jpg')[0]
boxes = result.boxes.xyxy             # 原图像素坐标
points = result.keypoints.xy          # [实例数, 2, 2]，tip 在前，tail 在后
confidence = result.keypoints.conf    # 预测置信度，不是标注可见性 0/1/2
```

训练后自动重新加载 PT、验证 40 张验证图并生成 4 张预测预览。
`python evaluate_pose.py` 进一步检查验证集原图像素误差，保存至 `models/endpoint_evaluation.json`。
两段视频来自相同拍摄场景，这些验证结果不能代表跨场景泛化能力。
