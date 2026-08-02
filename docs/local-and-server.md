# 本地与 T4 部署说明

## macOS 本地

安装依赖：

```bash
python -m pip install -r requirements-mac.txt
```

当前项目根目录已有 `yolo11s-pose.pt`，本地配置直接使用这个文件。加载器仍兼容服务器常用的 `models/yolo11s-pose.pt` 路径，因此部署到服务器时无需改代码。

macOS Apple Silicon 可以直接运行 YOLO，程序会自动选择 PyTorch `mps`；Intel macOS 或 MPS 不可用时才使用 CPU。已将 `YOLO_Trainer/backend/yolov8s.pt` 复用到当前项目的 `models/yolov8s.pt`，但它是普通 `detect` 权重，只能提供人物框，不能提供姿态关键点，也不能替代 TrackNet。需要使用时，将 `yolo_pose.weights` 改为该路径并设置 `task: detect`，MediaPipe 会继续负责姿态关键点和角度。

当前项目已接入公开 `TrackNetV3_TableTennis` 的 `TrackNet_best.pt`，实际文件位于：

```text
models/TrackNet_best.pt
```

它来自项目根目录的 `TrackNetV3_ckpts.zip`，加载器会根据 checkpoint 自动识别
`seq_len=8`、`bg_mode=concat` 和 27 通道输入。在线处理时使用当前 8 帧窗口估计中值背景，
再按官方格式拼接为“背景 + 8 帧 RGB”。压缩包中的 `InpaintNet_best.pt` 暂未接入，因为当前回合分析只需要 TrackNet 热力图。

在 Apple Silicon Mac 上会用 PyTorch `mps` 推理；Intel Mac 或 MPS 不可用时使用 CPU。Mac 不支持 NVIDIA TensorRT engine，因此本地使用 `.pt` 权重即可。

模型下载链接来自该项目 README：

<https://1drv.ms/u/c/ab3b33d5410e04f3/IQCwzwpuGP6pSpgw0VyyRSCzAa4jTyVFYiFWUgSd8gPeCf0?e=hWQh7G>

没有这个文件时，程序会明确记录 `TrackNet 权重不可用`，继续使用纯 CV 回退。此回退适合开发和合成视频测试，不等价于真实 TrackNet 精度。

启动本地服务：

```bash
python -m pingpong_analyst.api --host 127.0.0.1 --port 8077
```

访问 <http://127.0.0.1:8077>。

手工切片测试可直接使用当前保留的真实视频：

```text
data/uploads/0c42afad_test.mp4
```

自动化测试仍使用临时生成的小视频，只验证接口和算法边界，不会每次运行都解码 1 GB 文件，也不会再向正式 `data/` 视频库写入测试文件。

## 两类分析入口

回合分析和动作分析是两个独立任务，不会互相加载模型：

- `POST /api/analyze/{video_id}`：只运行 TrackNet/纯 CV 球检测、球台轨迹和回合切分，生成回合剪辑；不会加载 YOLO-Pose 或 MediaPipe，也不输出击球者和动作类型。
- `POST /api/action/analyze/{video_id}`：只运行 YOLO-Pose 和 MediaPipe，返回逐帧关键点、手臂角度、躯干指标和身份关联结果，不生成回合剪辑。

动作分析默认处理原始视频。分析某个已生成的片段时传入
`?clip_filename=rally_001_b8_1.0s.mp4`，分析该视频的全部片段时传入
`?all_clips=true`。身份关联优先使用服装颜色等外观特征和短期轨迹；无法确认时返回 `unknown`，不会强行按画面左右侧定义选手身份。

实时 WebSocket 也支持 `?mode=rally` 和 `?mode=action`。网页中的“回合追踪”使用前者，“动作分析”使用独立的后台任务入口。

## TrackNet 手动适配

网页选择视频后，点击“标注球”打开标注工作区。拖动时间轴选择帧，在画面中点击球心保存
`ball` 标注；没有看见球时使用“无球 / 不可见”，不确定的帧使用“跳过此帧”。标注只写入
`data/tracknet_annotations/<video_id>.json`，不会修改原视频。

建议至少标注 30 个球和 10 个无球帧。少于 12 个 `ball` 标注时不会启动训练。训练使用
TrackNetV3 的官方 27 通道输入（3 通道中值背景 + 8 帧 RGB），只对明确标注的目标帧计算损失，
因此相邻未标注帧不会被误当成负样本。

点击“开始微调”后 API 会立即返回 job id，训练在后台运行。macOS Apple Silicon 使用 MPS，
MPS 不可用时回退 CPU；T4 服务器使用 CUDA。训练不使用 TensorRT。训练完成后必须手动点击
“保存模型版本”，否则临时结果可以放弃。保存后的模型位于：

```text
models/runs/<run_id>/TrackNet_best.pt
models/runs/<run_id>/metadata.json
models/runs/index.json
```

基础模型和保存版本会同时出现在 TrackNet 模型下拉框中。新版本不会自动替换基础模型，
回合追踪、回合剪辑和 WebSocket 只有在用户手动选择后才会携带 `model_id`；动作分析入口
不使用 TrackNet 模型选择。训练后的 `.pt` 可以先在本地验证，服务器上如需 TensorRT 推理，
再针对 `[1, 27, 288, 512] -> [1, 8, 288, 512]` 导出对应 engine。

## T4 服务器

服务器需要先安装匹配 CUDA 12.x 和 TensorRT 8.6+ 的系统运行时，再安装：

```bash
python -m pip install -r requirements-server.txt
```

推荐放置以下资产：

```text
models/yolo11s-pose.pt
models/yolo11s-pose.engine
models/TrackNet_best.pt
models/TrackNetV3.engine
```

TrackNetV3 engine 必须与当前配置一致：

```text
input:  [1, 27, 288, 512]
output: [1, 8, 288, 512]
```

### 球候选区域规则

回合追踪默认只接受画面中间走廊内的 TrackNet 候选，默认归一化范围是
`x=0.25..0.75`、`y=0.10..0.90`。这是两名选手之间的高概率区域，可以在
`config.yaml -> models.tracknet.center_region` 中按机位调整。回合检测器确认一个回合
结束后，追踪器会按 `outside_after_rally_frames`（默认 30 帧）临时放宽到全画面，覆盖
得分后球被打到场边或离开中间区域的情况；之后自动恢复中间区域过滤。

这条规则是候选过滤和轨迹重置，不会加载 YOLO 或 MediaPipe，也不会把躯干分析带入回合入口。

YOLO engine 可用 Ultralytics 导出：

```bash
yolo export model=models/yolo11s-pose.pt format=engine imgsz=640 half=True device=0
```

服务器启动脚本默认使用 `/opt/pingpong_analyst`、Conda 环境 `pingpong` 和 8077 端口：

```bash
bash start.sh start
bash start.sh status
```

如果项目目录或 Conda 环境不同，先修改 `start.sh` 顶部的 `APP_DIR`、`CONDA_ENV` 和 `CONDA_SH`。

## 球台四点标定

在固定机位下，按“图像左上、右上、右下、左下”的顺序填写球台四角，坐标单位是原始视频像素：

```yaml
analysis:
  rally:
    table:
      corners:
        - [210, 180]
        - [1710, 190]
        - [1880, 900]
        - [80, 890]
      turning_min_displacement: 0.02
      player_side_margin: 0.15
```

换机位、裁剪方式或分辨率后，需要重新填写四点。未填写时仍使用原图坐标模式，但不会启用球台两端约束。
