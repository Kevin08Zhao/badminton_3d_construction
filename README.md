# ShuttleVision · 羽毛球 3D 轨迹重建

基于**单目视频**的羽毛球分析系统：从一段固定机位录像中检测羽毛球 2D 轨迹、识别击球事件、重建 3D 飞行曲线，并在 Web 界面中可视化分析结果。

## 功能概览

- **TrackNet**：逐帧检测羽毛球 2D 位置
- **YOLO Pose + ByteTrack**：跟踪近端/远端球员骨架
- **HitNet (GRU)**：识别击球时刻，切分回合
- **3D 重建**：结合 6 点场地标定，优化物理轨迹并输出指标
- **Web UI**：上传视频、点击标定、查看进度日志、下载 CSV / 视频 / 图表

## 环境要求

- Python 3.10–3.13
- Node.js 18+（仅 Web UI 开发模式需要）
- macOS / Linux / Windows；GPU 可选（CPU / Apple MPS 亦可运行）

## 安装

```bash
cd badminton_3d_construction

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

若使用 NVIDIA GPU，请先从 [pytorch.org](https://pytorch.org) 安装对应 CUDA 版 PyTorch，再执行 `pip install -r requirements.txt`。

### 模型权重

仓库已通过 Git LFS 包含以下权重（克隆后若缺失，请确认已安装 [Git LFS](https://git-lfs.com) 并执行 `git lfs pull`）：

| 文件 | 用途 |
|------|------|
| `yolov8x-pose.pt` | 球员姿态检测 |
| `data/weights/ckpts/TrackNet_best.pt` | 羽毛球 2D 检测 |
| `data/weights/hitnet_output/hitnet_overfit_best.pth` | 击球事件识别 |

## 运行（推荐：Web 应用）

**终端 1 — 启动后端（端口 8000）：**

```bash
source .venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

**终端 2 — 启动前端：**

```bash
cd web-ui
npm install
npm run dev
```

浏览器打开 Vite 提示的地址（通常为 `http://localhost:5173`）。

### Web 使用流程

1. 上传一段羽毛球比赛/训练视频
2. 在视频帧上点击 **6 个标定点**（映射到 3D 球场坐标）
3. 选择性能模式（`fast` / `standard` / `precise`）并开始分析
4. 实时查看日志与进度；完成后下载 3D 轨迹 CSV、渲染视频、热力图等产物

任务输出保存在 `result/web/<job_id>/`（该目录已在 `.gitignore` 中排除）。

## 运行（命令行批处理）

编辑 `main.py` 末尾的路径（默认 `data/video/clip3.mp4`），然后：

```bash
source .venv/bin/activate
python main.py
```

标定工具（手动点击 4 个场地角点）：

```bash
python get_court_corners.py
```

## 可选环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SHUTTLEVISION_TRACKNET` | TrackNet 权重路径 | `data/weights/ckpts/TrackNet_best.pt` |
| `SHUTTLEVISION_HITNET` | HitNet 权重路径 | `data/weights/hitnet_output/hitnet_overfit_best.pth` |
| `SHUTTLEVISION_POSE` | YOLO 姿态权重路径 | `yolov8x-pose.pt` |
| `MPLBACKEND` | Matplotlib 后端 | `Agg` |

## 项目结构

```
badminton_3d_construction/
├── main.py                 # 命令行完整流水线
├── pipeline.py             # 可编程调用接口
├── render_player_pose_2d.py
├── get_court_corners.py
├── backend/                # FastAPI 后端
├── trackNetV3/             # 羽毛球 2D 检测
├── hitNet/                 # 击球识别模块
├── web-ui/                 # React 前端
├── data/
│   ├── video/              # 示例视频
│   └── weights/            # 模型权重
└── requirements.txt
```

## API 端点

- `POST /api/upload` — 上传视频
- `POST /api/jobs` — 创建分析任务
- `GET /api/jobs/{job_id}` — 查询任务状态
- `GET /api/jobs/{job_id}/logs` — 增量日志
- `GET /api/artifacts/{job_id}/{name}` — 下载产物
