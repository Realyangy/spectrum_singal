# 频谱信号端侧

本项目提供两个端侧任务：

- **信号分离与恢复**：使用 WaveNet 恢复 QPSK 或 OFDMQPSK 目标信号，并对有真值的固定样本计算 MSE 和 BER。
- **频谱检测与调制识别**：将宽带 IQ 转换为时频图，由 YOLO11n 检测信号位置，再提取对应 IQ，使用 XCiT 识别调制家族。

端侧发起请求，云侧选择并处理模型，Agent 接收模型，端侧校验、加载模型并推理，然后回传结果和图片。原始输入 IQ 默认留在端侧，不随任务请求上传。

## 项目结构

```text
spectrum-signal-edge/
├── frontend/                 # 前端页面及结果链接
├── backend/                  # 任务服务、Agent、预处理、推理和 BER
├── spectrum-separation/      # 分离模型代码、样本和运行结果
├── spectrum-recognition/     # 检测识别模型代码、样本和运行结果
├── received-models/          # 校验后的下发模型副本
├── all_results/              # 固定样本的整体实验资料
├── requirements-rf-spectrum.txt
├── requirements-rf-sionna.txt
└── start.sh                  # 启动频谱后台和页面
```

固定样本库位于各任务的 `sample-library/`：分离任务 300 个样本，识别任务 100 个样本。当前页面读取固定样本库，不能直接选择外部 `.dat` 或 `.pcm` 文件。

## 运行环境

本机使用两个 Conda 环境：

| 环境 | 本机 Python | 用途 |
|---|---|---|
| `rf-spectrum` | `/home/lenovo/anaconda3/envs/rf-spectrum/bin/python` | 后台、Agent、预处理和模型推理 |
| `rf-sionna` | `/home/lenovo/anaconda3/envs/rf-sionna/bin/python` | 信号分离 BER 计算 |

依赖参考分别为 `requirements-rf-spectrum.txt`、`requirements-rf-sionna.txt`。迁移时需按目标设备检查 Python、GPU 驱动、CUDA 及依赖兼容性。当前模型包在端侧解码后加载为浮点权重，不应据此认定使用了原生 INT8 加速。

## 启动和连接云侧

`start.sh` **只启动频谱后台和前端页面**。云侧服务、网络连接和 Agent 需要已运行；如果已有同编号公共 Agent 正在服务，应复用它，不要重复启动同编号 Agent。

### 1. 检查云侧和 Agent

本机网线环境可通过已有 SSH 隧道访问云侧，频谱后台默认云地址为 `http://127.0.0.1:18091`。确认云侧可访问：

```bash
curl --noproxy '*' http://127.0.0.1:18091/api/health
```

服务器服务只监听本机时，需要先在本机 WSL 中建立隧道；网线连接示例（保持终端运行）：

```bash
ssh -N -p 22 -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18091:127.0.0.1:8091 flyvideo@192.168.50.1
```

如果 18091 已由有效的 SSH 隧道占用，直接复用即可。服务器网线 IP 和 SSH 端口应以现场实际配置为准。

在云侧确认 `lenovo-wsl` 终端在线。若没有 Agent，可在另一个终端启动项目内 Agent：

```bash
cd /home/lenovo/cloud-edge/spectrum-signal-edge
/home/lenovo/anaconda3/envs/rf-spectrum/bin/python backend/edge_agent.py \
  --cloud http://127.0.0.1:18091 \
  --node-id lenovo-wsl \
  --model-dir /home/lenovo/cloud-edge/model_repository
```

Agent 和后台使用的终端编号、模型目录必须一致。项目外也可能运行服务于其他任务的公共 Agent，请先检查，避免同编号 Agent 重复运行。

### 2. 启动频谱页面

在本机 WSL 新开终端：

```bash
cd /home/lenovo/cloud-edge/spectrum-signal-edge
bash start.sh
```

浏览器打开 <http://127.0.0.1:8080/>。结束后台时，在启动它的终端按 `Ctrl+C`。脚本不负责停止 Agent 或 SSH 隧道。

需要修改参数时，可在启动前设置以下环境变量；Agent 必须使用相同的云地址、编号和模型目录：

```bash
export RF_CLOUD_URL=http://127.0.0.1:18091
export RF_NODE_ID=lenovo-wsl
export RF_MODEL_ROOT=/home/lenovo/cloud-edge/model_repository
export RF_PORT=8080
export RF_SPECTRUM_PYTHON=/home/lenovo/anaconda3/envs/rf-spectrum/bin/python
export RF_SIONNA_PYTHON=/home/lenovo/anaconda3/envs/rf-sionna/bin/python
bash start.sh
```

## 使用流程

1. 确认云侧服务、网络和相同终端编号的 Agent 在线。
2. 打开端侧页面，选择任务和测试样本；信号分离还需选择目标信号和干扰类型。
3. 发起请求。云侧自动或手动选择模型并下发，具体由云侧当前模式决定。
4. 端侧校验并加载本次模型，处理样本，显示结果并回传结果 JSON 与图片。

单次任务记录位于 `spectrum-separation/interactions/<任务ID>/` 或 `spectrum-recognition/interactions/<任务ID>/`；前端通过 `frontend/runs/` 中的链接访问结果。模型接收位置由 `RF_MODEL_ROOT` 配置，校验后的副本在 `received-models/`。页面整体实验数据位于 `frontend/live/experiment-data.json`，相关实验资料位于 `all_results/`。

云侧协议中的任务标识分别为 `signal-separation`、`spectrum-analysis`，与本地目录名不同，不要仅因目录改名而修改协议值。

## 常见检查

- **页面打不开**：检查 `bash start.sh` 是否运行、8080 端口是否被占用；也可通过 `RF_PORT` 设置其他端口。
- **云侧未连接**：检查云侧健康接口、SSH 隧道和 Agent 心跳。
- **无法接收模型**：检查 Agent 编号及模型目录是否与后台一致，并查看 Agent 日志。
- **信号分离 BER 失败**：检查 `rf-sionna` 环境及样本真值。
- **迁移到新设备**：`.gitignore` 排除固定样本、运行记录和部分实验明细；仅克隆仓库可能不足以运行，需另行准备这些文件和模型。

- **端口号被占用**sudo lsof -iTCP:8080 -sTCP:LISTEN -nP查询占用该端口号的进程，然后kill相应的PID即可。