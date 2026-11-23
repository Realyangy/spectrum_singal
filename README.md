# RF-SepEdge 频谱信号端侧

## 一、系统功能

端侧包含两个任务：

- **任务一：混合信号分离与恢复**。选择目标信号、干扰类型和样本，使用 WaveNet 恢复目标信号，展示误码率（BER）、均方误差（MSE）和波形等结果。样本库包含六组，每组 50 个样本，共 300 个样本。
- **任务二：频谱检测与调制识别**。选择宽带 IQ 测试样本，使用 YOLO11n 检测信号，再使用 XCiT 识别调制家族。样本库包含 100 个样本。页面可查看整体时频图、各信号局部图和家族识别结果。

两个任务都由端侧发起请求，云侧选择并量化模型，Agent 接收模型，端侧校验和加载本次下发的模型完成推理，最后将结果回传云侧。端侧提交样本信息和本机路径引用，**不会上传输入 IQ 文件内容**。

任务二的标准家族来自样本库标注。页面按检测框与标注框 IoU ≥ 0.5 进行一对一匹配，未匹配的检测不会显示标准家族。页面的样本库整体表现来自已完成的实验结果，与当前单次任务分别展示。

## 二、目录和文件

本机项目路径：`/home/lenovo/cloud-edge/spectrum-signal-edge`。

```text
spectrum-signal-edge/
├── frontend/live/                     # 端侧页面及整体实验数据
├── frontend/runs/                     # 页面访问任务结果的链接
├── backend/                       # 端侧服务、Agent、预处理、推理、BER 及 qmodel 解码
├── spectrum-separation/
│   ├── sample-library/            # 任务一样本及真值
│   ├── interactions/              # 任务一运行记录与结果
│   ├── pretrained-models/         # WaveNet 原始权重
│   ├── src/wavenet_model/         # WaveNet 模型和配置
│   ├── src/rfcutils/              # 信号生成和解调工具
│   └── src/                       # 信号分离相关代码
├── spectrum-recognition/
│   ├── sample-library/            # 任务二样本和标注
│   ├── interactions/              # 任务二运行记录与结果
│   ├── models/                    # YOLO、XCiT 原始权重
│   ├── src/xcit_model/            # XCiT 共用模型实现及许可
│   ├── src/xcit_model/finetune_xcit_joint.py # XCiT 微调脚本
│   ├── configs/                   # 频谱任务配置
│   └── src/                       # 频谱分析相关代码
├── received-models/               # 端侧校验后的下发模型副本
├── all_results/                       # 样本库整体实验结果和实验脚本
├── results.zip                    # 整体实验结果压缩包
├── requirements-rf-spectrum.txt   # 频谱环境依赖参考
├── requirements-rf-sionna.txt     # BER 环境依赖参考
├── start.sh                       # 端侧服务启动脚本
└── README.md                      # 本说明
```

页面展示的整体实验数据位于 `frontend/live/experiment-data.json`。`all_results/` 是实验结果资料；单次在线任务的结果分别保存在两个任务的 `interactions/` 中。

## 三、环境与设备

需要能够运行 Linux 或 WSL 的设备、浏览器、可访问云侧服务的网络，以及两个 Python 环境：

| 环境 | 本机 Python | 用途 |
|---|---|---|
| `rf-spectrum` | `/home/lenovo/anaconda3/envs/rf-spectrum/bin/python`，Python 3.10 | 端侧服务、预处理和模型推理 |
| `rf-sionna` | `/home/lenovo/anaconda3/envs/rf-sionna/bin/python`，Python 3.7 | 任务一 BER 计算 |

`requirements-rf-spectrum.txt` 和 `requirements-rf-sionna.txt` 记录依赖版本，可用于配置新环境。XCiT 模型实现位于 `spectrum-recognition/src/xcit_model`。依赖版本、CUDA 和 NVIDIA 驱动能否在目标设备上使用，需要按目标设备实际情况核对。

`start.sh` 默认使用上表中的本机 Python 路径，并设置 XCiT 的 `PYTHONPATH`。迁移到其他设备时，请通过环境变量 `RF_SPECTRUM_PYTHON` 和 `RF_SIONNA_PYTHON` 指定新路径。推理在 PyTorch 可使用 CUDA 时使用 GPU，否则尝试 CPU。当前 qmodel 会被解码为浮点权重加载，不能据此认为使用了原生 INT8/INT4 推理加速。

可在项目根目录检查两个环境的基础依赖：

```bash
cd /home/lenovo/cloud-edge/spectrum-signal-edge
export RF_SPECTRUM_PYTHON=/home/lenovo/anaconda3/envs/rf-spectrum/bin/python
export RF_SIONNA_PYTHON=/home/lenovo/anaconda3/envs/rf-sionna/bin/python
export PYTHONPATH="$PWD/spectrum-recognition/src${PYTHONPATH:+:$PYTHONPATH}"
"$RF_SPECTRUM_PYTHON" -c 'import torch, numpy, cv2, ultralytics, torchsig, xcit_model, omegaconf; print("频谱环境可导入，CUDA:", torch.cuda.is_available())'
"$RF_SIONNA_PYTHON" -c 'import numpy, tensorflow, sionna; print("Sionna 环境可导入")'
```

导入检查只确认基础依赖可用。迁移后还应分别运行两个任务，检查模型接收、推理和结果回传。

## 四、连接云侧并启动

本机当前使用的参数：

| 项目 | 值 |
|---|---|
| 云服务器 SSH | `flyvideo@59.110.167.128`，端口 `6652` |
| 云侧 HTTP 服务 | 服务器回环地址 `127.0.0.1:8091` |
| 本机 SSH 隧道入口 | `http://127.0.0.1:18091` |
| 终端设备编号 | `lenovo-wsl` |
| Agent 模型接收目录 | `/home/lenovo/cloud-edge/model_repository` |
| 端侧页面 | `http://127.0.0.1:8080/` |

以下命令在本机 WSL/Linux 中执行。迁移时根据实际环境修改服务器地址、端口、Python 路径、设备编号和目录。

### 1. 建立 SSH 隧道

云侧服务需先在服务器 `127.0.0.1:8091` 监听。在本机终端运行并保持窗口打开：

```bash
ssh -N -p 6652 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 18091:127.0.0.1:8091 flyvideo@59.110.167.128
```

如果设备通过其他方式直接访问云侧 HTTP 服务，将下列 `RF_CLOUD_URL` 改为实际可访问的地址即可。

### 2. 启动 Agent

另开一个终端：

```bash
export RF_PROJECT=/home/lenovo/cloud-edge/spectrum-signal-edge
export RF_SPECTRUM_PYTHON=/home/lenovo/anaconda3/envs/rf-spectrum/bin/python
export RF_CLOUD_URL=http://127.0.0.1:18091
export RF_NODE_ID=lenovo-wsl
export RF_MODEL_ROOT=/home/lenovo/cloud-edge/model_repository

"$RF_SPECTRUM_PYTHON" "$RF_PROJECT/backend/edge_agent.py" \
  --cloud "$RF_CLOUD_URL" --check

"$RF_SPECTRUM_PYTHON" "$RF_PROJECT/backend/edge_agent.py" \
  --cloud "$RF_CLOUD_URL" --node-id "$RF_NODE_ID" --model-dir "$RF_MODEL_ROOT"
```

`--check` 只检查云侧连通性。Agent 启动后保持窗口运行；云侧控制台应显示对应终端在线。如果已有同一设备编号的 Agent 正在运行，请确认其云侧地址和模型接收目录一致，避免重复启动。

### 3. 启动端侧页面与后台

再开一个终端。注意环境变量只在当前终端生效：

```bash
export RF_PROJECT=/home/lenovo/cloud-edge/spectrum-signal-edge
export RF_SPECTRUM_PYTHON=/home/lenovo/anaconda3/envs/rf-spectrum/bin/python
export RF_SIONNA_PYTHON=/home/lenovo/anaconda3/envs/rf-sionna/bin/python
export RF_CLOUD_URL=http://127.0.0.1:18091
export RF_NODE_ID=lenovo-wsl
export RF_MODEL_ROOT=/home/lenovo/cloud-edge/model_repository

cd "$RF_PROJECT"
bash start.sh
```

浏览器打开 `http://127.0.0.1:8080/`。端侧服务默认只监听本机回环地址；如 8080 已占用，可在启动前设置 `export RF_PORT=8081`，然后访问 `http://127.0.0.1:8081/`。

Agent 的 `--node-id` 必须与端侧 `RF_NODE_ID` 一致，Agent 的 `--model-dir` 必须与端侧 `RF_MODEL_ROOT` 指向同一目录。停止时，分别在端侧服务、Agent 和 SSH 隧道的终端按 `Ctrl+C`。

## 五、执行任务

1. 确认云侧任务服务运行、所需模型已入库，且云侧控制台显示终端在线。
2. 在云侧“终端任务请求”选择处理模式：**自动**模式下，后续新任务无需人工确认，云侧按场景默认配置处理；**手动**模式下，需在云侧确认任务并操作模型选择、量化和下发。模式设置对后续新收到的终端任务生效。
3. 在端侧选择任务与测试样本，点击“发起请求”。任务一还需选择目标信号和干扰类型。
4. 云侧处理并下发模型后，Agent 接收文件；端侧验证文件大小和 SHA-256，将量化模型复制到项目的 `received-models/`，加载本次模型并处理样本。
5. 查看端侧“样本处理结果”。任务二可切换整体时频图和单个信号；任务一可切换波形和时频图。任务完成后，端侧会将 `result.json` 和结果图片回传云侧。

单次任务的样本副本、模型输入、模型信息、日志和结果分别保存在：

```text
spectrum-separation/interactions/<任务ID>/
spectrum-recognition/interactions/<任务ID>/
```

`frontend/runs/<任务ID>` 提供页面访问结果的链接。输入 IQ 和任务一恢复后的 IQ 保留在端侧。

## 六、清理已下发模型

端侧切换到要清理的任务，在“云边协同处理”中点击“清理本任务模型”。页面会显示可清理的文件数量和大小，确认后删除能由本地任务记录识别的 qmodel，包括 Agent 接收目录中的文件和项目内的推理副本。

任务运行时不能清理。该功能保留样本、处理结果、原始权重及其他业务模型；没有对应接收记录或校验不通过的文件会保留。清理后发起新任务时，Agent 会按云侧下发的任务重新接收模型。

## 七、常见问题

- **云侧连接失败**：检查云侧服务、SSH 隧道和 `RF_CLOUD_URL`，并使用 Agent 的 `--check` 检查连通性。
- **Agent 不在线或模型未收到**：检查设备编号、云侧地址和模型接收目录是否与端侧服务一致，并查看云侧任务和下发状态。
- **模型校验或加载失败**：查看对应任务的 `model-info.json` 和 `worker.log`，确认云侧下发的 qmodel 包含当前任务所需模型。项目内原始权重不会自动替代未下发的模型。
- **任务一 BER 计算失败**：检查 `RF_SIONNA_PYTHON` 是否指向可用的 `rf-sionna` 环境，并查看任务日志。
- **样本或模型输入校验失败**：检查样本文件是否完整及依赖环境是否匹配；不要直接修改样本清单中的哈希。
- **页面显示旧内容**：刷新浏览器；修改端侧 Python 服务后需要重启端侧服务。

云端任务标识仍使用 `signal-separation` 和 `spectrum-analysis`，与本地目录名分别为 `spectrum-separation` 和 `spectrum-recognition`。
