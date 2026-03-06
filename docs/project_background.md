# 项目背景

> 本文档面向新手和协作 Agent，概述 VideoThinkBench 项目的研究背景、核心论文发现和可复现资源。

## 项目立场

**本项目属于 Thinking with Video 阵营。** VideoThinkBench 是我们自己的代码仓库和评测基准。VBVR 是另一个团队的独立工作，其数据格式、任务定义与 VideoThinkBench 完全不同，**两者数据不兼容**。VBVR 对我们的价值仅在于**借鉴其训练配置和 Scaling 经验**。

---

## 核心研究问题

视频生成模型（VGMs）在时空一致的环境中编码空间结构、物理动力学和因果关系，有望成为推理基底。两篇核心论文分别从**评测**和**训练**两个角度切入：

- **Thinking with Video**（CVPR 2026）：提出 VideoThinkBench 评测基准，证明 Sora-2 在视觉推理上可超越 VLM。
- **VBVR**：构建 100 万级训练数据 + 规则化评测框架，证明 LoRA 微调 Wan2.2-14B 即可超越所有闭源模型。

---

## 资源总览

### Thinking with Video

| 资源                   | 地址                                                          | 说明                                                   |
| ---------------------- | ------------------------------------------------------------- | ------------------------------------------------------ |
| GitHub 代码            | https://github.com/tongjingqi/Thinking-with-Video             | 评测脚本、数据生成器                                   |
| VideoThinkBench 数据集 | https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench | 含 minitest（750 样本）和 test（4,149 样本）两个 split |
| 项目主页               | https://thinking-with-video.github.io                         | Leaderboard                                            |

### VBVR

| 资源             | 地址                                                         | 说明                              |
| ---------------- | ------------------------------------------------------------ | --------------------------------- |
| VBVR-Wan2.2 模型 | https://huggingface.co/Video-Reason/VBVR-Wan2.2              | LoRA 权重（基于 Wan2.2-I2V-A14B） |
| VBVR-Dataset     | https://huggingface.co/datasets/Video-Reason/VBVR-Dataset    | 100 万训练样本                    |
| VBVR-Bench-Data  | https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data | 测试集（100 任务 × 5 样本）       |
| VBVR-EvalKit     | https://github.com/Video-Reason/VBVR-EvalKit                 | 评测代码（规则化评分器）          |
| VBVR-DataFactory | https://github.com/VBVR-DataFactory                          | 所有数据生成器                    |

---

## VideoThinkBench 任务结构

### Vision-Centric（2,696 样本）

| 任务类型                      | 样本数 | 评估方式         | 输入格式            |
| ----------------------------- | ------ | ---------------- | ------------------- |
| Eyeballing Puzzles（23 子类） | 1,050  | 像素级自动验证   | 条件图 + 文字指令   |
| Mazes（方形/六角形/迷宫型）   | 150    | 路径连通性验证   | 迷宫图 + 起终点     |
| Visual Puzzles（10 子类）     | 496    | 模式匹配         | 多帧示例 + 待推理帧 |
| ARC-AGI-2                     | 1,000  | 颜色矩阵精确匹配 | 输入输出示例对      |

当前仓库的统一任务注册表以 `tasks/specs.py` 为准，包含 23 个 `eyeballing`、3 个 `maze`、10 个 `visual_puzzle`，合计 36 个任务。上表中的样本数用于背景说明，不替代当前代码中的任务注册表统计。

### Text-Centric（1,453 样本）

| 来源                     | 样本数 | 领域          |
| ------------------------ | ------ | ------------- |
| GSM8K                    | ~100   | 小学数学      |
| MATH-500                 | ~100   | 高中/竞赛数学 |
| AIME24 + AIME25          | ~60    | 数学奥赛      |
| MMLU + MMLU-Pro          | ~200   | 综合知识      |
| MMMU + MMBench           | ~200   | 多模态理解    |
| GPQA-diamond + SuperGPQA | ~100   | 科学推理      |
| MathVista + MathVision   | ~200   | 多模态数学    |

---

## 已有训练实验（迷宫数据）

| 配置     | 详情                                          |
| -------- | --------------------------------------------- |
| 基模     | Wan2.2-TI2V-**5B**                            |
| 微调方式 | LoRA，rank=32                                 |
| 训练数据 | 7×7、9×9、10×10 迷宫各 10,000 条（共 30,000） |
| 训练轮数 | 5 epoch                                       |
| 结果     | **无效果**                                    |

更早之前还在 maze + eyeballing 的多任务混合数据上训过（同样用 5B），也没有效果。

---

## VBVR 给我们的经验启示

| 我们的做法      | VBVR 的做法    | 差距               |
| --------------- | -------------- | ------------------ |
| 基模 5B         | 基模 14B       | ~3 倍              |
| 单任务或 2 任务 | 100 个任务混合 | ~50 倍             |
| 30,000 样本     | 1,000,000 样本 | ~33 倍             |
| 5 epoch         | 1 epoch        | 我们训的更多但无效 |
| LoRA rank=32    | LoRA rank=32   | 相同               |

**核心差异是模型规模和数据多样性**，不是训练轮数或 LoRA 配置。

---

## VBVR Scaling 曲线

| 数据量     | Overall   | ID Avg.   | OOD Avg.  |
| ---------- | --------- | --------- | --------- |
| 0K（基线） | 0.371     | 0.412     | 0.329     |
| 50K        | 0.549     | 0.576     | 0.522     |
| 200K       | **0.689** | **0.767** | **0.611** |
| 500K       | 0.685     | 0.760     | 0.610     |
| 人类       | 0.974     | 0.960     | 0.988     |

**200K 是性价比最高的数据量**——之后 ID/OOD 均进入平台期。
