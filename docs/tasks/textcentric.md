# Text-Centric 独立流程说明

## 当前状态

`Text-Centric` 目前仍是独立流程，没有接入统一 `Canonical Manifest` 主线。

相关代码位于：

- `data/textcentric/request_videos.py`。
- `evaluation/textcentric/`。

## 当前流程

### 生成视频

入口脚本：

- `data/textcentric/request_videos.py`。
- 兼容脚本 `scripts/run_textcentric.sh`。

当前输入通常是题目 JSON，输出是：

- 模型原始响应。
- 视频下载地址。
- 下载后的视频文件。
- `responses.json` 与 `questions.json`。

### 评测视频

入口脚本：

- `evaluation/textcentric/evaluate_videos.py`。
- 兼容脚本 `scripts/eval_textcentric.sh`。

当前评测主要看：

- 最后一帧是否包含正确答案。
- 音频转写后是否包含正确答案。
- 两者同时正确。
- 任一正确。

## 为什么暂时不并入统一主线

因为这条链路的核心对象是“题目文本 + 生成视频 + 音频转写 + LLM 评判”，与当前 `Vision-Centric` 的静态题图 / 解图 / 解题视频中间层差异较大。

如果后续要并入统一主线，建议先设计一套独立于 `Vision-Centric` 的 `Text-Centric Canonical Schema`。
