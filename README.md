# FLUX.2 4B Gradio 项目

基于 `black-forest-labs/FLUX.2-klein-4B` 的 Gradio 图像生成与编辑演示。

## 环境要求

- Python 3.12
- 建议使用 GPU；CPU 可运行但速度较慢

## 安装（uv）

建议先安装适合你的 PyTorch 版本（CPU/CUDA），再安装其余依赖：

```bash
uv venv --python 3.12
uv sync --python 3.12
uv pip install "torch"
```

如果你需要 CUDA 版本，请按 PyTorch 官网指引安装对应轮子。

已在 `uv.toml` 中配置清华源，`uv` 会自动使用：

```
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
```

## 运行

```bash
uv run --python 3.12 python app.py
```

启动后在浏览器打开终端输出的本地地址即可。

## 可选环境变量

- `FLUX_MODEL_ID`：替换模型 ID（默认 `black-forest-labs/FLUX.2-klein-4B`）
- `USE_CPU_OFFLOAD=1`：启用 CPU offload 以降低显存占用（默认开启）

## 使用说明

- “编辑图片”支持上传多张参考图，模型会结合这些图进行编辑。
- 输出尺寸会对齐到 16 的倍数。
