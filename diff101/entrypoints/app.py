import os

import gradio as gr
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

MODEL_ID_DEFAULT = "black-forest-labs/FLUX.2-klein-4B"

_pipe: Flux2KleinPipeline | None = None
_pipe_device: str | None = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _load_pipe() -> tuple[Flux2KleinPipeline, str]:
    global _pipe, _pipe_device
    if _pipe is not None:
        if _pipe_device is None:
            raise gr.Error("模型缓存状态异常：设备信息为空")
        return _pipe, _pipe_device

    env_model_id = os.environ.get("FLUX_MODEL_ID")
    if env_model_id:
        model_id = env_model_id
    else:
        model_id = MODEL_ID_DEFAULT

    device = _get_device()
    dtype = _get_dtype(device)

    try:
        local_only = os.path.isdir(model_id)
        pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
    except Exception as e:
        raise gr.Error(e)

    if device == "cuda":
        # CPU offload reduces VRAM usage; turn off by setting USE_CPU_OFFLOAD=0
        if os.environ.get("USE_CPU_OFFLOAD", "1") == "1":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        pipe.to("cpu")

    _pipe = pipe
    _pipe_device = device
    return pipe, device


def _seed_generator(seed: int | None, device: str) -> torch.Generator | None:
    if seed is None or seed < 0:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _align_to_multiple(value: int, multiple: int = 16, min_value: int = 64) -> int:
    value = max(int(value), min_value)
    return value - (value % multiple)


def _sanitize_size(size: int, fallback: int) -> int:
    if size and size > 0:
        return int(size)
    return int(fallback)


def _image_list(base: Image.Image, extra_paths: list[str] | None) -> list[Image.Image]:
    images = [base]
    if extra_paths:
        for path in extra_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception:
                continue
    return images


def generate_image(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
) -> Image.Image:
    pipe, device = _load_pipe()
    width = _align_to_multiple(_sanitize_size(width, 1024))
    height = _align_to_multiple(_sanitize_size(height, 1024))
    generator = _seed_generator(seed, device)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
    return result.images[0]


def edit_image(
    base_image: Image.Image,
    extra_images: list[str] | None,
    prompt: str,
    use_input_size: bool,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
) -> Image.Image:
    if base_image is None:
        raise gr.Error("请先上传一张图片")

    pipe, device = _load_pipe()
    generator = _seed_generator(seed, device)

    if use_input_size:
        width, height = base_image.size
    else:
        width = _sanitize_size(width, base_image.size[0])
        height = _sanitize_size(height, base_image.size[1])
    width = _align_to_multiple(width)
    height = _align_to_multiple(height)

    image_list = _image_list(base_image, extra_images)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=image_list,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
    return result.images[0]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FLUX.2 4B Image Generation and Editing") as demo:
        with gr.Tabs():
            with gr.TabItem("生成图片"):
                with gr.Row():
                    with gr.Column(scale=3, min_width=360):
                        prompt = gr.Textbox(
                            label="提示词", lines=3, placeholder="输入英文或中文提示词"
                        )
                        with gr.Row():
                            width = gr.Slider(
                                512, 2048, value=1024, step=64, label="宽度"
                            )
                            height = gr.Slider(
                                512, 2048, value=1024, step=64, label="高度"
                            )
                        with gr.Row():
                            steps = gr.Slider(1, 12, value=4, step=1, label="步数")
                            guidance = gr.Slider(
                                0.0, 10.0, value=1.0, step=0.1, label="Guidance"
                            )
                            seed = gr.Number(
                                value=-1, label="Seed（-1 为随机）", precision=0
                            )
                        run = gr.Button("生成", variant="primary")
                    with gr.Column(scale=4, min_width=360):
                        output = gr.Image(label="输出", type="pil")

                run.click(
                    generate_image,
                    inputs=[prompt, width, height, steps, guidance, seed],
                    outputs=[output],
                )

            with gr.TabItem("编辑图片"):
                with gr.Row():
                    with gr.Column(scale=3, min_width=360):
                        base_image = gr.Image(label="基础图片", type="pil")
                        extra_images = gr.File(
                            label="额外参考图片（可选，多选）",
                            file_count="multiple",
                            type="filepath",
                        )
                        prompt = gr.Textbox(
                            label="编辑提示词",
                            lines=3,
                            placeholder="描述你想修改的内容",
                        )
                        use_input_size = gr.Checkbox(value=True, label="使用输入尺寸")
                        with gr.Row():
                            width = gr.Slider(
                                512, 2048, value=1024, step=64, label="宽度"
                            )
                            height = gr.Slider(
                                512, 2048, value=1024, step=64, label="高度"
                            )
                        with gr.Row():
                            steps = gr.Slider(1, 12, value=4, step=1, label="步数")
                            guidance = gr.Slider(
                                0.0, 10.0, value=1.0, step=0.1, label="Guidance"
                            )
                            seed = gr.Number(
                                value=-1, label="Seed（-1 为随机）", precision=0
                            )
                        run = gr.Button("编辑", variant="primary")
                    with gr.Column(scale=4, min_width=360):
                        output = gr.Image(label="输出", type="pil")

                run.click(
                    edit_image,
                    inputs=[
                        base_image,
                        extra_images,
                        prompt,
                        use_input_size,
                        width,
                        height,
                        steps,
                        guidance,
                        seed,
                    ],
                    outputs=[output],
                )

    return demo


if __name__ == "__main__":
    # Preload model at startup to avoid lazy initialization on first request.
    _load_pipe()
    demo = build_ui()
    demo.queue().launch()
