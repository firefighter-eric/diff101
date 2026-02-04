from datetime import datetime

import torch
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import time


model_id = "black-forest-labs/FLUX.2-klein-4B"
device = "cuda"
steps = 6
seed = 0
edit_seed = 1
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = f"data/flux2/diffsynth-FLUX.2-klein-4B-{timestamp}.png"

vram_config = {}

# vram_config = {
#     "offload_dtype": "disk",
#     "offload_device": "disk",
#     "onload_dtype": torch.float8_e4m3fn,
#     "onload_device": "cpu",
#     "preparing_dtype": torch.float8_e4m3fn,
#     "preparing_device": "cuda",
#     "computation_dtype": torch.bfloat16,
#     "computation_device": "cuda",
# }

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        ModelConfig(
            model_id=model_id,
            origin_file_pattern="text_encoder/*.safetensors",
            **vram_config,
        ),
        ModelConfig(
            model_id=model_id,
            origin_file_pattern="transformer/*.safetensors",
            **vram_config,
        ),
        ModelConfig(
            model_id=model_id,
            origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            **vram_config,
        ),
    ],
    tokenizer_config=ModelConfig(
        model_id=model_id,
        origin_file_pattern="tokenizer/",
    ),
)


def generate() -> None:
    prompt = "A cat holding a sign that says hello world"
    prompt = "A fantasy style illustration of a heroic cat warrior standing on a hilltop, holding a sign that says 'hello world', vibrant colors, detailed background, dramatic lighting"
    height = 1152
    width = 2048
    start_time = time.time()
    image = pipe(
        prompt,
        seed=seed,
        height=height,
        width=width,
        rand_device=device,
        num_inference_steps=steps,
        cfg_scale=1.0,
        # embedded_guidance=1.0,
    )
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Generation took {elapsed:.2f} seconds")
    width, height = image.size
    print(f"Generated image of size ({width}, {height}) for prompt: {prompt}")
    image.save(output_path)

    # edit_prompt = "change the color of the clothes to red"
    # edited = pipe(
    #     edit_prompt,
    #     # edit_image=[image],
    #     seed=edit_seed,
    #     rand_device=rand_device,
    #     num_inference_steps=steps,
    # )
    # edited.save(edit_output_path)


if __name__ == "__main__":
    # Warm up
    generate()

    for _ in range(4):
        generate()
