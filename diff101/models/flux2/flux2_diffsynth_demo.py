from datetime import datetime

import torch
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig




model_id = "black-forest-labs/FLUX.2-klein-4B"
device = "cuda"
steps = 4
seed = 0
edit_seed = 1
rand_device = "cuda"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = f"data/flux2/diffsynth-FLUX.2-klein-4B-{timestamp}.png"

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

# vram_config = {}

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        ModelConfig(model_id=model_id, origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id=model_id, origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id=model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(
        model_id=model_id,
        origin_file_pattern="tokenizer/",
    ),
)

def generate() -> None:
    prompt ="A cat holding a sign that says hello world"
    # prompt = 'A dog wearing a spacesuit, digital art'
    image = pipe(
        prompt,
        seed=seed,
        height=1024,
        width=1024,
        rand_device=device,
        num_inference_steps=steps,
        # cfg_scale=4.0,
    )
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
    generate()
