import torch
from diffusers import Flux2KleinPipeline
import time

device = "cuda"
dtype = torch.bfloat16

use_fp8 = False
model_path = "black-forest-labs/FLUX.2-klein-4B"
fp8_model_path = "black-forest-labs/FLUX.2-klein-4b-fp8"
model_path = fp8_model_path if use_fp8 else model_path


torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

pipeline:Flux2KleinPipeline = Flux2KleinPipeline.from_pretrained(
    model_path,
    torch_dtype=dtype,
    # local_files_only=True,
)
pipeline = pipeline.to(device)
pipeline.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
# pipeline.transformer = torch.compile(
#     pipeline.transformer,
#     # mode="max-autotune",
#     fullgraph=True,
#     dynamic=True,
# )
# pipeline.vae.decode = torch.compile(
#     pipeline.vae.decode,
#     mode="max-autotune",
#     fullgraph=True
# )


def generate() -> None:
    width = 2048
    height = 1152
    prompt = "A cat holding a sign that says hello world"
    start_time = time.time()
    image = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device=device).manual_seed(0),
    ).images[0]
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Generation took {elapsed:.2f} seconds")
    width, height = image.size
    print(f"Generated image of size ({width}, {height}) for prompt: {prompt}")
    # image.save("flux-klein.png")


# Warm up
generate()

# Actual run
generate()
