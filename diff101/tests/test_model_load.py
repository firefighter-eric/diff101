import torch
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

model_path = "black-forest-labs/FLUX.2-klein-4B"
pipe = Flux2KleinPipeline.from_pretrained(
    model_path,
    torch_dtype=dtype,
    # local_files_only=True,
)
# pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
image.save("flux-klein.png")
