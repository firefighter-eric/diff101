import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
WORKFLOW_PATH = os.getenv("COMFYUI_WORKFLOW")
OUT_DIR = os.getenv("COMFYUI_OUT_DIR", "./comfyui_outputs")

TEXT_NODES = {
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "CLIPTextEncodeFlux",
    "CLIPTextEncodeFluxSchnell",
    "CLIPTextEncodeFluxL",
}
SAMPLER_NODES = {
    "KSampler",
    "KSamplerAdvanced",
    "KSamplerFlux",
    "FluxSampler",
    "SamplerCustom",
}
LATENT_NODES = {
    "EmptyLatentImage",
    "LatentImage",
    "EmptyLatentImageFlux",
    "FluxLatent",
}
GUIDANCE_KEYS = ("cfg", "guidance", "guidance_scale")


def _parse_id_list(value: Optional[str]) -> Optional[set]:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _request_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{COMFYUI_URL}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_bytes(path: str) -> bytes:
    url = f"{COMFYUI_URL}{path}"
    with urllib.request.urlopen(url) as response:
        return response.read()


def load_workflow(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        raise ValueError(
            "COMFYUI_WORKFLOW is not set. Export a workflow JSON from ComfyUI "
            "and set COMFYUI_WORKFLOW to its path."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Workflow JSON must be a dict of node_id -> node")
    return data


def update_workflow(
    workflow: Dict[str, Any],
    prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    guidance: float,
) -> Dict[str, Any]:
    text_ids = _parse_id_list(os.getenv("COMFYUI_TEXT_NODE_IDS"))
    sampler_ids = _parse_id_list(os.getenv("COMFYUI_SAMPLER_NODE_IDS"))
    size_ids = _parse_id_list(os.getenv("COMFYUI_SIZE_NODE_IDS"))

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        if class_type in TEXT_NODES and (text_ids is None or node_id in text_ids):
            if "text" in inputs:
                inputs["text"] = prompt

        if class_type in SAMPLER_NODES and (sampler_ids is None or node_id in sampler_ids):
            if "seed" in inputs:
                inputs["seed"] = seed
            if "steps" in inputs:
                inputs["steps"] = steps
            for key in GUIDANCE_KEYS:
                if key in inputs:
                    inputs[key] = guidance

        if class_type in LATENT_NODES and (size_ids is None or node_id in size_ids):
            if "width" in inputs and "height" in inputs:
                inputs["width"] = width
                inputs["height"] = height

    return workflow


def submit_workflow(workflow: Dict[str, Any]) -> str:
    response = _request_json("POST", "/prompt", {"prompt": workflow})
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Unexpected /prompt response: {response}")
    return prompt_id


def wait_for_history(prompt_id: str, timeout_s: int = 300, poll_s: float = 1.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        history = _request_json("GET", f"/history/{prompt_id}")
        if history:
            item = history.get(prompt_id) or next(iter(history.values()))
            outputs = item.get("outputs") if isinstance(item, dict) else None
            if outputs:
                return item
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for prompt_id {prompt_id}")


def _iter_images(outputs: Dict[str, Any]) -> Iterable[Tuple[str, str, str]]:
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for image in output.get("images", []) or []:
            filename = image.get("filename")
            if not filename:
                continue
            subfolder = image.get("subfolder", "")
            image_type = image.get("type", "output")
            yield filename, subfolder, image_type


def download_images(result: Dict[str, Any], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    outputs = result.get("outputs", {})
    saved: List[str] = []

    for filename, subfolder, image_type in _iter_images(outputs):
        query = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": image_type}
        )
        data = _request_bytes(f"/view?{query}")
        target_dir = os.path.join(out_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)
        with open(target_path, "wb") as f:
            f.write(data)
        saved.append(target_path)

    return saved


def run_demo() -> None:
    prompt = os.getenv("PROMPT", "A cat holding a sign that says hello world")
    seed = int(os.getenv("SEED", "0"))
    steps = int(os.getenv("STEPS", "4"))
    width = int(os.getenv("WIDTH", "512"))
    height = int(os.getenv("HEIGHT", "512"))
    guidance = float(os.getenv("GUIDANCE", "1.0"))

    workflow = load_workflow(WORKFLOW_PATH)
    workflow = update_workflow(workflow, prompt, seed, steps, width, height, guidance)

    prompt_id = submit_workflow(workflow)
    result = wait_for_history(prompt_id)
    saved = download_images(result, OUT_DIR)

    if saved:
        print("Saved:")
        for path in saved:
            print(path)
    else:
        print("No images found in ComfyUI outputs.")


if __name__ == "__main__":
    run_demo()
