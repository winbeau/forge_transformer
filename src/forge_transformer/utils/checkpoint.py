import torch


def save_checkpoint(model, optimizer, iteration: int, out_path: str):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(payload, out_path)


def load_checkpoint(model, optimizer, src_path: str) -> int:
    payload = torch.load(src_path, map_location="cpu", weights_only=True)  # 确保即使保存用 GPU，CPU 环境依然可加载
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    iteration = payload.get("iteration", 0)  # 默认 0
    return iteration
