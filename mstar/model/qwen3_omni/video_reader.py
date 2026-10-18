"""A bounded replacement for qwen-omni-utils' whole-clip video reader."""
import torch

from mstar.model.media_limits import VideoBudget


def read_video_whole_budgeted(ele: dict) -> tuple[torch.Tensor, dict, float]:
    """Drop-in for qwen-omni-utils' torchvision reader, which ``fetch_video``
    falls back to when torchcodec or decord raises and which decodes the whole
    clip. Same frames (``torchvision.io.read_video`` decodes with PyAV too) and
    the same sampling, but every frame is counted as it is decoded: a header
    can understate the clip, so checking it first would not bound this.

    The original no longer runs at all from torchvision 0.27, which removed
    ``torchvision.io.read_video``."""
    import av
    from qwen_omni_utils.v2_5.vision_process import smart_nframes

    path = ele["video"]
    if path.startswith("file://"):
        path = path[len("file://"):]
    start, end = ele.get("video_start", 0.0), ele.get("video_end")
    budget = VideoBudget()
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for frame in container.decode(stream):
            if frame.time is not None and frame.time < start:
                continue
            if end is not None and frame.time is not None and frame.time > end:
                break
            budget.add(frame.height, frame.width)
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
    if not frames:
        raise ValueError("no video frames could be decoded")
    total = len(frames)
    nframes = smart_nframes(ele, total_frames=total, video_fps=fps)
    idx = torch.linspace(0, total - 1, nframes).round().long()
    video = torch.stack([frames[i] for i in idx.tolist()])
    metadata = dict(fps=fps, frames_indices=idx, total_num_frames=total, video_backend="torchvision")
    return video, metadata, nframes / max(total, 1e-6) * fps
