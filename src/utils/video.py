from typing import List

import imageio
import numpy as np


def write_video(frames: List[np.ndarray], output_path: str, fps: int = 10) -> None:
    imageio.mimsave(output_path, frames, fps=fps)
