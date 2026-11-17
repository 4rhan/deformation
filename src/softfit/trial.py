"""One processed recording (written by scripts/rgbd_to_pcd.py), stored as a pickle-free .npz.

Fields (SI units, sponge frame: z up, table at z = 0, sponge centred on x = 0):
  t          (F,)    frame times [s]
  plate_h    (F,)    height of the indenter's lower face [m]: the measured dent while pressed, parked above
                     the sponge after each release
  pts        (M,3)   all frames' front-face point clouds concatenated
  offsets    (F+1,)  frame f is pts[offsets[f]:offsets[f+1]]
  size       (3,)    sponge extents x, y, z [m] (z = measured rest height)
  plate_size (2,)    x, y size of the indenter [m] (NaN = flat plate over the whole top)
  plate_x    ()      x of the indenter's centre [m] (0 = middle)
  name               clip name
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trial:
    name: str
    t: np.ndarray
    plate_h: np.ndarray
    clouds: list
    size: np.ndarray
    plate_size: np.ndarray | None = None
    plate_x: float = 0.0

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.concatenate([[0], np.cumsum([len(c) for c in self.clouds])])
        pts = np.concatenate(self.clouds).astype(np.float32) if self.clouds else np.zeros((0, 3), np.float32)
        np.savez_compressed(path, t=self.t, plate_h=self.plate_h, pts=pts, offsets=offsets,
                            size=np.asarray(self.size, float), name=np.array(self.name),
                            plate_size=np.full(2, np.nan) if self.plate_size is None else np.asarray(self.plate_size, float),
                            plate_x=np.array(float(self.plate_x)))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        o, p = z["offsets"], z["pts"].astype(np.float64)
        return cls(name=str(z["name"]), t=z["t"], plate_h=z["plate_h"],
                   clouds=[p[o[i]:o[i + 1]] for i in range(len(o) - 1)], size=z["size"],
                   plate_size=z["plate_size"] if np.isfinite(z["plate_size"]).all() else None,
                   plate_x=float(z["plate_x"]))
