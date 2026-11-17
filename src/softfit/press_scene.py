"""The MuJoCo model: a volumetric FEM sponge on a table, pressed from above by a rigid indenter.

The indenter is a mocap box of the measured width at the measured x (or a flat plate over the whole top). Its
height follows the measured dent while the hand presses and is lifted away at each release, so the sponge's
spring-back is simulated, not replayed.

Material model: MuJoCo flexcomp `elasticity` (linear FEM) with
    young  E   [Pa]   stiffness
    poisson nu [-]    lateral bulge
    damping tau [s]   Rayleigh stiffness-proportional damping, D = tau * K
Frame: z up, table at z = 0, sponge centred on the origin, indenter moves along -z.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass
class PressParams:
    size: tuple = (0.135, 0.080, 0.035)    # sponge full extents x, y, z(height) [m]
    count: tuple = (5, 6, 4)                # flex grid vertices per axis
    dof: str = "full"                       # full | quadratic | trilinear
    young: float = 3.0e4                    # Pa
    poisson: float = 0.25
    damping: float = 0.02                   # s
    density: float = 30.0                   # kg/m^3
    friction: float = 0.8                   # indenter/table <-> foam
    contact_solref: tuple = (0.002, 1.0)
    # hard contact: the flex vertices are very light, so the default solimp lets the indenter sink ~1-2 mm
    # into the foam; this keeps it < 0.2 mm, so the indenter's height = the dent depth
    contact_solimp: tuple = (0.99, 0.999, 0.0002)
    plate_margin: float = 0.015             # flat plate only: overhang beyond the sponge on every side [m]
    plate_size: tuple | None = None         # (x, y) full size of the indenter [m]; None = flat plate over the top
    plate_x: float = 0.0                    # x of the indenter's centre [m] (off-centre presses)
    plate_thick: float = 0.004
    radius: float = 0.001                   # flex vertex collision radius
    timestep: float = 5e-4
    # render camera; scripts/replay_trial.py puts it at the real camera's pose
    cam_pos: tuple = (0.10, -0.22, 0.11)
    cam_target: tuple = (0.0, 0.0, 0.02)
    cam_fovy: float = 58.0

    def to_dict(self):
        return asdict(self)


def fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def look_at_xyaxes(pos, target, up=(0, 0, 1)):
    """MuJoCo camera xyaxes (x right, y up; camera looks along -z) for a camera at pos aimed at target."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return np.concatenate([x, y])


def build_xml(p: PressParams) -> str:
    sx, sy, sz = p.size
    nx, ny, nz = p.count
    r = p.radius
    # vertex grid spans size - 2r so the collision surface (vertices + radius) matches the real extents
    sp = ((sx - 2 * r) / (nx - 1), (sy - 2 * r) / (ny - 1), (sz - 2 * r) / (nz - 1))
    mass = p.density * sx * sy * sz
    hx, hy, hz = sx / 2 + p.plate_margin, sy / 2 + p.plate_margin, p.plate_thick / 2
    if p.plate_size is not None:
        hx, hy = p.plate_size[0] / 2, p.plate_size[1] / 2
    fr = f"{p.friction} 0.005 0.0001"
    sr = (f'solref="{p.contact_solref[0]} {p.contact_solref[1]}" '
          f'solimp="{" ".join(map(str, p.contact_solimp))}"')
    return f"""
<mujoco model="foam_press">
  <compiler autolimits="true"/>
  <option timestep="{p.timestep}" solver="Newton" tolerance="1e-8" integrator="discrete"
          cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
  <size memory="64M"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient=".35 .35 .35" diffuse=".6 .6 .6"/>
    <rgba bv="0 0 0 0" bvactive="0 0 0 0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .87 .9" rgb2=".75 .78 .82" width="256" height="256"/>
    <material name="table" texture="grid" texrepeat="20 20"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.3 0.6" dir="-0.3 0.5 -1" directional="true"/>
    <geom name="table" type="plane" size="0.5 0.5 0.01" material="table" friction="{fr}" {sr}/>
    <camera name="depthcam" pos="{fmt(p.cam_pos)}" xyaxes="{fmt(look_at_xyaxes(p.cam_pos, p.cam_target))}" fovy="{p.cam_fovy}"/>
    <body name="plate" mocap="true" pos="{p.plate_x} 0 {sz + hz + 0.010}">
      <geom name="plate" type="box" size="{hx} {hy} {hz}" rgba=".2 .5 .9 .6" friction="{fr}" {sr}/>
    </body>
    <flexcomp name="foam" type="grid" dim="3" dof="{p.dof}" count="{nx} {ny} {nz}"
              spacing="{sp[0]} {sp[1]} {sp[2]}" pos="0 0 {sz / 2}" radius="{r}" mass="{mass}"
              rgba=".85 .75 .2 1">
      <elasticity young="{p.young}" poisson="{p.poisson}" damping="{p.damping}"/>
      <contact selfcollide="none" condim="3" friction="{fr}" {sr}/>
    </flexcomp>
  </worldbody>
</mujoco>
"""


class PressSim:
    """Prescribe the indenter's lower-face height h(t); step; read the deformed sponge."""

    def __init__(self, p: PressParams):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(build_xml(p))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.plate_mocap = m.body("plate").mocapid[0]
        self.plate_geom = m.geom("plate").id    # replay_trial colours it
        self.flex_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "foam")
        a, n = m.flex_vertadr[self.flex_id], m.flex_vertnum[self.flex_id]
        self._vsl = slice(a, a + n)
        sa, sn = m.flex_shelldataadr[self.flex_id], m.flex_shellnum[self.flex_id]
        self.shell = m.flex_shell[sa:sa + 3 * sn].reshape(sn, 3)   # surface triangles (local vert ids)
        mujoco.mj_forward(m, self.data)
        v = self.vertices()
        self.rest = v.copy()
        z = v[:, 2]
        self.top_ids = np.where(z > z.max() - 1e-6)[0]
        hw = self.p.plate_size[0] / 2 if self.p.plate_size is not None else np.inf
        under = np.abs(v[self.top_ids, 0] - self.p.plate_x) < hw
        self.press_ids = self.top_ids[under] if under.any() else self.top_ids   # top vertices under the indenter

    # ---- inputs ---------------------------------------------------------------------------
    def set_plate(self, h_bottom: float):
        """h_bottom = height of the indenter's lower face above the table [m]."""
        self.data.mocap_pos[self.plate_mocap] = (self.p.plate_x, 0.0, h_bottom + self.p.plate_thick / 2)

    # ---- readouts -------------------------------------------------------------------------
    def vertices(self) -> np.ndarray:
        return self.data.flexvert_xpos[self._vsl].copy()

    def surface_points(self, verts: np.ndarray | None = None, per_tri: int = 6, seed: int = 0) -> np.ndarray:
        """Dense samples of the deformed outer surface (fixed barycentric coords per triangle)."""
        v = self.vertices() if verts is None else verts
        if not hasattr(self, "_bary") or self._bary.shape[1] != per_tri:
            rng = np.random.default_rng(seed)
            u = rng.random((len(self.shell), per_tri, 2))
            flip = u.sum(-1) > 1
            u[flip] = 1 - u[flip]
            self._bary = np.concatenate([1 - u.sum(-1, keepdims=True), u], -1)   # (T,k,3)
        tri = v[self.shell]                                                       # (T,3,3)
        pts = np.einsum("tkj,tjd->tkd", self._bary, tri)                          # (T,k,3)
        # the visible/collision surface sits one vertex radius outside the shell triangles
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        n *= np.sign(np.einsum("td,td->t", n, tri.mean(1) - v.mean(0)))[:, None]  # outward (convex-ish body)
        return (pts + self.p.radius * n[:, None, :]).reshape(-1, 3)

    def press_top(self) -> float:
        """Mean height of the sponge's top surface under the indenter [m]."""
        return float(self.vertices()[self.press_ids, 2].mean()) + self.p.radius

    def unstable(self) -> bool:
        d = self.data
        return (not np.isfinite(d.qacc).all()) or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0

    # ---- rollout --------------------------------------------------------------------------
    def run(self, t: np.ndarray, h: np.ndarray, sample_t: np.ndarray, keep_verts=True, on_sample=None):
        """Drive the indenter height h(t) (linear interp), record the sponge at sample_t.

        Returns dict of arrays, plus 'ok' = False if the sim blew up (outputs then truncated)."""
        m, d = self.model, self.data
        out = {"t": [], "plate_h": [], "press_top": [], "verts": []}
        k = 0
        ok = True
        while k < len(sample_t):
            self.set_plate(float(np.interp(d.time, t, h)))
            mujoco.mj_step(m, d)
            if self.unstable():
                ok = False
                break
            while k < len(sample_t) and d.time >= sample_t[k] - 1e-9:
                out["t"].append(d.time)
                out["plate_h"].append(float(np.interp(d.time, t, h)))
                out["press_top"].append(self.press_top())
                if keep_verts:
                    out["verts"].append(self.vertices())
                if on_sample is not None:
                    on_sample(self)
                k += 1
        res = {key: np.asarray(v) for key, v in out.items()}
        res["ok"] = ok
        return res

