"""Teacher-only constraint on actual horizontal heading rate in MotrixSim."""
import math
import numpy as np
from numba import njit

from motrix_env_core.config import configclass
from motrix_env_core.manager import RewardTerm, RewardTermCfg
from motrix_env_core.numba.manager.context import BuildContext, ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim import LinkAngularVelocityQuery, LinkQuaternionQuery
from motrix_envs.locomotion.humanoid.cfg import WalkRewardsCfg


@njit
def horizontal_heading_rate(q, world_w):
    """Derivative of atan2 of world XY projection of local +X (xyzw quaternion)."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + z * w)
    fz = 2.0 * (x * z - y * w)
    denominator = fx * fx + fy * fy
    # Horizontal heading is undefined when local +X points vertically.
    # Bound that singularity; the normal upright task is far from this case.
    denominator = max(denominator, 1.0e-6)
    return world_w[2] - fz * (world_w[0] * fx + world_w[1] * fy) / denominator


@dispatch
def heading_rate_bounded_cost(_ctx: ManagerContext, q: np.ndarray, world_w: np.ndarray,
                             sigma: np.float32) -> float:
    x, y, z, w = q[0], q[1], q[2], q[3]
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + z * w)
    if fx * fx + fy * fy < 1.0e-4:
        # Undefined horizontal heading receives the full bounded cost.
        return 1.0
    rate = horizontal_heading_rate(q, world_w)
    # Positive bounded cost, multiplied by a NEGATIVE config weight once.
    # This teacher profile trains only zero-turn straight walking commands.
    return 1.0 - math.exp(-rate * rate / sigma)


@configclass(kw_only=True)
class HeadingRatePenaltyCfg(RewardTermCfg):
    body: str = "robot"
    sigma: float = 0.015

    def __call__(self, ctx: BuildContext) -> RewardTerm:
        base = ctx.model.bodies[self.body].base_link_name
        return RewardTerm(heading_rate_bounded_cost,
                          LinkQuaternionQuery(link=base), LinkAngularVelocityQuery(link=base),
                          np.float32(self.sigma))


@configclass
class HeadingConstrainedRewardsCfg(WalkRewardsCfg):
    penalty_heading_rate: HeadingRatePenaltyCfg = HeadingRatePenaltyCfg(weight=-15.0)
