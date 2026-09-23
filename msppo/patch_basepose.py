"""patch_basepose.py -- MSPPO_BASE_POSE="x,y,z" moves the Panda base in every TableSceneBuilder-based env
(default sim: -0.615,0,0). Eval-side probe for the REAL lab geometry: robot base 7.5 cm BELOW the
table top and the camera-aligned sim frame putting the base at x=-0.4495 -> e.g. "-0.4495,0,-0.075".
Only the robot moves; table, cubes, camera, goals stay at the training positions, so the student sees the
same world with a different qpos<->tcp coupling (its proprioception is what the probe measures)."""
import os
_APPLIED = None


def maybe_patch():
    global _APPLIED
    s = os.environ.get("MSPPO_BASE_POSE", "")
    if not s:
        return False
    if _APPLIED == s:          # idempotent: kp_teacher.make_env calls this for every env it builds
        return True
    _APPLIED = s
    xyz = [float(v) for v in s.split(",")]
    import sapien
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    orig = TableSceneBuilder.initialize

    def initialize(self, env_idx, *a, **k):
        orig(self, env_idx, *a, **k)
        self.env.agent.robot.set_pose(sapien.Pose(xyz))
    TableSceneBuilder.initialize = initialize
    print(f"[patch_basepose] robot base -> {xyz}")
    return True
