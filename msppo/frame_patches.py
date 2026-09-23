"""Apply the framework-teacher (psi) env patches for a set of tasks in ONE process -- what a multi-task distillation or
evaluation needs when its teachers/students carry the psi block (k1 approach direction + h carry height).
stack -> msppo.patch_stack_frame (MSPPO_STACK_FRAME=1); pickcube/liftpeg/peginsert -> msppo.patch_frame.maybe_patch(task).
Reward knobs are irrelevant to a frozen teacher; what matters is the psi PRIOR (cone axis / h range), which the patches take
from their per-task defaults (peg cone axis = the measured natural approach used by pi_v9_frame4_s0)."""
import os


def apply(tasks):
    done = []
    for t in tasks:
        if t == "stack":
            os.environ["MSPPO_STACK_FRAME"] = "1"
            from msppo.patch_stack_frame import maybe_patch
            assert maybe_patch(), "stack frame patch failed"
        else:
            from msppo.patch_frame import maybe_patch
            assert maybe_patch(t), f"frame patch failed for {t}"
        done.append(t)
    return done
