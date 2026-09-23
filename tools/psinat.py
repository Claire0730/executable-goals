"""psinat.py <task> <seed> [repo] -- natural-axis psi bank (k1 = teacher cone axis, theta 0;
h = trace carry height). Natural-axis conversion of the psi bank written by psi_bank.py."""
import os
import sys

import numpy as np
TAG_MIX4, TAG_HEAD, TAG_GMAP = (os.environ.get("TAG_MIX4", "mix4_realcam_n2400"), os.environ.get("TAG_HEAD", "mix5_t2k_n3000"), os.environ.get("TAG_GMAP", "mix5_t2k_gmap"))

t, sd = sys.argv[1], sys.argv[2]
REPO = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("EG_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRIOR = {"pickcube": [0, 0, -1], "liftpeg": [0, 0, -1],
         "peginsert": [-0.28, -0.56, -0.78], "stack": [0, 0, -1]}
z = np.load(f"{REPO}/results/{t}_psi_{TAG_MIX4}_{sd}.npz", allow_pickle=True)
a = np.array(PRIOR[t], np.float32); a /= np.linalg.norm(a)
k1 = np.tile(a, (len(z["h"]), 1))
np.savez_compressed(f"{REPO}/results/{t}_psinat_{TAG_MIX4}_{sd}.npz",
                    k1=k1, h=z["h"], ok=z["ok"],
                    prov="natural-axis psi: k1 = teacher cone axis (theta 0), h = trace carry height")
print(t, sd, "psinat written")
