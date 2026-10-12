"""Round 25 RRTWIN edits (plan8 / design8 / step8): a timing twin whose norms do not read their residual input from DDR."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "tools"))
from shm_paths import SCRATCH  # noqa: E402

D = SCRATCH / "npu-tapnext"


def rep(f, pairs):
    p = D / f
    s = p.read_text()
    for a, b in pairs:
        assert s.count(a) == 1, (f, a[:100])
        s = s.replace(a, b)
    p.write_text(s)


rep("plan8.py", [
    ("""ZDROP = os.environ.get("ZDROP") == "1"  # round 24: SRES SSMs drain h only (ew8 -DZDROP)
""",
     """ZDROP = os.environ.get("ZDROP") == "1"  # round 24: SRES SSMs drain h only (ew8 -DZDROP)
RRTWIN = os.environ.get("RRTWIN") == "1"  # round 25 timing twin (outputs wrong): norms skip their residual input (a norm's output)
"""),
    ("""        self.writer = {}  # region key -> name of the last gemm/attn op that wrote it (EW ops are synchronous)
""",
     """        self.writer = {}  # region key -> name of the last gemm/attn op that wrote it (EW ops are synchronous)
        self.norm_dsts = set()  # (round 25 RRTWIN: the bases norms write = the residual inputs of later norms)
"""),
    ("""        srcs = sorted(srcs, key=lambda s: s["base"])  # the sum is commutative: stream order = address order
        n_in = len(srcs)
""",
     """        srcs = sorted(srcs, key=lambda s: s["base"])  # the sum is commutative: stream order = address order
        if RRTWIN and len(srcs) > 1:  # round 25 timing twin: the residual (an earlier norm's X') not read
            keep = [s for s in srcs if s["base"] not in self.norm_dsts]
            assert len(keep) == len(srcs) - 1, (name, [s["base"] for s in srcs])
            srcs = keep
        n_in = len(srcs)
"""),
    ("""        if dst is None:
            dst = self.q16(name + "_out")
        if params_bytes is None:
""",
     """        if dst is None:
            dst = self.q16(name + "_out")
        self.norm_dsts.add(dst["base"])
        if params_bytes is None:
"""),
])
rep("step8.py", [
    ("""        plan8.ZDROP = bool(bj.get("zdrop", False))  # round 24 ZDROP: SRES SSMs drain h only
""",
     """        plan8.ZDROP = bool(bj.get("zdrop", False))  # round 24 ZDROP: SRES SSMs drain h only
        plan8.RRTWIN = bool(bj.get("rrtwin", False))  # round 25 timing twin: norms without their residual input
"""),
])
rep("design8.py", [
    ("""yxh=os.environ.get("YXH") == "1", zdrop=os.environ.get("ZDROP") == "1")))""",
     """yxh=os.environ.get("YXH") == "1", zdrop=os.environ.get("ZDROP") == "1",
                                                     rrtwin=os.environ.get("RRTWIN") == "1")))"""),
])
print("RRTWIN patched")
