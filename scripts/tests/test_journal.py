"""CPU tests for the journal-extension protocol and degradations.  python -m pytest scripts/tests/test_journal.py -q"""
import numpy as np
from PIL import Image

from scripts.journal import protocol as P
from src import degradations as D


def test_auroc_and_matched_f1():
    gold = np.array([1, 1, 0, 0, 1, 0], bool)
    s = np.array([3.0, 2.0, -0.5, -1.0, 0.5, 0.2])
    assert P.auroc(s, gold) == 1.0
    # top-3 by score are exactly the 3 positives
    assert P.matched_f1(s, 0.5, gold) == 1.0
    assert P.matched_f1(s, 0.0, gold) == 0.0


def test_cluster_boot_keeps_images_whole():
    imgs = np.array(["a", "a", "b", "b", "b", "c"])
    rng = np.random.default_rng(0)
    for idx in P.cluster_boot(imgs, 20, rng):
        picked = imgs[idx]
        for im in set(picked):
            assert (picked == im).sum() % (imgs == im).sum() == 0


def test_vcd_margin_mask_and_decision_score():
    class R:
        cols = dict(yes=np.array([0.0, 0.0, 5.0]), no=np.array([0.0, -10.0, 0.0]), top=np.array([0.0, 0.0, 5.0]),
                    cd_yes=np.array([1.0, 0.0, 0.0]), cd_no=np.array([0.0, 0.0, 0.0]))

        def col(self, k):
            return self.cols[k].copy()
    m = P.vcd_margin(R())
    assert m[0] == -1.0                       # 2*(0-0) - (1-0)
    assert np.isposinf(m[1])                  # No below log(0.1)+top -> forced Yes
    assert np.isposinf(m[2])                  # No = 0 < log .1 + 5
    s = P.decision_score(m, np.array([0.0, 10.0, 5.0]))
    assert s[1] > s[2] > s[0]


def test_sensor_degradations_deterministic_and_bounded():
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    for kind in D.SENSOR_LEVELS:
        a = np.asarray(D.degrade(img, kind, 2, "x.jpg"))
        b = np.asarray(D.degrade(img, kind, 2, "x.jpg"))
        assert a.shape == (48, 64, 3) and a.dtype == np.uint8 and (a == b).all()


def test_motion_kernel_normalised():
    for L in (5, 15.5, 40):
        k = D.motion_kernel(L, 33.0)
        assert abs(k.sum() - 1) < 1e-9 and k.shape[0] % 2 == 1


def test_lowlight_mean_preserved_in_linear_light():
    img = Image.fromarray(np.full((64, 64, 3), 128, np.uint8))
    out = np.asarray(D.low_light(img, 1e6, 0)).astype(float)
    assert abs(out.mean() - 128) < 1.0          # huge photon count -> essentially noise-free
