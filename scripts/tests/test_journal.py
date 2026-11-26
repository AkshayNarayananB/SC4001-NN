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


def test_shard_merge_and_task_ids(tmp_path, monkeypatch):
    import json
    import numpy as np
    monkeypatch.setattr(P, "JOURNAL_ROOT", tmp_path)
    base = tmp_path / "pope" / "m" / "clean_s0"
    rows_all, hid_all = [], []
    for i in range(2):
        d = base / f"shard{i}of2"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"hidden_layers": [1, 2]}))
        rows = [{"image": f"im{i}{j}", "text": "q", "m": float(j)} for j in range(3)]
        hid = np.random.default_rng(i).normal(size=(3, 2, 4)).astype(np.float16)
        (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        hid.tofile(d / "_hidden.f16")
        rows_all += rows
        hid_all.append(hid)
    import pytest
    with pytest.raises(FileNotFoundError):
        P.Run("pope", "m", "clean_s0")          # shards not DONE yet
    for i in range(2):
        (base / f"shard{i}of2" / "DONE").touch()
    r = P.Run("pope", "m", "clean_s0")
    assert [x["image"] for x in r.rows] == [x["image"] for x in rows_all]
    assert (r.hidden() == np.concatenate(hid_all)).all()

    import os
    os.environ.setdefault("VCD_OUTPUT_ROOT", str(tmp_path))
    from scripts.journal.aspire import pool_worker as W
    t = ("m", "exdark", "clean_s0", "--vcd --shard 3 --nshards 8")
    assert W.task_id(t) == "exdark__m__clean_s0__shard3of8"
    assert str(W.run_dir(t)).endswith("exdark/m/clean_s0/shard3of8")
    assert str(W.run_dir(("m", "pope", "noise_s4", "--views blur1 blur2 --batch-views"))).endswith("pope/m/noise_s4__blur1")


def test_capture_task_ids(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("VCD_OUTPUT_ROOT", str(tmp_path))
    from scripts.journal.aspire import pool_worker as W
    t = ("m", "pope_fresh", "@capture", "--grid full --limit-images 500 --shard 2 --nshards 16")
    assert W.task_id(t) == "pope_fresh__m__capture_full_lim500__shard2of16"
    assert str(W.run_dir(t)).endswith("pope_fresh/m/capture_full_lim500/shard2of16")
    t = ("m", "pope", "@capture", "--grid full --caption --limit-images 200")
    assert W.task_id(t) == "pope__m__caption_full_lim200"


def test_capture_prefix_consistency():
    from src import capture as C
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (40, 56, 3), dtype=np.uint8))
    a, _ = C.frames(img, "x.jpg", 0.3, 0.5, 8.0, 4)
    b, _ = C.frames(img, "x.jpg", 0.3, 0.5, 8.0, 2)
    assert all((x == y).all() for x, y in zip(a[:2], b))
    hi = np.asarray(C.capture(img, "x.jpg", C.Capture(1e6, 0.0, 33.3)), float)
    assert np.abs(hi - np.asarray(img, float)).mean() < 1.0
