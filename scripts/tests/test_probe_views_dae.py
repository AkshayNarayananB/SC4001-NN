"""CPU tests for probe x views x DAE (no model weights or GPU).   python -m pytest scripts/tests -q

The fake run directories mimic scripts.run_layers: one row per unique (image, text) per condition and
view, each view's cache in its own random row order, so a view looked up with the wrong hidx would
lose its signal."""
import json

import numpy as np
import pytest
from PIL import Image

import scripts.analyze_layers as al
import scripts.analyze_probe_views_dae as pv
import scripts.run_layers as rl

N_IMAGES, DIM = 30, 8
SIGNAL = {"orig": 1.0, "blur1": 0.5, "blur2": 0.5, "blur3": 0.5, "hflip": 0.5, "dae": 3.0}
NO_OBJECT = {"random": "dog", "popular": "car", "adversarial": "bird"}


def write_fake(root, views=pv.VIEWS, rng=None):
    rng = rng or np.random.default_rng(0)
    qdir = root / "questions"
    qdir.mkdir(parents=True, exist_ok=True)
    questions = {}
    for split in al.POPE_SPLITS:
        qs = []
        for i in range(N_IMAGES):
            img = f"img{i:02d}.jpg"
            qs.append(dict(question_id=2 * i, image=img, text="Is there a cat in the image?", label="yes"))
            qs.append(dict(question_id=2 * i + 1, image=img, text=f"Is there a {NO_OBJECT[split]} in the image?", label="no"))
        questions[split] = qs
        (qdir / f"coco_pope_{split}.json").write_text("\n".join(json.dumps(q) for q in qs) + "\n")
    unique = sorted({(q["image"], q["text"], q["label"]) for qs in questions.values() for q in qs})
    for kind, level in al.CONDITIONS:
        for v in views:
            run = root / "out" / rl.run_dir_name(kind, level, v)
            run.mkdir(parents=True, exist_ok=True)
            order = rng.permutation(len(unique))              # this view's own cache row order
            hidx = {u[:2]: int(r) for u, r in zip(unique, order)}
            H = rng.normal(size=(len(unique), len(al.HIDDEN_LAYERS), DIM))
            margins = {}
            for (img, text, label), r in zip(unique, order):
                g = 1.0 if label == "yes" else -1.0
                H[r, :, 0] += SIGNAL[v] * g * 0.6
                m = rng.normal(size=32) + 0.5 * g
                m[31] += 3 * g if kind == "clean" else 0.0   # clean layer 32 is the (good) teacher
                margins[(img, text)] = m
            H.astype(np.float16).tofile(run / "_hidden.f16")
            for split, qs in questions.items():
                with open(run / f"{split}.jsonl", "w") as f:
                    for q in qs:
                        m = margins[(q["image"], q["text"])]
                        f.write(json.dumps(dict(question_id=q["question_id"], image=q["image"], label=q["label"],
                                                hidx=hidx[(q["image"], q["text"])], layer_yes=list(m),
                                                layer_no=[0.0] * 32, layer_ent=[1.0] * 32, top1_final=0)) + "\n")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(al, "HIDDEN_DIM", DIM)
    monkeypatch.setattr(al, "N_DEV_IMAGES", 10)
    monkeypatch.setattr(al, "pope_question_file", lambda split: tmp_path / "questions" / f"coco_pope_{split}.json")
    return tmp_path


def setup(fake):
    write_fake(fake)
    cells = al.load_cells()
    dev_set = al.dev_images(cells)
    data = pv.load(cells)
    test = {k: pv.subset(c, ~np.isin(c["image"], list(dev_set))) for k, c in data.items()}
    dev = {k: pv.subset(c, np.isin(c["image"], list(dev_set))) for k, c in data.items()}
    return cells, dev_set, data, dev, test


def test_views_are_aligned_by_question_not_by_row(fake):
    _, dev_set, data, dev, test = setup(fake)
    assert len(data) == 27
    folds = pv.fold_of(dev_set)
    layers, C, target = pv.PROBES["Gold probe L24"]
    S = pv.probe_scores(pv.fit_probe(dev, layers, C, target, folds), layers, sorted({k[:2] for k in data}))
    P = pv.per_view(test, S, folds, oof=False)
    auc = lambda v: np.mean([al.auroc(p[v], test[k]["gold"]) for k, p in P.items()])
    assert auc("dae") > 0.95 and auc("orig") > 0.75      # signal survives each view's own row order
    assert auc("dae") > auc("orig") > auc("blur1")


def test_out_of_fold_scores_never_come_from_a_probe_that_saw_the_image(fake):
    _, dev_set, data, dev, _ = setup(fake)
    folds = pv.fold_of(dev_set)
    layers, C, target = pv.PROBES["Gold probe L24"]
    models = pv.fit_probe(dev, layers, C, target, folds)
    S = pv.probe_scores(models, layers, [("clean", 0)])
    key = ("clean", 0, "random")
    c = dev[key]
    P = pv.per_view({key: c}, S, folds, oof=True)[key]["orig"]
    H = pv.hidden("clean", 0, "orig")
    for i in range(len(c["gold"])):
        f = folds[c["image"][i]]
        expected = models[1 + f].decision_function(pv.features(H, [c["hidx_orig"][i]], layers))[0]
        assert P[i] == pytest.approx(expected)


def test_viewlayer_reference_is_mean_layer20_margin(fake):
    _, _, _, _, test = setup(fake)
    key = ("noise", 4, "popular")
    c = test[key]
    expected = np.mean([c[f"m_{v}"][:, 19] for v in pv.VIEWLAYER_DAE], axis=0)
    np.testing.assert_allclose(pv.margin_mean(test, pv.VIEWLAYER_DAE)[key], expected)
    assert len(pv.VIEWLAYER_DAE) == 6 and pv.VIEWLAYER_DAE[-1] == "dae"


def test_main_end_to_end_picks_the_restored_view(fake):
    write_fake(fake)
    out = fake / "results"
    table = pv.main(["--boot", "20", "--out", str(out)])
    md = (out / "probe_views_dae_table.md").read_text()
    assert "Against ViewLayer+DAE" in md and "Partial" not in md
    assert {"ViewLayer*", "ViewLayer+DAE*", "Gold probe, × ViewLayer+DAE views*"} <= set(table.Method)
    sweep = __import__("pandas").read_csv(out / "probe_views_dae_dev_sweep.csv")
    for probe in pv.PROBES:
        assert "dae" in sweep[sweep.probe == probe].iloc[0]["views"]


def test_missing_view_skips_the_condition(fake, capsys):
    write_fake(fake)
    for split in al.POPE_SPLITS:
        (al.OUTPUT_ROOT / rl.run_dir_name("blur", 2, "dae") / f"{split}.jsonl").unlink()
    data = pv.load(al.load_cells())
    assert len(data) == 24 and not any(k[:2] == ("blur", 2) for k in data)
    assert "blur_s2" in capsys.readouterr().out


def test_dae_view_uses_the_dae(monkeypatch):
    from src.fixes.dae import DAE
    monkeypatch.setattr(rl, "_dae", {"name": "fake", "model": DAE().eval()})   # zero last layer: identity
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (37, 50, 3), dtype=np.uint8))
    np.testing.assert_array_equal(np.asarray(rl.make_view(img, "dae")), np.asarray(img))
    assert "dae" in rl.VIEWS
