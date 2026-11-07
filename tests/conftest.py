import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth import make_dataset  # noqa: E402

from aidetect.config import Config  # noqa: E402
from aidetect.dataset import discover, scan_dataset  # noqa: E402
from aidetect.trainer import train  # noqa: E402


def fast_cfg(tmp: Path, **kw) -> Config:
    base = dict(cache_dir=str(tmp / "cache"), n_jobs=2, cv_folds=3, crops_per_image=4, heatmap_max_crops=9,
                model_params={"extra_trees": {"n_estimators": 60}, "hgb": {"max_iter": 60}})
    base.update(kw)
    return Config.from_dict(base)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("data"), n_per_class=36, seed=7)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, data_dir):
    tmp = tmp_path_factory.mktemp("model")
    cfg = fast_cfg(tmp)
    scan = scan_dataset(discover(cfg, root=data_dir), cfg, show_progress=False)
    summary = train(scan.samples, cfg, tmp / "m", audit=scan.audit)
    return tmp / "m", cfg, summary
