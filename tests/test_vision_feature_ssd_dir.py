from pathlib import Path
from types import SimpleNamespace

from omlx.engine.vlm import vision_feature_ssd_dir


def test_hot_cache_only_keeps_vision_features_in_memory(tmp_path):
    config = SimpleNamespace(paged_ssd_cache_dir=str(tmp_path), hot_cache_only=True)
    assert vision_feature_ssd_dir(config) is None


def test_ssd_cache_stores_vision_features_beside_kv_blocks(tmp_path):
    config = SimpleNamespace(paged_ssd_cache_dir=str(tmp_path), hot_cache_only=False)
    assert vision_feature_ssd_dir(config) == Path(tmp_path) / "vision_features"


def test_no_ssd_cache_dir_or_config_means_memory_only():
    assert vision_feature_ssd_dir(SimpleNamespace(paged_ssd_cache_dir=None)) is None
    assert vision_feature_ssd_dir(None) is None
