# SPDX-License-Identifier: Apache-2.0
"""Immutable namespaces prevent cross-checkpoint native prefix-cache reuse."""
import hashlib
import json
from pathlib import Path

CACHE_FIELDS = ('model_revision', 'tokenizer_digest', 'template_digest',
                'representation', 'runtime_version')


def cache_namespace(identity):
    if any(not isinstance(identity.get(field), str) or not identity[field] for field in CACHE_FIELDS):
        raise ValueError('native_cache_identity_incomplete')
    payload = {field: identity[field] for field in CACHE_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def scope_native_cache(config, identity):
    """Copy config; retain the prior cache directory and never reuse its entries."""
    import copy
    result = copy.copy(config)
    if not result.paged_ssd_cache_dir:
        raise ValueError('native_cache_directory_required')
    result.paged_ssd_cache_dir = str(Path(result.paged_ssd_cache_dir) / 'native' / cache_namespace(identity))
    return result
