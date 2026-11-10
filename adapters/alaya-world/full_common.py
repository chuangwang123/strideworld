"""CPU-only full-campaign contracts; the OFFICIAL_V1 candidate profile is imported unchanged."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

RUNTIME_OFFICIAL_V1 = Path(__file__).resolve().parent.parent / 'runtime_official_v2'
sys.path.insert(0, str(RUNTIME_OFFICIAL_V1))
from common import COMMIT, HOST_ROOT, REPRO_ROOT, PROFILE, load, sha

CAMPAIGN = HOST_ROOT/'work/strideworld_refresh_20260915/runtime/native_inputs/alaya-world/native_campaign.json'
EXPECTED_SEEDS = {0: 1960, 1: 840, 2: 840}
ALLOWED_ACTIONS = {'W', 'S', 'A', 'D', 'YL', 'YR'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def relative_path(value):
    require(isinstance(value, str) and bool(value), 'nonempty relative path required')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and path.parts and path.as_posix() == value,
            f'unsafe or noncanonical relative path: {value}')
    return Path(*path.parts)


def source_image(row, data_root):
    uri = row['condition']['image_uri']
    require(isinstance(uri, str) and uri.startswith('data://cases/'), f'unsupported image URI: {uri}')
    path = (data_root / relative_path(uri[len('data://'):])).resolve()
    require(path.is_relative_to(data_root.resolve()), f'image URI escaped data root: {uri}')
    return path


def original_record(row):
    return {key: value for key, value in row.items() if key != 'alaya_official_v1'}


def validate_campaign(records):
    """extra-seed 2026-11-08 (prep_B): inventory check for a extra-seed seed manifest (gate/smoke/bulk) instead of the frozen 3640-row
    campaign (was: exactly 3640 rows, seeds 0:1960/1:840/2:840, 140 cases with 14/6/6 rows). Per-row checks unchanged."""
    require(len(records) > 0, 'empty extra-seed manifest')
    ids, destinations = set(), set()
    for index, row in enumerate(records):
        require(type(row['seed']) is int and row['seed'] in EXPECTED_SEEDS, f'invalid seed at {index}')
        require(row.get('kind') == 'TASK1_TASK3_SHARED_COMPLEX', f'extra-seed manifests hold composed routes only: {row.get("record_id")}')
        rid = row['record_id']
        require(isinstance(rid, str) and rid and rid not in ids, f'duplicate/invalid record ID: {rid}')
        ids.add(rid)
        plan = row['v10_plan']
        for key in ('scene_id', 'case_id', 'route_id', 'expanded_unit_actions'):
            require(key in plan, f'missing v10_plan.{key} for {rid}')
        actions = plan['expanded_unit_actions']
        require(isinstance(actions, list) and actions and all(a in ALLOWED_ACTIONS for a in actions),
                f'invalid expanded actions: {rid}')
        destination = (row.get('output') or row['v10_plan'].get('output', {}))['attempt_relpath']
        relative_path(destination)
        require(destination not in destinations, f'duplicate output.attempt_relpath: {destination}')
        destinations.add(destination)
        expected_hash = row['condition']['image_sha256']
        require(isinstance(expected_hash, str) and len(expected_hash) == 64
                and all(c in '0123456789abcdef' for c in expected_hash), f'invalid source image SHA: {rid}')
        require(isinstance(row['condition']['prompt'], str), f'prompt must be a string: {rid}')
    return {'records': len(records), 'seed_counts': dict(Counter(r['seed'] for r in records)),
            'unique_cases': len({(r['scene_id'], r['case_id']) for r in records}), 'unique_output_directories': len(destinations)}


def validate_authoritative_plan(records, plan_path):
    """Cross-check campaign identity against the HOST model-specific authoritative plan."""
    if plan_path is None:
        return {'status': 'NOT_CHECKED'}
    plan_path = Path(plan_path)
    require(plan_path.is_file(), f'missing authoritative plan: {plan_path}')
    rows = {}
    with plan_path.open() as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                key = canonical(item.get('primary_key', {
                    'model': item.get('model'), 'scene_id': item.get('scene_id'),
                    'case_id': item.get('case_id'), 'seed': item.get('seed'),
                    'task': item.get('task'), 'route_id': item.get('route_id')}))
                rows[key] = item
    require(len(rows) == len(records), f'authoritative plan rows {len(rows)} != campaign {len(records)}')
    for row in records:
        p = row['v10_plan']; key_obj = p.get('primary_key')
        require(key_obj is not None, f'missing campaign primary_key: {row["record_id"]}')
        key = canonical(key_obj); auth = rows.get(key)
        require(auth is not None, f'campaign key absent from authoritative plan: {row["record_id"]}')
        require(canonical(auth.get('expanded_unit_actions')) == canonical(p.get('expanded_unit_actions')),
                f'authoritative actions mismatch: {row["record_id"]}')
        require(canonical(auth.get('output', {}).get('attempt_relpath')) == canonical(p.get('output', {}).get('attempt_relpath')),
                f'authoritative output mismatch: {row["record_id"]}')
    return {'status': 'PASS', 'path': str(plan_path), 'sha256': sha(plan_path), 'rows': len(rows)}


def verify_acceptance(path):
    require(path is not None, 'generation requires --smoke-acceptance-json from the completed visual review')
    path = Path(path).resolve()
    acceptance = load(path)
    require(acceptance.get('visual_pass') is True, 'smoke acceptance must have visual_pass: true')
    require(canonical(acceptance.get('profile')) == canonical(PROFILE),
            'accepted profile must exactly equal runtime_official_v2.common.PROFILE')
    return {'path': str(path), 'sha256': sha(path), 'visual_pass': True, 'profile': PROFILE}


def durable_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def verify_row_files(row):
    native = row['alaya_official_v1']
    require(digest(original_record(row)) == native['source_record_sha256'],
            f'original campaign fields changed: {row["record_id"]}')
    require(canonical(native['profile']) == canonical(PROFILE), 'row profile drift')
    require(sha(native['source_image']) == row['condition']['image_sha256'], 'source RGB hash changed')
    require(sha(native['source_camera']) == native['source_camera_sha256'], 'source camera hash changed')
    for item in native['input_files']:
        require(sha(item['path']) == item['sha256'], f'prepared input changed: {item["path"]}')
    return native
