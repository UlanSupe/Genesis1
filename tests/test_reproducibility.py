import hashlib
import json
import random

from conftest import ALL4, new_env
from traceforge.episode_io import load_episode, private_files, public_files
from traceforge.generator.builder import dataset_specs, generate_dataset, generate_episode
from traceforge.runner.episode_runner import make_agent, run_episode
from traceforge.environment.audit_env import AuditEnv


def _digest(ep):
    h = hashlib.sha256()
    for files in (public_files(ep.public), private_files(ep.private)):
        for k in sorted(files):
            h.update((k + files[k]).encode())
    return h.hexdigest()


def test_same_seed_same_config_identical_public_and_private_bytes():
    assert _digest(generate_episode(11, ALL4)) == _digest(generate_episode(11, ALL4))


def test_config_change_changes_episode():
    assert _digest(generate_episode(11, ALL4)) != _digest(generate_episode(11, {**ALL4, "difficulty": "hard"}))


def test_reset_is_reproducible_across_env_instances():
    a, _, _ = new_env(5, **ALL4)
    b, _, _ = new_env(5, **ALL4)
    o1, _ = a.reset(5, ALL4)
    o2, _ = b.reset(5, ALL4)
    assert json.dumps(o1, sort_keys=True) == json.dumps(o2, sort_keys=True)


def test_repeated_reset_yields_same_episode():
    env, obs, _ = new_env(5, **ALL4)
    again, _ = env.reset(5, ALL4)
    assert obs == again


def test_full_run_is_reproducible_including_scores():
    runs = []
    for _ in range(2):
        r = run_episode(AuditEnv("eval"), make_agent("random", 3), seed=8, config=ALL4)
        r.pop("elapsed_time")
        runs.append(r)
    assert runs[0] == runs[1]


def test_global_random_state_does_not_influence_generation():
    random.seed(1)
    a = _digest(generate_episode(3, ALL4))
    random.seed(2)
    random.random()
    assert a == _digest(generate_episode(3, ALL4))


def test_episode_id_depends_on_seed_and_config_but_is_not_the_seed():
    a = generate_episode(1, ALL4).public.manifest["episode_id"]
    b = generate_episode(2, ALL4).public.manifest["episode_id"]
    assert a != b and a.startswith("ep-") and "1" != a


def test_stored_datasets_equal_regeneration(tmp_path):
    from pathlib import Path
    stored = Path(__file__).resolve().parents[1] / "datasets"
    assert (stored / "train" / "index.json").exists(), "run `python -m traceforge generate`"
    generate_dataset(tmp_path)
    for split in ("train", "dev"):
        for name, *_ in dataset_specs(split):
            assert _digest(load_episode(stored / split / name)) == _digest(load_episode(tmp_path / split / name))
