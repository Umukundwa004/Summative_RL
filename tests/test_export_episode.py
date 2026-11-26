from export_episode import build_episode_index, episode_path


def test_episode_paths_keep_seed_zero_compatible_and_separate_other_seeds(tmp_path):
    assert episode_path("ppo", 0, tmp_path).name == "ppo.json"
    assert episode_path("ppo", 2, tmp_path).name == "ppo-seed-2.json"


def test_episode_index_lists_existing_model_seeds(tmp_path):
    for filename in ("dqn.json", "dqn-seed-2.json", "unknown-seed-3.json"):
        (tmp_path / filename).touch()

    assert build_episode_index(tmp_path) == [
        {"id": "dqn", "label": "DQN (seed 0)"},
        {"id": "dqn-seed-2", "label": "DQN (seed 2)"},
    ]
