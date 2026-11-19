import json

import pytest

from conftest import ALL4, act, finding, new_env, oracle_action, submit
from traceforge.environment.audit_env import AuditEnv
from traceforge.exceptions import EnvError, EpisodeFinished, NotReset

OBS_KEYS = {"task", "manifest", "policy", "last_result", "error", "steps_remaining"}


def test_reset_returns_observation_and_info():
    env, obs, info = new_env(1, **ALL4)
    assert set(obs) == OBS_KEYS
    assert obs["steps_remaining"] == 100 and obs["last_result"] is None and obs["error"] is None
    assert set(info) == {"episode_id", "mode"} and info["episode_id"] == obs["manifest"]["episode_id"]


def test_step_returns_five_tuple_and_decrements_steps():
    env, obs, _ = new_env(1)
    out = act(env, "calculator", expression="1+1")
    assert len(out) == 5
    obs2, reward, terminated, truncated, info = out
    assert obs2["steps_remaining"] == 99 and reward == 0.0 and not terminated and not truncated
    assert obs2["last_result"]["value"] == "2" and info == {"step": 1}


def test_observation_contains_only_whitelisted_keys_at_every_step():
    env, obs, _ = new_env(1, **ALL4)
    for a in ({"action": "list_records", "table": "ledger"}, {"action": "nope"}, {"action": "get_record", "table": "fx", "id": "x"}):
        obs, *_ = env.step(a)
        assert set(obs) == OBS_KEYS


def test_list_records_max_page_is_50():
    env, *_ = new_env(1)
    obs, *_ = act(env, "list_records", table="ledger", limit=50)
    assert len(obs["last_result"]["records"]) == 50
    obs, *_ = act(env, "list_records", table="ledger", limit=51)
    assert obs["error"]["code"] == "invalid_limit" and obs["last_result"]["ok"] is False


def test_default_page_is_50_records():
    env, *_ = new_env(1)
    obs, *_ = act(env, "list_records", table="ledger")
    assert len(obs["last_result"]["records"]) == 50 and obs["last_result"]["next_cursor"] == "c50"


def test_unknown_id_consumes_a_step_with_bounded_error():
    env, *_ = new_env(1)
    obs, reward, term, trunc, _ = act(env, "get_record", table="ledger", id="ROW-99999")
    assert obs["steps_remaining"] == 99 and obs["error"] == {"code": "not_found", "message": "no such record"}
    assert "ROW-99999" not in json.dumps(obs["error"])
    obs, *_ = act(env, "get_document", doc_id="DOC-9999")
    assert obs["steps_remaining"] == 98 and obs["error"]["code"] == "not_found"


@pytest.mark.parametrize("bad", [{"action": "rm_rf"}, {}, {"action": None}, "list_records", 42, None,
                                 {"action": "get_record", "table": "secrets", "id": "x"},
                                 {"action": "get_record", "table": "ledger"},
                                 {"action": "calculator", "expression": 5},
                                 {"action": "submit"}])
def test_invalid_action_consumes_a_step(bad):
    env, *_ = new_env(1)
    obs, reward, term, trunc, _ = env.step(bad)
    assert obs["steps_remaining"] == 99 and obs["error"] is not None
    assert reward == 0.0 and not term and not trunc


def test_get_record_works_for_every_table():
    env, *_ = new_env(1)
    pub = env._pub
    for table, key in (("ledger", pub.ledger[0]["row_id"]), ("invoices", pub.invoices[0]["invoice_id"]),
                       ("payments", pub.payments[0]["payment_id"]),
                       ("fx", f"FX-{pub.fx[0]['date']}-{pub.fx[0]['source_currency']}-{pub.fx[0]['target_currency']}")):
        obs, *_ = act(env, "get_record", table=table, id=key)
        assert obs["error"] is None and obs["last_result"]["record"]


def test_get_document_returns_invoice_and_payment():
    env, *_ = new_env(1)
    for doc in (env._pub.invoices[0], env._pub.payments[0]):
        obs, *_ = act(env, "get_document", doc_id=doc["doc_id"])
        assert obs["last_result"]["document"] == doc


def test_calculator_exact_decimal_result():
    env, *_ = new_env(1)
    obs, *_ = act(env, "calculator", expression="1234.56 * 1.0825")
    assert obs["last_result"]["value"] == "1336.4112" and obs["error"] is None


def test_submit_terminates_and_train_mode_returns_reward():
    env, obs, _ = new_env(1, **ALL4)
    obs, reward, term, trunc, info = env.step(oracle_action(env, obs))
    assert term and not trunc and reward == 1.0 and info["submission_valid"] is True


def test_exploration_steps_give_zero_reward_and_terminal_reward_pays_step_cost():
    env, obs, _ = new_env(1, **ALL4)
    for _ in range(5):
        assert act(env, "calculator", expression="2*2")[1] == 0.0
    _, reward, term, *_ = env.step(oracle_action(env, obs))
    assert term and reward == pytest.approx(1.0 - 0.002 * 5)


def test_100_steps_without_submit_truncates_with_empty_answer():
    env, *_ = new_env(1, clean=True)
    for i in range(99):
        _, _, term, trunc, _ = act(env, "calculator", expression="1")
        assert not term and not trunc
    obs, reward, term, trunc, info = act(env, "calculator", expression="1")
    assert trunc and not term and obs["steps_remaining"] == 0
    assert env.result.n_findings == 0 and env.result.metrics.inspection_steps == 100
    assert reward == pytest.approx(max(-1, 1 - 0.2))              # clean episode, empty answer: F1=1
    with pytest.raises(EpisodeFinished):
        act(env, "calculator", expression="1")


def test_truncation_on_episode_with_errors_scores_empty_answer():
    env, *_ = new_env(1, **ALL4)
    for _ in range(100):
        out = act(env, "calculator", expression="1")
    assert out[3] and env.result.metrics.tp == 0 and env.result.metrics.fn == 4 and out[1] == -1.0


def test_submit_as_the_100th_action_is_accepted():
    env, obs, _ = new_env(1, **ALL4)
    for _ in range(99):
        act(env, "calculator", expression="1")
    _, reward, term, trunc, _ = env.step(oracle_action(env, obs))
    assert term and not trunc and env.result.metrics.inspection_steps == 99 and env.result.metrics.tp == 4


def test_step_before_reset_raises():
    with pytest.raises(NotReset):
        AuditEnv().step({"action": "calculator", "expression": "1"})


def test_second_reset_starts_a_fresh_episode():
    env, obs, _ = new_env(1, **ALL4)
    for _ in range(10):
        act(env, "calculator", expression="1")
    submit(env, [])
    obs2, info2 = env.reset(1, ALL4)
    assert obs2["steps_remaining"] == 100 and env.result is None
    assert act(env, "calculator", expression="1")[0]["steps_remaining"] == 99


def test_eval_mode_hides_reward_until_runner_reads_it():
    env, obs, _ = new_env(1, mode="eval", **ALL4)
    _, reward, term, _, info = env.step(oracle_action(env, obs))
    assert term and reward == 0.0 and info == {"step": 1}
    assert env.result.reward == 1.0                       # evaluator side only


def test_train_and_eval_scores_agree():
    scores = []
    for mode in ("train", "eval"):
        env, obs, _ = new_env(2, mode=mode, **ALL4)
        env.step(oracle_action(env, obs))
        scores.append(env.result.metrics.reward)
    assert scores[0] == scores[1] == 1.0


def test_observation_and_info_are_json_roundtrippable():
    env, obs, info = new_env(1, **ALL4)
    out = act(env, "list_records", table="payments")
    for x in (obs, info, out[0], out[4]):
        assert json.loads(json.dumps(x)) == x


def test_seed_truth_and_paths_never_appear_in_agent_view():
    seed = 987654321
    env, obs, info = new_env(seed, **ALL4)
    seen = [obs, info]
    for table in ("ledger", "invoices", "payments", "fx"):
        seen.append(act(env, "list_records", table=table)[0])
    seen.append(act(env, "get_record", table="ledger", id="NOPE")[0])
    text = json.dumps(seen)
    for token in env.leak_tokens() + [str(seed), "C:\\", "/home", "Users", "Traceback", "expected_amount"]:
        assert token not in text, token


def test_invalid_structure_submission_gives_minus_one_and_internal_log():
    env, obs, _ = new_env(1, **ALL4)
    obs, reward, term, _, info = env.step({"action": "submit", "findings": "not-a-list"})
    assert term and reward == -1.0 and info["submission_valid"] is False
    assert env.result.internal_log and "invalid submission" in env.result.internal_log[0]
    assert "invalid" not in json.dumps(obs["last_result"]).lower().replace("received", "")


def test_submit_with_wrong_episode_id_is_invalid():
    env, *_ = new_env(1, **ALL4)
    assert submit(env, [], episode_id="other")[1] == -1.0


def test_submit_accepts_bare_findings_list_for_convenience():
    env, obs, _ = new_env(1, **ALL4)
    _, reward, *_ = env.step({"action": "submit", "findings": [finding(t) for t in env.evaluator_truth()]})
    assert reward == 1.0


def test_invalid_mode_rejected():
    with pytest.raises(EnvError):
        AuditEnv("debug")


def test_stored_episode_can_be_loaded_by_directory(tmp_path):
    from traceforge.episode_io import save_episode
    from conftest import episode
    save_episode(episode(1, **ALL4), tmp_path / "ep")
    env = AuditEnv("train")
    obs, info = env.reset(config={"episode_dir": str(tmp_path / "ep")})
    assert obs["manifest"]["episode_id"] == episode(1, **ALL4).public.manifest["episode_id"]
    assert len(env.evaluator_truth()) == 4


def test_returned_records_are_copies_agents_cannot_corrupt_environment_state():
    env, *_ = new_env(1, **ALL4)
    page = act(env, "list_records", table="ledger", limit=5)[0]["last_result"]["records"]
    page[0]["debit_minor"] = -999
    page[0]["account"] = "HACKED"
    again = act(env, "list_records", table="ledger", limit=5)[0]["last_result"]["records"]
    assert again[0]["debit_minor"] != -999 and again[0]["account"] != "HACKED"
    rec = act(env, "get_record", table="invoices", id=env._pub.invoices[0]["invoice_id"])[0]["last_result"]["record"]
    rec["total_minor"] = 1
    rec["line_items"].append({"description": "x", "amount_minor": 1})
    doc = act(env, "get_document", doc_id=env._pub.invoices[0]["doc_id"])[0]["last_result"]["document"]
    assert doc["total_minor"] != 1 and len(doc["line_items"]) == len(env._pub.invoices[0]["line_items"])
    assert env._pub.ledger[0]["account"] != "HACKED"
