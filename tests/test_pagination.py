import pytest

from conftest import ALL4, act, new_env
from traceforge.environment.actions import ActionError
from traceforge.environment.pagination import paginate


def _walk(env, table, limit=50):
    out, cursor, pages = [], None, 0
    while True:
        obs, *_ = act(env, "list_records", table=table, cursor=cursor, limit=limit)
        r = obs["last_result"]
        out += r["records"]
        pages += 1
        cursor = r["next_cursor"]
        if cursor is None:
            return out, pages


@pytest.mark.parametrize("table,attr", [("ledger", "ledger"), ("invoices", "invoices"),
                                        ("payments", "payments"), ("fx", "fx")])
def test_pages_cover_every_record_once_in_order(table, attr):
    env, *_ = new_env(1, **ALL4)
    rows, pages = _walk(env, table)
    src = getattr(env._pub, attr)
    assert len(rows) == len(src) and pages == -(-len(src) // 50)
    key = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "date"}[table]
    assert [r[key] for r in rows] == [r[key] for r in src]


def test_pagination_is_stable_across_traversals_and_page_sizes():
    env, *_ = new_env(1, **ALL4)
    a, _ = _walk(env, "ledger", 50)
    b, _ = _walk(env, "ledger", 50)
    c, _ = _walk(env, "ledger", 7)
    assert a == b == c


def test_cursor_is_stable_regardless_of_interleaved_actions():
    env, *_ = new_env(1, **ALL4)
    first = act(env, "list_records", table="ledger", limit=10)[0]["last_result"]
    act(env, "get_record", table="ledger", id="ROW-00001")
    act(env, "list_records", table="payments")
    second = act(env, "list_records", table="ledger", cursor=first["next_cursor"], limit=10)[0]["last_result"]
    again = act(env, "list_records", table="ledger", cursor=first["next_cursor"], limit=10)[0]["last_result"]
    assert second["records"] == again["records"] and second["records"][0]["row_id"] == "ROW-00011"


def test_last_page_has_no_next_cursor_and_reports_total():
    env, *_ = new_env(1, **ALL4)
    n = len(env._pub.payments)
    obs, *_ = act(env, "list_records", table="payments", limit=50)
    r = obs["last_result"]
    assert r["total"] == n
    if n <= 50:
        assert r["next_cursor"] is None


def test_page_size_boundaries():
    for limit in (1, 49, 50):
        assert len(paginate(list(range(120)), None, limit)[0]) == limit
    for bad in (0, 51, -1, 1.5, "5", True, None):
        with pytest.raises(ActionError):
            paginate(list(range(120)), None, bad)


@pytest.mark.parametrize("cursor", ["", "50", "c-1", "cabc", "c9999999", 5, ["c0"], "c1; DROP"])
def test_invalid_cursors_are_rejected_and_cost_a_step(cursor):
    env, *_ = new_env(1)
    obs, *_ = act(env, "list_records", table="ledger", cursor=cursor)
    assert obs["error"]["code"] == "invalid_cursor" and obs["steps_remaining"] == 99


def test_cursor_at_end_returns_empty_page():
    page, nxt = paginate([1, 2, 3], "c3", 50)
    assert page == [] and nxt is None
