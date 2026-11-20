import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import main
from auth import create_token


class FakeCursor:
    def __init__(self):
        self.query = ""
        self.params = ()
        self.history_query = None
        self.history_params = None

    def execute(self, query, params=()):
        self.query = query
        self.params = params
        if "FROM chats" in query:
            self.history_query = query
            self.history_params = params

    def fetchone(self):
        return (False,)

    def fetchall(self):
        if "FROM chats" in self.query:
            return [
                ("latest same-mode question", "latest same-mode answer"),
                ("older same-mode question", "older same-mode answer"),
            ]
        return []

    def close(self):
        pass


class FakeConnection:
    def __init__(self):
        self.fake_cursor = FakeCursor()

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        pass

    def close(self):
        pass


@pytest.mark.parametrize("mode", ["personal", "enterprise", "combined"])
def test_chat_history_is_limited_to_active_mode(monkeypatch, mode):
    conn = FakeConnection()
    monkeypatch.setattr(main, "get_db_connection", lambda *_: conn)
    captured = {}

    async def fake_generate(*args, **kwargs):
        captured.update(kwargs)
        return {"answer": "test answer", "sources": []}

    monkeypatch.setattr(main, "generate_rag_response", fake_generate)
    token = create_token({"email": "employee@example.com", "role": "employee"})
    response = TestClient(main.app).post(
        "/ask",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": "employee@example.com", "question": "test", "mode": mode},
    )

    assert response.status_code == 200
    assert "scope = %s" in conn.fake_cursor.history_query
    assert conn.fake_cursor.history_params == ("employee@example.com", mode)
    assert captured["chat_history"] == [
        {"question": "older same-mode question", "answer": "older same-mode answer"},
        {"question": "latest same-mode question", "answer": "latest same-mode answer"},
    ]
