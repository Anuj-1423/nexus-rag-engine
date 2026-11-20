import os
import sys

from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auth import create_token
from main import app


def test_employee_cannot_query_list_or_delete_another_users_documents():
    token = create_token({"email": "employee@example.com", "role": "employee"})
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(app)

    ask = client.post(
        "/ask",
        headers=headers,
        json={
            "email": "other-user@example.com",
            "question": "show me their documents",
            "mode": "combined",
        },
    )
    listing = client.get("/documents/other-user@example.com", headers=headers)
    deletion = client.delete(
        "/documents/other-user@example.com/private.txt?scope=personal",
        headers=headers,
    )

    assert ask.status_code == 403
    assert listing.status_code == 403
    assert deletion.status_code == 403
