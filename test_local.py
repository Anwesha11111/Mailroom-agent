"""Protocol tests (no real model): LLM_PROVIDER=mock. Run: python test_local.py"""
import base64
import os
import tempfile

os.environ["LLM_PROVIDER"] = "mock"
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

PROFILE = main.PROFILE
client = None


def b64u(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def make_key():
    sk = Ed25519PrivateKey.generate()
    raw = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sk, {"kty": "OKP", "crv": "Ed25519", "x": b64u(raw)}


def dossier(i):
    return {"dossierId": f"d{i}", "sources": [{"sourceId": f"s{i}", "lines": [{"lineId": f"d{i}-L1", "text": "hi"}]}]}


def sign(sk, eid, idig, receipt):
    msg = main.canon({"profile": PROFILE, "evaluationId": eid, "inputDigest": idig, "receipt": receipt})
    return base64.b64encode(sk.sign(msg)).decode()


def receipt_for(sk, eid, idig, p, accepted=True, rid=None):
    r = {"dossierId": p["dossierId"], "callId": p["callId"], "action": p["action"], "accepted": accepted,
         "proposalDigest": main.proposal_digest(p), "receiptId": rid or "rcpt-" + p["dossierId"]}
    r["receiptSignature"] = sign(sk, eid, idig, r)
    return r


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


with TestClient(main.app) as client:
    sk, jwk = make_key()
    dossiers = [dossier(1), dossier(2), dossier(3)]
    req = {"profile": PROFILE, "operation": "propose", "evaluationId": "e1", "dossiers": dossiers,
           "verifierPublicKey": jwk}

    r = client.post("/", json=req)
    check("propose 200", r.status_code == 200)
    prop = r.json()
    check("awaiting_receipts", prop["status"] == "awaiting_receipts")
    check("inputDigest", prop["inputDigest"] == main.sha(main.canon(dossiers)))
    check("one proposal per dossier", [p["dossierId"] for p in prop["proposals"]] == ["d1", "d2", "d3"])
    check("replay identical", client.post("/", json=req).json() == prop)
    changed = dict(req, dossiers=[dossier(1), dossier(2), dict(dossier(3), extra=1)])
    check("changed content -> 409", client.post("/", json=changed).status_code == 409)
    r2 = client.post("/", json=dict(req, evaluationId="e2")).json()
    check("stable callIds across evaluations", [p["callId"] for p in r2["proposals"]] == [p["callId"] for p in prop["proposals"]])
    check("malformed json -> 400", client.post("/", content=b"{nope", headers={"content-type": "application/json"}).status_code == 400)
    check("dup dossierId -> 400", client.post("/", json=dict(req, evaluationId="e3", dossiers=[dossier(1), dossier(1)])).status_code == 400)
    check("bad profile -> 400", client.post("/", json=dict(req, profile="x")).status_code == 400)

    idig = prop["inputDigest"]
    props = prop["proposals"]
    commit = lambda rc, eid="e1", dig=None: {"profile": PROFILE, "operation": "commit", "evaluationId": eid,
                                             "inputDigest": dig or idig, "receipts": rc}

    other_sk, _ = make_key()
    bad = [receipt_for(sk, "e1", idig, props[0]), receipt_for(other_sk, "e1", idig, props[1])]
    check("bad signature rejects whole commit", client.post("/", json=commit(bad)).status_code == 400)
    moved = receipt_for(sk, "e1", idig, props[0])
    moved["dossierId"] = "d2"
    check("moved receipt rejected", client.post("/", json=commit([moved])).status_code == 400)
    check("unknown evaluation -> 400", client.post("/", json=commit([], eid="nope")).status_code == 400)

    good = [receipt_for(sk, "e1", idig, props[0], True), receipt_for(sk, "e1", idig, props[1], False)]
    r = client.post("/", json=commit(good))
    check("commit 200", r.status_code == 200)
    out = r.json()
    check("completed", out["status"] == "completed")
    check("executed/rejected", [o["status"] for o in out["outcomes"]] == ["executed", "rejected"])
    check("commit replay identical", client.post("/", json=commit(good)).json() == out)
    other = [receipt_for(sk, "e1", idig, props[2])]
    check("different commit after completion -> 409", client.post("/", json=commit(other)).status_code == 409)
print("ALL OK")
