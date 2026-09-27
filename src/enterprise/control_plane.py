"""Enterprise control-plane primitives: entitlements, moderation, compliance, identity, audit, supply-chain, secrets, rollout and DR."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class EnterpriseDB:
    def __init__(self, path="data/cache/enterprise.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.executescript("""
        CREATE TABLE IF NOT EXISTS entitlements(tenant TEXT,feature TEXT,value_json TEXT,PRIMARY KEY(tenant,feature));
        CREATE TABLE IF NOT EXISTS usage(tenant TEXT,meter TEXT,period TEXT,quantity REAL,PRIMARY KEY(tenant,meter,period));
        CREATE TABLE IF NOT EXISTS legal_holds(tenant TEXT,hold_id TEXT PRIMARY KEY,reason TEXT,created_at INTEGER,released_at INTEGER);
        CREATE TABLE IF NOT EXISTS identities(id TEXT PRIMARY KEY,tenant TEXT,user_name TEXT,active INTEGER,attrs_json TEXT); CREATE UNIQUE INDEX IF NOT EXISTS identity_name ON identities(tenant,user_name);
        CREATE TABLE IF NOT EXISTS groups(id TEXT PRIMARY KEY,tenant TEXT,name TEXT,members_json TEXT); CREATE UNIQUE INDEX IF NOT EXISTS group_name ON groups(tenant,name);
        """)

    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c


class EntitlementService:
    def __init__(self, db: EnterpriseDB):
        self.db = db

    def set(self, tenant, feature, value):
        with self.db.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO entitlements VALUES(?,?,?)",
                (tenant, feature, json.dumps(value)),
            )

    def get(self, tenant, feature, default=None):
        with self.db.conn() as c:
            r = c.execute(
                "SELECT value_json FROM entitlements WHERE tenant=? AND feature=?",
                (tenant, feature),
            ).fetchone()
        return json.loads(r[0]) if r else default

    def require(self, tenant, feature):
        if not self.get(tenant, feature, False):
            raise PermissionError(f"feature {feature!r} is not entitled")

    def meter(self, tenant, meter, quantity=1.0, period=None):
        period = period or time.strftime("%Y-%m", time.gmtime())
        with self.db.conn() as c:
            c.execute(
                "INSERT INTO usage VALUES(?,?,?,?) ON CONFLICT(tenant,meter,period) DO UPDATE SET quantity=quantity+excluded.quantity",
                (tenant, meter, period, float(quantity)),
            )

    def usage(self, tenant, period=None):
        period = period or time.strftime("%Y-%m", time.gmtime())
        with self.db.conn() as c:
            rows = c.execute(
                "SELECT meter,quantity FROM usage WHERE tenant=? AND period=?",
                (tenant, period),
            ).fetchall()
        return {r["meter"]: r["quantity"] for r in rows}


@dataclass(frozen=True)
class ModerationResult:
    flagged: bool
    categories: dict[str, bool]
    scores: dict[str, float]
    model: str = "rules-v1"


class ModerationService:
    DEFAULT = {
        "violence": [r"\bkill\b", r"\bmurder\b"],
        "self_harm": [r"\bsuicide\b", r"\bself[- ]harm\b"],
        "sexual_minors": [r"\bchild porn"],
        "hate": [r"\bgenocide\b.{0,20}\b(race|religion)\b"],
    }

    def __init__(
        self,
        rules=None,
        classifier: Callable[[str], dict[str, float]] | None = None,
        threshold=0.5,
    ):
        self.rules = rules or self.DEFAULT
        self.classifier = classifier
        self.threshold = threshold

    def moderate(self, text):
        scores = {
            k: (1.0 if any(re.search(p, text, re.IGNORECASE) for p in pats) else 0.0)
            for k, pats in self.rules.items()
        }
        if self.classifier:
            for k, v in self.classifier(text).items():
                scores[k] = max(scores.get(k, 0), float(v))
        cats = {k: v >= self.threshold for k, v in scores.items()}
        return ModerationResult(
            any(cats.values()),
            cats,
            scores,
            "hybrid-v1" if self.classifier else "rules-v1",
        )


class ComplianceService:
    def __init__(self, db: EnterpriseDB, residency: dict[str, set[str]] | None = None):
        self.db = db
        self.residency = residency or {}

    def check_region(self, tenant, region):
        allowed = self.residency.get(tenant)
        return True if not allowed else region in allowed

    def place_hold(self, tenant, reason):
        ident = f"hold_{uuid.uuid4().hex}"
        with self.db.conn() as c:
            c.execute(
                "INSERT INTO legal_holds VALUES(?,?,?,?,NULL)",
                (tenant, ident, reason, int(time.time())),
            )
        return ident

    def release_hold(self, tenant, hold_id):
        with self.db.conn() as c:
            c.execute(
                "UPDATE legal_holds SET released_at=? WHERE tenant=? AND hold_id=?",
                (int(time.time()), tenant, hold_id),
            )

    def under_hold(self, tenant):
        with self.db.conn() as c:
            return (
                c.execute(
                    "SELECT 1 FROM legal_holds WHERE tenant=? AND released_at IS NULL",
                    (tenant,),
                ).fetchone()
                is not None
            )

    def enforce_retention(self, tenant, records, now=None):
        if self.under_hold(tenant):
            return records
        now = now or int(time.time())
        return [
            r for r in records if not r.get("expires_at") or int(r["expires_at"]) > now
        ]

    def report(self, tenant):
        return {
            "tenant": tenant,
            "legal_hold": self.under_hold(tenant),
            "allowed_regions": sorted(self.residency.get(tenant, set())),
            "generated_at": int(time.time()),
        }


class SCIMDirectory:
    def __init__(self, db: EnterpriseDB):
        self.db = db

    def upsert_user(self, tenant, user_name, active=True, **attrs):
        with self.db.conn() as c:
            old = c.execute(
                "SELECT id FROM identities WHERE tenant=? AND user_name=?",
                (tenant, user_name),
            ).fetchone()
            ident = old[0] if old else f"usr_{uuid.uuid4().hex}"
            c.execute(
                "INSERT OR REPLACE INTO identities VALUES(?,?,?,?,?)",
                (ident, tenant, user_name, int(active), json.dumps(attrs)),
            )
        return {"id": ident, "userName": user_name, "active": active, **attrs}

    def users(self, tenant):
        with self.db.conn() as c:
            rows = c.execute(
                "SELECT * FROM identities WHERE tenant=?", (tenant,)
            ).fetchall()
        return [
            {
                "id": r["id"],
                "userName": r["user_name"],
                "active": bool(r["active"]),
                **json.loads(r["attrs_json"]),
            }
            for r in rows
        ]

    def upsert_group(self, tenant, name, members=()):
        with self.db.conn() as c:
            old = c.execute(
                "SELECT id FROM groups WHERE tenant=? AND name=?", (tenant, name)
            ).fetchone()
            ident = old[0] if old else f"grp_{uuid.uuid4().hex}"
            c.execute(
                "INSERT OR REPLACE INTO groups VALUES(?,?,?,?)",
                (ident, tenant, name, json.dumps(list(members))),
            )
        return {"id": ident, "displayName": name, "members": list(members)}


class PolicyEngine:
    def __init__(
        self,
        local_rules: dict[str, set[str]] | None = None,
        external_url: str | None = None,
    ):
        self.local_rules = local_rules or {}
        self.external_url = external_url

    def allow(self, subject_roles, action, resource=None):
        if self.external_url:
            import httpx

            r = httpx.post(
                self.external_url,
                json={
                    "input": {
                        "roles": list(subject_roles),
                        "action": action,
                        "resource": resource,
                    }
                },
                timeout=5,
            )
            r.raise_for_status()
            return bool(r.json().get("result", False))
        return any(
            action in self.local_rules.get(role, set())
            or "*" in self.local_rules.get(role, set())
            for role in subject_roles
        )


class TamperEvidentAuditLog:
    def __init__(self, path, secret):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.secret = secret.encode()

    def append(self, event):
        prev = "0" * 64
        if self.path.exists():
            lines = self.path.read_text("utf-8").splitlines()
            prev = json.loads(lines[-1])["hash"] if lines else prev
        payload = {"ts": int(time.time()), "event": event, "prev_hash": prev}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        sig = hmac.new(self.secret, digest.encode(), hashlib.sha256).hexdigest()
        rec = {**payload, "hash": digest, "signature": sig}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def verify(self):
        prev = "0" * 64
        for line in (
            self.path.read_text("utf-8").splitlines() if self.path.exists() else []
        ):
            r = json.loads(line)
            payload = {"ts": r["ts"], "event": r["event"], "prev_hash": r["prev_hash"]}
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if (
                r["prev_hash"] != prev
                or digest != r["hash"]
                or not hmac.compare_digest(
                    hmac.new(self.secret, digest.encode(), hashlib.sha256).hexdigest(),
                    r["signature"],
                )
            ):
                return False
            prev = r["hash"]
        return True


class ArtifactAdmission:
    def __init__(self, secret: bytes):
        self.secret = secret

    def manifest(self, path):
        p = Path(path)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        return {
            "path": p.name,
            "sha256": digest,
            "size": p.stat().st_size,
            "sbom": {"format": "llm-engine-sbom-v1", "generated_at": int(time.time())},
        }

    def sign(self, manifest):
        return hmac.new(
            self.secret,
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
            hashlib.sha256,
        ).hexdigest()

    def verify(self, path, manifest, signature):
        return (
            hmac.compare_digest(self.sign(manifest), signature)
            and hashlib.sha256(Path(path).read_bytes()).hexdigest()
            == manifest["sha256"]
        )


class SecretManager:
    def __init__(self, prefix="GOPI_SECRET_"):
        self.prefix = prefix

    def get(self, name):
        env = os.getenv(self.prefix + name.upper().replace("-", "_"))
        if env is not None:
            return env
        backend = os.getenv("GOPI_SECRET_BACKEND", "").lower()
        if backend == "aws":
            import boto3

            return (
                boto3.client("secretsmanager")
                .get_secret_value(SecretId=name)
                .get("SecretString")
            )
        if backend == "vault":
            import httpx

            base = os.environ["GOPI_VAULT_ADDR"].rstrip("/")
            token = os.environ["GOPI_VAULT_TOKEN"]
            r = httpx.get(
                f"{base}/v1/{name}", headers={"X-Vault-Token": token}, timeout=5
            )
            r.raise_for_status()
            data = r.json()["data"]
            return data.get("value") or data.get("data", {}).get("value")
        raise KeyError(name)


class SandboxRunner:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def run(self, command, cwd=None):
        wrapper = []
        if shutil.which("bwrap"):
            wrapper = [
                "bwrap",
                "--unshare-all",
                "--die-with-parent",
                "--ro-bind",
                "/usr",
                "/usr",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
            ]
        elif shutil.which("firejail"):
            wrapper = ["firejail", "--quiet", "--private", "--net=none"]
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
        return subprocess.run(
            wrapper + list(command),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )


class DeploymentManager:
    @staticmethod
    def kubernetes(name, image, replicas=2, min_replicas=2, max_replicas=10, port=8000):
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": name},
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": {"app": name}},
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "containers": [
                            {
                                "name": name,
                                "image": image,
                                "ports": [{"containerPort": port}],
                                "readinessProbe": {
                                    "httpGet": {"path": "/health", "port": port}
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": port}
                                },
                                "resources": {
                                    "requests": {"cpu": "500m", "memory": "1Gi"},
                                    "limits": {"cpu": "4", "memory": "16Gi"},
                                },
                            }
                        ]
                    },
                },
            },
        }
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": name},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": name,
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {"type": "Utilization", "averageUtilization": 70},
                        },
                    }
                ],
            },
        }
        return {"deployment": deployment, "hpa": hpa}

    @staticmethod
    def rollout(current, candidate, metrics, limits):
        bad = [k for k, v in metrics.items() if k in limits and v > limits[k]]
        return {
            "action": "rollback" if bad else "promote",
            "from": current,
            "to": candidate,
            "violations": bad,
        }


class BackupManager:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def backup(self, paths):
        dest = self.root / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        dest.mkdir()
        manifest = []
        for p in map(Path, paths):
            if not p.exists():
                continue
            target = dest / p.name
            shutil.copytree(p, target) if p.is_dir() else shutil.copy2(p, target)
            manifest.append(
                {
                    "name": p.name,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()
                    if target.is_file()
                    else None,
                }
            )
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return dest

    def restore(self, backup_dir, destination):
        src = Path(backup_dir)
        dst = Path(destination)
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.name == "manifest.json":
                continue
            target = dst / p.name
            shutil.copytree(
                p, target, dirs_exist_ok=True
            ) if p.is_dir() else shutil.copy2(p, target)


__all__ = [
    "ArtifactAdmission",
    "BackupManager",
    "ComplianceService",
    "DeploymentManager",
    "EnterpriseDB",
    "EntitlementService",
    "ModerationResult",
    "ModerationService",
    "PolicyEngine",
    "SCIMDirectory",
    "SandboxRunner",
    "SecretManager",
    "TamperEvidentAuditLog",
]


class KMSProvider:
    """Signing adapter for local HMAC, AWS KMS asymmetric keys, or PKCS#11 HSM keys."""

    def __init__(
        self,
        *,
        local_secret: bytes | None = None,
        aws_key_id: str | None = None,
        pkcs11_uri: str | None = None,
    ):
        self.local_secret = local_secret
        self.aws_key_id = aws_key_id
        self.pkcs11_uri = pkcs11_uri

    def sign(self, payload: bytes) -> bytes:
        if self.local_secret:
            return hmac.new(self.local_secret, payload, hashlib.sha256).digest()
        if self.aws_key_id:
            import boto3

            return boto3.client("kms").sign(
                KeyId=self.aws_key_id,
                Message=payload,
                MessageType="RAW",
                SigningAlgorithm="RSASSA_PSS_SHA_256",
            )["Signature"]
        if self.pkcs11_uri:
            import pkcs11

            lib = pkcs11.lib(os.environ["GOPI_PKCS11_LIBRARY"])
            token = lib.get_token(token_label=os.environ["GOPI_PKCS11_TOKEN"])
            with token.open(user_pin=os.environ["GOPI_PKCS11_PIN"]) as session:
                key = session.get_key(
                    object_class=pkcs11.ObjectClass.PRIVATE_KEY, label=self.pkcs11_uri
                )
                return bytes(
                    key.sign(payload, mechanism=pkcs11.Mechanism.SHA256_RSA_PKCS_PSS)
                )
        raise RuntimeError("no KMS/HSM signing backend configured")


class WebhookRegistry:
    def __init__(self, path="data/cache/webhooks.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS hooks(id TEXT PRIMARY KEY,tenant TEXT,url TEXT,secret TEXT,events_json TEXT,active INTEGER)"
            )

    def register(self, tenant, url, secret, events):
        if not url.startswith(("https://", "http://")):
            raise ValueError("webhook URL must be http(s)")
        ident = f"wh_{uuid.uuid4().hex}"
        with sqlite3.connect(self.path) as c:
            c.execute(
                "INSERT INTO hooks VALUES(?,?,?,?,?,1)",
                (ident, tenant, url, secret, json.dumps(sorted(set(events)))),
            )
        return ident

    def subscriptions(self, tenant, event):
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM hooks WHERE tenant=? AND active=1", (tenant,)
            ).fetchall()
        return [
            dict(r)
            for r in rows
            if event in json.loads(r["events_json"])
            or "*" in json.loads(r["events_json"])
        ]

    def disable(self, tenant, ident):
        with sqlite3.connect(self.path) as c:
            c.execute(
                "UPDATE hooks SET active=0 WHERE tenant=? AND id=?", (tenant, ident)
            )


class ReplicaRegistry:
    def __init__(self, ttl_seconds=30):
        self.ttl = ttl_seconds
        self._nodes = {}
        self._cursor = 0

    def heartbeat(self, replica_id, url, region="default", healthy=True, metadata=None):
        self._nodes[replica_id] = {
            "id": replica_id,
            "url": url,
            "region": region,
            "healthy": healthy,
            "metadata": metadata or {},
            "seen": time.time(),
        }

    def available(self, region=None):
        return [
            n
            for n in self._nodes.values()
            if n["healthy"]
            and time.time() - n["seen"] <= self.ttl
            and (region is None or n["region"] == region)
        ]

    def route(self, region=None):
        nodes = self.available(region) or self.available()
        if not nodes:
            raise RuntimeError("no healthy serving replicas")
        node = nodes[self._cursor % len(nodes)]
        self._cursor += 1
        return node


class MultiRegionFailover:
    def __init__(self, primary, secondaries):
        self.primary = primary
        self.secondaries = list(secondaries)

    def choose(self, health):
        if health.get(self.primary, False):
            return self.primary
        for r in self.secondaries:
            if health.get(r, False):
                return r
        raise RuntimeError("no healthy region")


__all__ += ["KMSProvider", "MultiRegionFailover", "ReplicaRegistry", "WebhookRegistry"]
