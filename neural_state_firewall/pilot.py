"""Authenticated, single-realm loopback pilot with a killable inference process.

The parent owns authentication, configuration and the wall-clock deadline. One
worker owns one immutable permission realm. Restart the service after a worker
failure; failed requests never trigger automatic model reloads or retries.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import secrets
import stat
import threading
import time

from .artifacts import digest, read
from .server import make_server


def load_token(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("Token file must be a regular file owned by this user with mode 0600")
        if info.st_size > 4096:
            raise ValueError("Oversized authentication token")
        return os.read(fd, 4097).decode("ascii").strip()
    finally:
        os.close(fd)


def load_config(path):
    path = Path(path)
    raw = read(path)
    common = {"model", "revision", "policy_file",
                "layers", "max_context", "max_new_tokens", "timeout_seconds", "startup_timeout_seconds"}
    if not isinstance(raw, dict) or set(raw) not in (
            common | {"documents_file", "read_permissions_file"}, common | {"workload_file"}):
        raise ValueError("Pilot configuration has missing or unknown fields")
    file_keys = {key for key in raw if key.endswith("_file")}
    for name in {"model", "revision"} | file_keys:
        if not isinstance(raw[name], str) or not raw[name]:
            raise ValueError("Missing model identity or host configuration path")
    if len(raw["revision"]) != 40 or any(c not in "0123456789abcdef" for c in raw["revision"]):
        raise ValueError("Pilot requires a pinned model commit")
    for name in ("timeout_seconds", "startup_timeout_seconds"):
        if type(raw[name]) not in (float, int) or not math.isfinite(raw[name]) or raw[name] <= 0:
            raise ValueError("Invalid process deadline")
    if (type(raw["max_context"]) is not int or not 1 <= raw["max_context"] <= 32768
            or type(raw["max_new_tokens"]) is not int or not 1 <= raw["max_new_tokens"] < raw["max_context"]
            or not isinstance(raw["layers"], list) or not raw["layers"]
            or any(type(x) is not int or x < 0 for x in raw["layers"])):
        raise ValueError("Invalid context, generation budget or sensor layers")
    files = {key: path.parent / raw[key] for key in file_keys}
    config = {**{k: v for k, v in raw.items() if k not in files},
              "policy": files["policy_file"].read_text()}
    if "workload_file" in files:
        from .document_workload import load_workload
        return {**config, "documents": {}, "readable_sources": [],
                "workload": load_workload(files["workload_file"])}
    grants = read(files["read_permissions_file"])
    if not isinstance(grants, dict) or set(grants) != {"readable_sources"}:
        raise ValueError("Expected host readable_sources configuration")
    from .read_permissions import validate_sources
    documents = read(files["documents_file"])
    validate_sources(documents, grants["readable_sources"])
    return {**config, "documents": documents, "readable_sources": grants["readable_sources"]}


def _build_firewall(config):
    from .read_permissions import ReadPermissionAdapter
    from .runtime import Firewall
    adapter = ReadPermissionAdapter(
        config["model"], revision=config["revision"], local_files_only=True,
        documents=config["documents"], readable_sources=config["readable_sources"],
        policy=config["policy"], layers=config["layers"], max_context=config["max_context"], device="cpu")
    if "workload" in config:
        from .document_workload import DocumentQAFirewall
        return DocumentQAFirewall(adapter, config["workload"], timeout_seconds=config["timeout_seconds"])
    return Firewall(adapter, mode="permissions", timeout_seconds=config["timeout_seconds"])


def _worker(connection, factory, config):
    try:
        firewall = factory(config)
        connection.send({"ready": True})
        while True:
            task, context, horizon = connection.recv()
            result = firewall.run(task, context, max_new_tokens=horizon)
            connection.send(result)
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        # Never serialize exceptions that may include prompt text or secrets.
        try:
            connection.send({"ready": False})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class IsolatedFirewall:
    mode = "permissions"

    def __init__(self, config, *, factory=_build_firewall):
        self.config = json.loads(json.dumps(config, allow_nan=False))
        for key in ("timeout_seconds", "startup_timeout_seconds"):
            value = self.config[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid process deadline")
        if type(self.config["max_new_tokens"]) is not int or not 1 <= self.config["max_new_tokens"] <= 32768:
            raise ValueError("Invalid generation budget")
        self.configuration_sha256 = digest(self.config)
        self._factory = factory
        self._lock = threading.Lock()
        self._process = self._connection = None
        self._ready = False

    @property
    def ready(self):
        return bool(self._ready and self._process and self._process.is_alive())

    def start(self):
        with self._lock:
            self.close()
            if digest(self.config) != self.configuration_sha256:
                raise ValueError("Host configuration changed; construct a new service")
            ctx = multiprocessing.get_context("spawn")
            self._connection, child = ctx.Pipe()
            self._process = ctx.Process(target=_worker, args=(child, self._factory, self.config), daemon=True)
            try:
                self._process.start()
                child.close()
                if not self._connection.poll(self.config["startup_timeout_seconds"]):
                    raise TimeoutError("Worker initialization timed out")
                if self._connection.recv() != {"ready": True}:
                    raise RuntimeError("Worker initialization failed")
                self._ready = True
            except Exception:
                child.close()
                self.close()
                raise

    def close(self):
        self._ready = False
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(0.5)
                if self._process.is_alive():
                    self._process.kill()
            self._process.join(0.5)
            self._process.close()
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def run(self, task, context="", *, max_new_tokens=None):
        result = {"status": "error", "output": None, "mode": self.mode, "enforced": True,
                  "reason": "worker_unavailable", "request_id": secrets.token_hex(12),
                  "configuration_sha256": self.configuration_sha256}
        started = time.monotonic()
        def reject(reason):
            result.update(status="error", output=None, reason=reason)
            return result
        if not self._lock.acquire(blocking=False):
            return reject("worker_busy")
        try:
            if (not isinstance(task, str) or not task.strip() or len(task) > 16000 or context != ""
                    or max_new_tokens != self.config["max_new_tokens"]):
                return reject("invalid_request")
            if digest(self.config) != self.configuration_sha256:
                self.close()
                return reject("configuration_changed")
            if not self.ready:
                return result
            self._connection.send((task, context, max_new_tokens))
            if not self._connection.poll(self.config["timeout_seconds"]):
                self.close()
                return reject("generation_timeout")
            response = self._connection.recv()
            if (not isinstance(response, dict) or response.get("status") not in ("allowed", "blocked", "incomplete", "error")
                    or (response["status"] == "allowed" and not isinstance(response.get("output"), str))
                    or (response["status"] != "allowed" and response.get("output") is not None)):
                raise ValueError("Invalid worker response")
            if "workload" in self.config:
                from .document_workload import select_records, source_references
                expected = source_references(select_records(self.config["workload"], task))
                if response.get("sources") != expected:
                    raise ValueError("Worker source provenance mismatch")
                result["sources"] = expected
            result.update({key: response[key] for key in ("status", "output", "reason", "observed_steps") if key in response})
            return result
        except (OSError, EOFError, ValueError):
            self.close()
            return reject("worker_failure")
        finally:
            self._lock.release()
            # No task, document, token, output, credential or exception text.
            logging.getLogger(__name__).info(json.dumps({
                "request_id": result["request_id"], "configuration_sha256": self.configuration_sha256,
                "status": result["status"], "reason": result["reason"],
                "elapsed_seconds": round(time.monotonic() - started, 4)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    token = load_token(args.token_file)
    service = IsolatedFirewall(load_config(args.config))
    # Bind/authenticate before costly model initialization; still loopback only.
    server = make_server(service, service.config["max_new_tokens"], port=args.port, auth_token=token)
    try:
        service.start()
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        print(json.dumps({"status": "ready", "port": server.server_port,
                          "configuration_sha256": service.configuration_sha256}), flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
