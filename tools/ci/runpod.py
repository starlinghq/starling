"""RunPod control for the scheduled CUDA qualification job (.github/workflows/qualification.yml, job `cuda`).

Runs on the GitHub runner only. Reads the API key from the RUNPOD_API_KEY environment variable and never prints it,
never writes it to a file and never sends it anywhere but api.runpod.io / rest.runpod.io. It only ever creates,
inspects or terminates pods whose name starts with PREFIX; every other pod of the account is left alone.

  runpod.py preflight --min-balance <usd>      balance check; terminates stale PREFIX pods of earlier runs
  runpod.py create --name <n> --public-key <file> --backstop-minutes <m>
                                               creates the pod; prints id, GPU and hourly cost (GITHUB_OUTPUT pod_id)
  runpod.py ssh --name <n> [--wait-minutes <m>] prints "<ip> <port>" once public SSH is mapped
  runpod.py terminate --name <n>               terminates the pod, confirms through the API that it is gone, and
                                               logs its wall time and cost
  runpod.py check                              exit 1 if any PREFIX pod exists (prints names and ids)

The pod is created with the network volume that holds the .NET SDK, uv, the venvs and the verified models
(starling-q-vol, EU-CZ-1), on the image the volume was prepared for, on demand (secure cloud), on an RTX 3090 only.
The CUDA support contracts cover the RTX 3090 (Ampere), where every CUDA gate ran. No other GPU is ever a fallback:
when no RTX 3090 is in stock the job fails, and a pod that reports another GPU is terminated and fails the job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PREFIX = "starling-q-ci-"
VOLUME_ID = "mlav90o7cc"  # starling-q-vol
DATACENTER = "EU-CZ-1"
IMAGE = "runpod/pytorch:1.4.0-cu1300-torch291-ubuntu2404"
# The only GPU type the CUDA contracts cover. Run 38857860186 landed on an RTX 4090, where
# embeddinggemma-2/video-fp32-cuda failed G2 (docs/evidence/gpu-regression-2026-10-16/README.md). Never add a fallback.
GPU = "NVIDIA GeForce RTX 3090"
GPUS = [GPU]
CONTAINER_DISK_GB = 120


def _key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("error: RUNPOD_API_KEY is not set")
    return key


def _request(req: urllib.request.Request) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed https hosts)
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:1000]
            if e.code >= 500 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}") from None
        except urllib.error.URLError as e:
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"network error: {e.reason}") from None
    raise RuntimeError("unreachable")


def _headers() -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {_key()}", "User-Agent": "starling-qualification-ci"}


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request("https://api.runpod.io/graphql", data=body, headers=_headers())
    d = json.loads(_request(req))
    if d.get("errors"):
        raise RuntimeError(json.dumps(d["errors"])[:1000])
    return d["data"]


def rest(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request("https://rest.runpod.io/v1" + path, data=data, method=method, headers=_headers())
    txt = _request(req).decode()
    return json.loads(txt) if txt.strip() else {}


def account() -> dict:
    return gql(
        "query { myself { clientBalance currentSpendPerHr "
        "pods { id name desiredStatus costPerHr machine { gpuDisplayName } } } }"
    )["myself"]


def ours(acct: dict | None = None) -> list[dict]:
    return [p for p in (acct or account())["pods"] if p["name"].startswith(PREFIX)]


def output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def delete(pod: dict) -> None:
    assert pod["name"].startswith(PREFIX)
    rest("DELETE", f"/pods/{pod['id']}")


def confirm_gone(pod_id: str, minutes: float = 5) -> bool:
    end = time.time() + minutes * 60
    while time.time() < end:
        if all(p["id"] != pod_id for p in account()["pods"]):
            return True
        time.sleep(10)
    return False


def cmd_preflight(a: argparse.Namespace) -> int:
    acct = account()
    bal = float(acct["clientBalance"])
    print(f"RunPod balance ${bal:.2f}, account spend ${float(acct['currentSpendPerHr']):.3f}/h")
    for p in ours(acct):
        print(f"stale pod of an earlier run: {p['name']} {p['id']} ({p['desiredStatus']}); terminating")
        delete(p)
        print("  gone" if confirm_gone(p["id"]) else "  STILL PRESENT")
    if bal < a.min_balance:
        print(f"::error title=RunPod balance too low::${bal:.2f} is below the ${a.min_balance:.2f} floor; not starting a pod")
        return 1
    output("balance", f"{bal:.4f}")
    return 0


def cmd_create(a: argparse.Namespace) -> int:
    assert a.name.startswith(PREFIX)
    if any(p["name"] == a.name for p in ours()):
        print(f"error: {a.name} already exists")
        return 1
    with open(a.public_key, encoding="utf-8") as f:
        pub = f.read().strip()
    # Pod-side backstop: the container removes its own pod after --backstop-minutes, even if this runner is lost.
    # /start.sh is the image's own entry (sshd with PUBLIC_KEY). runpodctl and RUNPOD_POD_ID come with the pod.
    backstop = (
        f"(sleep {int(a.backstop_minutes) * 60}; runpodctl remove pod \"$RUNPOD_POD_ID\" || runpodctl stop pod \"$RUNPOD_POD_ID\") "
        ">/var/log/starling-backstop.log 2>&1 &"
    )
    body = {
        "name": a.name,
        "imageName": IMAGE,
        "gpuTypeIds": GPUS,
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "cloudType": "SECURE",  # on demand (not spot): the default when "interruptible" is not set
        "dataCenterIds": [DATACENTER],
        "networkVolumeId": VOLUME_ID,
        "volumeMountPath": "/workspace",
        "containerDiskInGb": CONTAINER_DISK_GB,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": pub},
        "allowedCudaVersions": ["13.0"],
        "dockerStartCmd": ["bash", "-c", backstop + " exec /start.sh"],
    }
    # The volume pins the datacenter, so a GPU may be out of stock there for a while: retry until --stock-wait-minutes.
    end = time.time() + a.stock_wait_minutes * 60
    while True:
        try:
            p = rest("POST", "/pods", body)
            break
        except RuntimeError as e:
            msg = str(e)
            if "no instances currently available" in msg and time.time() < end:
                print(f"{time.strftime('%H:%M:%S')} no RTX 3090 available in {DATACENTER}; retrying in 60 s", flush=True)
                time.sleep(60)
                continue
            if "HTTP 4" in msg and "dockerStartCmd" in body:
                # The workflow arms the same backstop over SSH when the start command is not accepted.
                print(f"::warning title=RunPod start command refused::{msg[:300]}; creating without it")
                del body["dockerStartCmd"]
                continue
            if "no instances currently available" in msg:
                print(
                    f"::error title=No RTX 3090 in stock::no RTX 3090 in {DATACENTER} for {a.stock_wait_minutes} min. "
                    "The CUDA contracts cover only the RTX 3090, so no other GPU is used; rerun the job later."
                )
            raise
    pod_id = p.get("id")
    if not pod_id:
        print("error: create returned no id: " + json.dumps({k: p.get(k) for k in ("name", "desiredStatus")}))
        return 1
    output("pod_id", pod_id)
    output("created_at", str(int(time.time())))
    gpu = ((p.get("machine") or {}).get("gpuDisplayName")) or (p.get("gpu") or {}).get("displayName") or "?"
    print(f"pod {a.name} id {pod_id}, GPU {gpu}, ${float(p.get('costPerHr') or 0):.3f}/h, backstop {a.backstop_minutes} min")
    if gpu != "?" and not is_required_gpu(gpu):
        wrong_gpu(gpu)
        delete({"name": a.name, "id": pod_id})
        print(f"pod {a.name} terminated: {'confirmed gone' if confirm_gone(pod_id) else 'STILL LISTED'}")
        return 1
    return 0


def is_required_gpu(name: str) -> bool:
    """True for the RTX 3090, whichever of RunPod's names for it is reported ("RTX 3090", "NVIDIA GeForce RTX 3090")."""
    return name.strip().removeprefix("NVIDIA ").removeprefix("GeForce ") == GPU.removeprefix("NVIDIA GeForce ")


def wrong_gpu(name: str, file=sys.stdout) -> None:
    print(
        f"::error title=Wrong GPU::the pod reports {name}, not the {GPU} the CUDA contracts cover; "
        "the run is refused rather than gathered on another GPU",
        file=file,
    )


def _pod(name: str) -> dict | None:
    return next((p for p in ours() if p["name"] == name), None)


def cmd_ssh(a: argparse.Namespace) -> int:
    end = time.time() + a.wait_minutes * 60
    while time.time() < end:
        p = _pod(a.name)
        if p is None:
            print("error: pod not found", file=sys.stderr)
            return 1
        d = gql(
            "query($id:String!){ pod(input:{podId:$id}) { costPerHr machine { gpuDisplayName } "
            "runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } } } }",
            {"id": p["id"]},
        )["pod"] or {}
        for port in (d.get("runtime") or {}).get("ports") or []:
            if port["privatePort"] == 22 and port["isIpPublic"]:
                gpu = (d.get("machine") or {}).get("gpuDisplayName", "?")
                print(f"pod {a.name} id {p['id']} on {gpu}, ${float(d.get('costPerHr') or 0):.3f}/h", file=sys.stderr)
                if not is_required_gpu(gpu):
                    # The workflow's terminate step (always run) removes the pod.
                    wrong_gpu(gpu, file=sys.stderr)  # stdout carries the endpoint
                    return 1
                output("gpu", gpu)
                output("cost_per_hr", str(d.get("costPerHr") or 0))
                print(port["ip"], port["publicPort"])
                return 0
        time.sleep(15)
    print("error: no public SSH port within the wait", file=sys.stderr)
    return 1


def cmd_terminate(a: argparse.Namespace) -> int:
    p = _pod(a.name)
    if p is None:
        print(f"{a.name}: no such pod (never created, or already gone)")
        rc = 0
    else:
        rate = float(p.get("costPerHr") or 0)
        gpu = (p.get("machine") or {}).get("gpuDisplayName", "?")
        delete(p)
        gone = confirm_gone(p["id"])
        line = f"pod {a.name} id {p['id']} on {gpu} terminated: {'confirmed gone' if gone else 'STILL LISTED'}"
        if a.created_at:
            hours = (time.time() - int(a.created_at)) / 3600
            line += f"; {hours * 60:.0f} min at ${rate:.3f}/h = about ${hours * rate:.2f}"
        print(line)
        summary(f"- RunPod: {line}")
        rc = 0 if gone else 1
    left = ours()
    if left:
        print("::error title=RunPod pods left::" + ", ".join(f"{q['name']} {q['id']}" for q in left))
        return 1
    bal = float(account()["clientBalance"])
    print(f"no {PREFIX}* pod remains; balance ${bal:.2f}")
    summary(f"- RunPod: no `{PREFIX}*` pod remains; balance ${bal:.2f}")
    return rc


def cmd_check(_: argparse.Namespace) -> int:
    left = ours()
    for q in left:
        print(q["name"], q["id"], q["desiredStatus"])
    print(f"{len(left)} {PREFIX}* pod(s)")
    return 1 if left else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("preflight")
    s.add_argument("--min-balance", type=float, default=5.0)
    s = sub.add_parser("create")
    s.add_argument("--name", required=True)
    s.add_argument("--public-key", required=True)
    s.add_argument("--backstop-minutes", type=int, required=True)
    s.add_argument("--stock-wait-minutes", type=float, default=0)
    s = sub.add_parser("ssh")
    s.add_argument("--name", required=True)
    s.add_argument("--wait-minutes", type=float, default=15)
    s = sub.add_parser("terminate")
    s.add_argument("--name", required=True)
    s.add_argument("--created-at", default="")
    sub.add_parser("check")
    a = ap.parse_args()
    return {"preflight": cmd_preflight, "create": cmd_create, "ssh": cmd_ssh, "terminate": cmd_terminate, "check": cmd_check}[
        a.cmd
    ](a)


if __name__ == "__main__":
    sys.exit(main())
