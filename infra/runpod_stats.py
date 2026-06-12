"""Pull live hardware info / utilization for the current RunPod pod.

Hits RunPod's GraphQL API and returns everything it exposes about the pod:
static config (disk/volume sizes, GPU type, cost) and live runtime telemetry
(GPU util %, VRAM util %, CPU %, system-RAM %, uptime).

Credentials come from the environment (RunPod sets ``RUNPOD_POD_ID`` inside the
pod automatically; you must supply ``RUNPOD_API_KEY`` yourself):

    export RUNPOD_API_KEY=...        # from runpod.io settings
    uv run python infra/runpod_stats.py    # pretty-prints the full info blob

The ``runtime`` block is ``None`` while the pod is stopped/starting — that is
RunPod's behavior, not an error.
"""

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

GRAPHQL_ENDPOINT = "https://api.runpod.io/graphql"

# Everything the pod query exposes: static config + live runtime telemetry.
POD_QUERY = """
query Pod($podId: String!) {
  pod(input: {podId: $podId}) {
    id
    name
    desiredStatus
    costPerHr
    memoryInGb
    vcpuCount
    gpuCount
    containerDiskInGb
    volumeInGb
    volumeMountPath
    machine {
      gpuDisplayName
    }
    runtime {
      uptimeInSeconds
      gpus {
        id
        gpuUtilPercent
        memoryUtilPercent
      }
      container {
        cpuPercent
        memoryPercent
      }
      ports {
        ip
        isIpPublic
        privatePort
        publicPort
        type
      }
    }
  }
}
"""


def fetch_pod_info(pod_id: str, api_key: str, timeout: float = 10.0) -> dict:
    """Return the full RunPod info blob for ``pod_id`` as a dict.

    Raises on missing credentials, transport errors, or a GraphQL error so
    problems surface immediately rather than as a silent empty result.
    """
    if not pod_id:
        raise ValueError("pod_id is empty — set RUNPOD_POD_ID or pass it explicitly.")
    if not api_key:
        raise ValueError("api_key is empty — set RUNPOD_API_KEY.")

    payload = json.dumps(
        {"query": POD_QUERY, "variables": {"podId": pod_id}}
    ).encode("utf-8")
    request = urllib.request.Request(
        GRAPHQL_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Cloudflare (in front of the API) blocks the default urllib
            # User-Agent with "error code: 1010", so send a normal one.
            "User-Agent": "runpod-stats/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"RunPod API HTTP {error.code}: {detail}") from error

    if "errors" in body:
        raise RuntimeError(f"RunPod GraphQL error: {body['errors']}")

    pod = body.get("data", {}).get("pod")
    if pod is None:
        raise RuntimeError(f"No pod returned for id {pod_id!r} (wrong id or API key?).")

    runtime = pod["runtime"]
    gpu = runtime["gpus"][0]
    return {
        "gpuDisplayName": pod["machine"]["gpuDisplayName"],
        "gpuUtilPercent": gpu["gpuUtilPercent"],
        "memoryUtilPercent": gpu["memoryUtilPercent"],
        "cpuPercent": runtime["container"]["cpuPercent"],
        "memoryPercent": runtime["container"]["memoryPercent"],
        "memoryInGb": pod["memoryInGb"],
        "vcpuCount": pod["vcpuCount"],
        "gpuCount": pod["gpuCount"],
    }


def log_pod_utilization(step: int) -> None:
    """Log live RunPod hardware util — best-effort, never crashes training."""
    try:
        info = fetch_pod_info(
            pod_id=os.environ.get("RUNPOD_POD_ID", ""),
            api_key=os.environ.get("RUNPOD_API_KEY", ""),
        )
    except Exception as exc:
        logger.warning("step %d | pod stats unavailable: %s", step, exc)
        return
    logger.info(
        "step %d | %s | GPU %d%% | VRAM %d%% | CPU %d%% | RAM %d%%",
        step,
        info["gpuDisplayName"],
        info["gpuUtilPercent"],
        info["memoryUtilPercent"],
        info["cpuPercent"],
        info["memoryPercent"],
    )


