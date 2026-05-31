# ─────────────────────────────────────────────────────────
#  utils/n8n_client.py — Kirim payload ke n8n webhook
# ─────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ALERT_THRESHOLD, CLASS_NAMES, N8N_TIMEOUT_SEC, N8N_WEBHOOK_URL


def build_payload(
    image_name: str,
    counts: dict[str, int],
    detections: list[dict],
) -> dict:
    """
    Susun payload JSON yang akan dikirim ke n8n.

    Payload Format:
    {
        "timestamp"       : "2026-05-24T10:30:00+07:00",
        "image_name"      : "traffic_001.jpg",
        "detections"      : {"car": 4, "van": 2, "bus": 1},
        "total_vehicles"  : 7,
        "avg_confidence"  : 0.87,
        "alert"           : false,
        "alert_message"   : ""
    }
    """
    total = sum(counts.values())
    avg_conf = (
        round(
            sum(d["confidence"] for d in detections) / len(detections), 3
        )
        if detections
        else 0.0
    )

    alert      = total > ALERT_THRESHOLD
    alert_msg  = (
        f"🚨 HIGH TRAFFIC: {total} kendaraan terdeteksi di {image_name}"
        if alert
        else ""
    )

    return {
        "timestamp"      : datetime.now().astimezone().isoformat(),
        "image_name"     : image_name,
        "detections"     : counts,
        "total_vehicles" : total,
        "avg_confidence" : avg_conf,
        "alert"          : alert,
        "alert_message"  : alert_msg,
    }


def send_to_n8n(
    image_name: str,
    counts: dict[str, int],
    detections: list[dict],
) -> tuple[bool, str]:
    """
    POST payload ke n8n webhook.

    Returns:
        (success: bool, message: str)
    """
    payload = build_payload(image_name, counts, detections)

    try:
        resp = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=N8N_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return True, f"✅ Log terkirim ke n8n (status {resp.status_code})"
    except requests.exceptions.ConnectionError:
        return False, "❌ Tidak dapat terhubung ke n8n — pastikan server n8n sedang berjalan."
    except requests.exceptions.Timeout:
        return False, f"⏱️ n8n tidak merespons dalam {N8N_TIMEOUT_SEC} detik."
    except requests.exceptions.HTTPError as e:
        return False, f"❌ n8n error: {e}"
    except Exception as e:
        return False, f"❌ Terjadi kesalahan: {e}"
