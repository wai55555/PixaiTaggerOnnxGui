"""プロバイダーから利用可能なモデル ID の一覧を取得する（260901_VLM_implement_plan 2.2）。

出荷時の推定モデル ID を人手で直させるのではなく、各サービスの「モデル一覧」API を
叩いて実在する ID から選ばせるための層。UI スレッドからは呼ばない（ワーカー経由）。

- OpenAI 互換（openai_chat_completions）: GET {base_url}/models -> {"data":[{"id": ...}]}
- Gemini（gemini_generate_content）  : GET {base_url}/models -> {"models":[{"name":"models/..."}]}
"""
from __future__ import annotations

from vlm_connections import VlmConnection
from vlm_errors import VlmAttemptError, VlmErrorReason
from vlm_protocols import VlmHttpRequest, apply_connection_auth, default_auth_key
from vlm_transport import RawHttpResponse, execute_http


def fetch_model_ids(conn: VlmConnection, api_key: str | None,
                    *, connect_timeout: float = 8.0, read_timeout: float = 20.0
                    ) -> list[str] | VlmAttemptError:
    """接続先のモデル一覧を取り、モデル ID の文字列リストを返す。失敗時は VlmAttemptError。"""
    base = (conn.base_url or "").rstrip("/")
    if not base or "{account_id}" in base:
        return VlmAttemptError(VlmErrorReason.UNKNOWN, None,
                               "connection base url is not ready (Cloudflare account id?)")
    # Cloudflare Workers AI はモデル一覧が別パス（/ai/models/search）。
    if conn.provider_id == "cloudflare" and base.endswith("/ai/v1"):
        url = base[: -len("/v1")] + "/models/search"
    else:
        url = f"{base}/models"

    req = VlmHttpRequest(method="GET", url=url, headers={}, params={}, json_body={})
    default_key = default_auth_key(conn.auth.type, api_key)
    if default_key:
        req.headers["Authorization"] = f"Bearer {default_key}"
    apply_connection_auth(req, conn.auth.type, api_key, conn.auth.header_name, conn.auth.query_param)

    raw = execute_http(req, connect_timeout=connect_timeout, read_timeout=read_timeout,
                       verify_tls=conn.verify_tls)
    if not isinstance(raw, RawHttpResponse):
        return raw
    if raw.status in (401, 403):
        return VlmAttemptError(VlmErrorReason.AUTH_ERROR, raw.status, f"{raw.status} auth rejected")
    if raw.status != 200 or not isinstance(raw.json_body, (dict, list)):
        return VlmAttemptError(VlmErrorReason.BAD_RESPONSE, raw.status,
                               f"HTTP {raw.status} (model list unavailable)")

    ids = _extract_ids(raw.json_body)
    if not ids:
        return VlmAttemptError(VlmErrorReason.BAD_RESPONSE, raw.status, "no model ids in response")
    # 重複除去しつつ順序維持。
    seen: set[str] = set()
    out: list[str] = []
    for mid in ids:
        if mid and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def _extract_ids(body) -> list[str]:
    # OpenAI 互換: {"data": [{"id": "..."}]} もしくは素の list
    rows = body.get("data") if isinstance(body, dict) else body
    if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "id" in rows[0]:
        return [str(r.get("id", "")).strip() for r in rows if r.get("id")]
    # Cloudflare Workers AI: {"result": [{"name": "@cf/...", "task": {"name": "..."}}]}
    cf = body.get("result") if isinstance(body, dict) else None
    if isinstance(cf, list) and cf and isinstance(cf[0], dict) and "name" in cf[0]:
        _drop = ("embedding", "classification", "speech", "translation", "detection",
                 "image-to-image", "text-to-image", "text-to-speech", "reranking")
        out = []
        for r in cf:
            task = ((r.get("task") or {}).get("name") or "").lower()
            if task and any(k in task for k in _drop):
                continue
            n = str(r.get("name", "")).strip()
            if n:
                out.append(n)
        return out
    # Gemini: {"models": [{"name": "models/gemma-3-27b-it", "supportedGenerationMethods": [...]}]}
    gm = body.get("models") if isinstance(body, dict) else None
    if isinstance(gm, list):
        out = []
        for r in gm:
            if not isinstance(r, dict):
                continue
            methods = r.get("supportedGenerationMethods") or r.get("supported_generation_methods") or []
            if methods and "generateContent" not in methods:
                continue
            name = str(r.get("name", "")).strip()
            out.append(name[len("models/"):] if name.startswith("models/") else name)
        return [x for x in out if x]
    return []
