import os
from pathlib import Path

from huggingface_hub import HfApi


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'").strip('"')
    return out


def _get_hf_token() -> str | None:
    # Prefer environment variables first (works in CI and terminals)
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_TOKEN"):
        v = os.getenv(k)
        if v:
            return v

    # Fall back to project .env (local dev convenience)
    env = _read_dotenv(Path(".env"))
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_TOKEN"):
        v = env.get(k)
        if v:
            return v
    return None


def main() -> None:
    token = _get_hf_token()
    if not token:
        raise SystemExit(
            "Missing Hugging Face token. Set HF_TOKEN (preferred) in your shell or in .env."
        )

    api = HfApi(token=token)
    info = api.whoami()
    username = info.get("name") or info.get("fullname") or info.get("email") or str(info)
    print(f"✅ Hugging Face token OK. Logged in as: {username}")


if __name__ == "__main__":
    main()

