"""Inference-only V1B adapter sidecar for the INTERNAL DEMO profile.

This process runs on the GPU host, inside the same virtual environment that the sealed training
and evaluation runs used, and serves ONE thing: greedy text generation from the frozen base
checkpoint with the owner-selected V1B LoRA adapter attached. It is the minimal adapter-aware
connection the repository's inference abstraction was missing: the existing vLLM backend is an
explicit stub, and the Ollama backend serves GGUF models, which this PEFT safetensors adapter is
not — converting it is forbidden.

What it refuses to do, by construction:

- it never merges, quantizes, converts, overwrites or retrains the adapter: the adapter file is
  opened read-only, hashed, and attached through ``PeftModel.from_pretrained`` with
  ``is_trainable=False``;
- it never starts unless the measured adapter digest equals the digest passed on the command line
  and the tokenizer's chat-template digest equals the one the sealed evaluation recorded, so a
  substituted adapter or a drifted template stops the process instead of answering;
- it binds a loopback address only. Reaching it from another machine requires an SSH tunnel, which
  is the project's existing private connection; nothing is published to a network interface;
- it serves exactly two paths and no files.

Dependencies: the Python standard library plus torch, transformers and peft, which are already
installed in the sealed-lineage virtual environment. Nothing is installed, upgraded or removed on
the host, no container is started, no cache is cleared, and no other process is touched.

Execution environment is intentionally not supplied in this review copy.
The original introductory usage example was replaced to avoid distributing internal run,
adapter, log and process paths. Runtime implementation below is unchanged.
Required command-line arguments are documented in main(); model and adapter files are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: The chat-template digest the sealed evaluation measured and pinned. A different template would
#: silently change every prompt, so a mismatch is a startup failure, not a warning.
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
#: The frozen decoding identity of the sealed evaluation, reused verbatim for the demo.
EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643
REPETITION_PENALTY = 1.0
ENABLE_THINKING = False
#: Hard ceiling on a single demo answer, regardless of what a caller asks for.
MAX_NEW_TOKENS_CEILING = 512
#: Hard ceiling on a request body, so an oversized post cannot exhaust host memory.
MAX_REQUEST_BYTES = 256 * 1024
#: Bounds on the prompt itself. The body cap alone is not a prompt cap: a single valid 256 KiB body
#: is tens of thousands of tokens, whose KV cache would dwarf the weights and block the one
#: generation lock for minutes. The demo's own prompts are a question of at most 2000 characters
#: plus at most three short records, so these ceilings are far above it and still bounded.
MAX_PROMPT_CHARS = 24 * 1024
MAX_MESSAGES = 8
#: The three tensors the sealed evaluation fingerprinted, reused here as a weight witness.
FINGERPRINT_KEYS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.35.mlp.down_proj.weight",
    "lm_head.weight",
)
ROLES = ("system", "user", "assistant")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )


class Engine:
    """The loaded model, its verified identity, and one serialized generate call."""

    def __init__(self, args: argparse.Namespace) -> None:
        # Offline by construction rather than by the operator remembering an environment prefix:
        # the base checkpoint and tokenizer are read from the host's existing local cache and
        # nothing is fetched from a model hub. Set before transformers is imported, which is what
        # these two variables require, and never overriding a value the operator chose.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._lock = threading.Lock()
        self._served = 0
        self._started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._max_new_tokens = min(int(args.max_new_tokens), MAX_NEW_TOKENS_CEILING)

        adapter_dir = os.path.expanduser(args.adapter)
        adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.isfile(adapter_file):
            raise SystemExit(f"STOP adapter_file_missing:{adapter_file}")
        measured_adapter = sha_file(adapter_file)
        if measured_adapter != args.expect_adapter_sha:
            raise SystemExit(f"STOP adapter_digest_mismatch:{measured_adapter}")
        with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as handle:
            adapter_config = json.load(handle)

        self._tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, revision=args.revision, local_files_only=True
        )
        template = getattr(self._tokenizer, "chat_template", None)
        if not template:
            raise SystemExit("STOP chat_template_absent")
        measured_template = sha_text(template)
        if measured_template != args.chat_template_sha:
            raise SystemExit(f"STOP chat_template_digest_mismatch:{measured_template}")

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            revision=args.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=None,
            local_files_only=True,
        )
        model.to("cuda")
        model.eval()
        fingerprint = self._fingerprint(model)

        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        model.eval()
        active = getattr(model, "active_adapters", None)
        self._model = model
        after = self._fingerprint(model)
        if after != fingerprint:
            # Attaching a LoRA adapter must not change the base weights; if it did, something
            # merged into them and this process must not serve.
            raise SystemExit("STOP base_weights_changed_after_adapter_attach")

        import peft
        import transformers

        self.identity = {
            "role": "v1b_internal_demo_sidecar",
            "base_model_id": args.base_model,
            "base_model_revision": args.revision,
            "adapter_dir_basename": os.path.basename(adapter_dir.rstrip("/\\")),
            "adapter_sha256_measured": measured_adapter,
            "adapter_r": adapter_config.get("r"),
            "adapter_alpha": adapter_config.get("lora_alpha"),
            "adapter_target_modules": sorted(adapter_config.get("target_modules") or []),
            "adapter_merged_into_base_weights": False,
            "adapter_is_trainable": False,
            "active_adapters": list(active() if callable(active) else (active or [])),
            "base_weight_fingerprint": fingerprint,
            "chat_template_sha256_measured": measured_template,
            "tokenizer_class": type(self._tokenizer).__name__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": REPETITION_PENALTY,
                "eos_token_id": EOS_TOKEN_ID,
                "pad_token_id": PAD_TOKEN_ID,
                "enable_thinking": ENABLE_THINKING,
                "max_new_tokens_default": self._max_new_tokens,
            },
            "inference_only": True,
            "training_executions": 0,
        }
        self.identity_sha256 = hashlib.sha256(canonical_json(self.identity)).hexdigest()

    def _fingerprint(self, model: object) -> list[list[object]]:
        """Sum the three witness tensors, resolving the names PEFT rewrites.

        Attaching a LoRA adapter renames a wrapped module's weight from
        ``...q_proj.weight`` to ``base_model.model....q_proj.base_layer.weight``, so the lookup
        matches on the suffix with that wrapper segment removed. The tensors are the same objects
        before and after attachment, which is what the caller is checking.
        """
        out: list[list[object]] = []
        resolved = {
            name.replace("base_layer.", ""): tensor for name, tensor in model.named_parameters()
        }
        for key in FINGERPRINT_KEYS:
            match = next(
                (tensor for name, tensor in resolved.items() if name.endswith(key)),
                None,
            )
            if match is None:
                raise SystemExit(f"STOP fingerprint_tensor_not_found:{key}")
            out.append([key, float(match.float().sum().item()), list(match.shape)])
        return out

    def status(self) -> dict[str, object]:
        return {
            "status": "ready",
            "started_utc": self._started_utc,
            "generations_served": self._served,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
        }

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> dict[str, object]:
        torch = self._torch
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        requested = max(1, min(int(max_new_tokens), self._max_new_tokens))
        started = time.monotonic()
        with self._lock:
            prompt = torch.tensor([prompt_ids], device="cuda")
            with torch.inference_mode():
                out = self._model.generate(
                    input_ids=prompt,
                    attention_mask=torch.ones_like(prompt),
                    max_new_tokens=requested,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=REPETITION_PENALTY,
                    eos_token_id=EOS_TOKEN_ID,
                    pad_token_id=PAD_TOKEN_ID,
                    use_cache=True,
                )
            new_ids = out[0][prompt.shape[1] :].tolist()
            del out, prompt
            self._served += 1
        finished_on_eos = bool(new_ids) and new_ids[-1] == EOS_TOKEN_ID
        stripped = new_ids[:-1] if finished_on_eos else list(new_ids)
        text = self._tokenizer.decode(stripped, skip_special_tokens=True)
        return {
            "answer_text": text,
            "prompt_sha256": sha_text(prompt_text),
            "prompt_tokens": len(prompt_ids),
            "generated_token_count": len(stripped),
            "finish_reason": "eos" if finished_on_eos else "length",
            "max_new_tokens": requested,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "identity_sha256": self.identity_sha256,
        }


def make_handler(
    engine: Engine, log_path: str | None, log_kind: str = "DEVELOPMENT_TESTING"
) -> type[BaseHTTPRequestHandler]:
    log_lock = threading.Lock()

    def record(entry: dict[str, object]) -> None:
        if not log_path:
            return
        with log_lock:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    class Handler(BaseHTTPRequestHandler):
        server_version = "v1b-demo-sidecar/1"
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, payload: dict[str, object]) -> None:
            body = canonical_json(payload)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, engine.status())
                return
            self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/generate":
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                # The declared body is never read on these two paths, so the connection must not be
                # reused: leftover bytes would otherwise be parsed as the next request line.
                self.close_connection = True
                self._send(400, {"error": "bad_content_length"})
                return
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self.close_connection = True
                self._send(413, {"error": "body_too_large_or_empty"})
                return
            raw = self.rfile.read(length)
            try:
                request = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                self._send(400, {"error": "body_is_not_json"})
                return
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send(400, {"error": "messages_required"})
                return
            if len(messages) > MAX_MESSAGES:
                self._send(400, {"error": "too_many_messages"})
                return
            cleaned: list[dict[str, str]] = []
            for message in messages:
                if not isinstance(message, dict):
                    self._send(400, {"error": "message_not_an_object"})
                    return
                role = message.get("role")
                content = message.get("content")
                if role not in ROLES or not isinstance(content, str) or not content.strip():
                    self._send(400, {"error": "message_role_or_content_invalid"})
                    return
                cleaned.append({"role": role, "content": content})
            if sum(len(message["content"]) for message in cleaned) > MAX_PROMPT_CHARS:
                self._send(400, {"error": "prompt_too_large"})
                return
            requested = request.get(
                "max_new_tokens", engine.identity["decoding"]["max_new_tokens_default"]
            )
            if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
                self._send(400, {"error": "max_new_tokens_invalid"})
                return
            try:
                result = engine.generate(cleaned, requested)
            except Exception as exc:
                record(
                    {
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "kind": log_kind,
                        "outcome": "error",
                        "error_type": type(exc).__name__,
                        "messages": cleaned,
                    }
                )
                self._send(500, {"error": "generation_failed", "error_type": type(exc).__name__})
                return
            record(
                {
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "kind": log_kind,
                    "outcome": "ok",
                    "messages": cleaned,
                    "result": result,
                }
            )
            self._send(200, result)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inference-only V1B adapter sidecar")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--expect-adapter-sha", required=True)
    parser.add_argument("--chat-template-sha", default=CHAT_TEMPLATE_SHA256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--log", default=None)
    parser.add_argument(
        "--log-kind",
        default="DEVELOPMENT_TESTING",
        help="provenance label written on every logged request; the default marks development use",
    )
    parser.add_argument("--pidfile", default=None)
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("STOP the sidecar binds loopback only; use an SSH tunnel for remote use")

    engine = Engine(args)
    log_path = os.path.expanduser(args.log) if args.log else None
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(engine, log_path, args.log_kind)
    )
    server.daemon_threads = True

    if args.pidfile:
        pidfile = os.path.expanduser(args.pidfile)
        os.makedirs(os.path.dirname(pidfile) or ".", exist_ok=True)
        with open(pidfile, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()) + "\n")

    def shutdown(signum: int, frame: object) -> None:
        del signum, frame
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    measured = engine.identity["adapter_sha256_measured"]
    print(
        f"SIDECAR_READY host={args.host} port={args.port}"
        f" adapter_sha256={measured} identity_sha256={engine.identity_sha256}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        if args.pidfile and os.path.exists(os.path.expanduser(args.pidfile)):
            os.remove(os.path.expanduser(args.pidfile))
        served = engine.status()["generations_served"]
        print(f"SIDECAR_STOPPED served={served}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
