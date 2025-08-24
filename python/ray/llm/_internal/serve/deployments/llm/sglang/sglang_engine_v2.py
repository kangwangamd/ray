import argparse
import asyncio
import json
import os
import random
import signal
import socket
import subprocess
from typing import AsyncGenerator, Optional, Union

import httpx
from starlette.requests import Request

from ray.llm._internal.serve.configs.openai_api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
)
from ray.llm._internal.serve.configs.server_models import DiskMultiplexConfig, LLMConfig
from ray.llm._internal.serve.deployments.llm.llm_engine import LLMEngine
from ray.llm._internal.serve.observability.logging import get_logger

logger = get_logger(__name__)

def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _kv(d: dict, k: str):
    """Render CLI flag from dict key/val: {'tp_size': 2} -> ['--tp-size','2'] or bool -> ['--flag']"""
    if d is None or k not in d: 
        return []
    v = d[k]
    flag = f"--{k.replace('_','-')}"
    if isinstance(v, bool):
        return [flag] if v else []
    if v is None:
        return []
    return [flag, str(v)]

class SGLangEngine(LLMEngine):
    """
    Ray LLM engine that launches an SGLang OpenAI-compatible server in-process
    and proxies OpenAI requests to it.
    """

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)
        self._running = False
        self._proc: Optional[subprocess.Popen] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._host: str = "127.0.0.1"
        self._port: Optional[int] = None
        self._base_url: Optional[str] = None
        self._api_key: Optional[str] = None  # Optionally set if you want SGLang API key enforcement

    async def start(self) -> None:
        if self._running:
            logger.info("SGLangEngine already running; skip restart.")
            return

        # Pull params from your LLMConfig/engine config.
        # Expect: engine_kwargs like {'model_path': '...', 'server_kwargs': {...}}
        engine_config = self.llm_config.get_engine_config()
        model_path = getattr(engine_config, "actual_hf_model_id", None) or getattr(engine_config, "model_path", None)
        if not model_path:
            raise ValueError("SGLangEngine requires model_path (or actual_hf_model_id) on engine config.")

        server_kwargs = getattr(engine_config, "server_kwargs", {}) or {}
        # If this replica is for embeddings, set --is-embedding in server_kwargs.
        # Example: server_kwargs['is_embedding'] = True

        self._port = int(server_kwargs.get("port") or _get_open_port())
        self._host = server_kwargs.get("host") or "127.0.0.1"
        self._api_key = server_kwargs.get("api_key")  # optional

        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", str(model_path),
            "--host", self._host,
            "--port", str(self._port),
        ]
        # Common useful knobs. Add/rename as you need; these map to SGLang server args.
        for key in [
            "tp_size", "pp_size", "dtype", "quantization", "kv_cache_dtype",
            "mem_fraction_static", "max_total_tokens", "chunked_prefill_size",
            "max_prefill_tokens", "schedule_policy", "device",
            "stream_interval", "enable_metrics", "log_level",
            "served_model_name", "chat_template", "completion_template",
            "is_embedding", "enable_lora", "lora_backend",
            "max_loras_per_batch", "max_lora_rank",
        ]:
            cmd += _kv(server_kwargs, key)

        # LoRA paths: pass as "name=path name2=path2"
        lora_paths = server_kwargs.get("lora_paths")
        if lora_paths:
            # lora_paths may be dict {"name": "/path", ...}
            if isinstance(lora_paths, dict):
                lp = " ".join([f"{k}={v}" for k, v in lora_paths.items()])
            else:
                lp = str(lora_paths)
            cmd += ["--lora-paths", lp]

        env = os.environ.copy()
        # Ray sets CUDA_VISIBLE_DEVICES for the actor; just inherit.

        logger.info(f"Launching SGLang: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Optionally, pipe logs
        async def _pump_logs():
            if not self._proc or not self._proc.stdout:
                return
            loop = asyncio.get_running_loop()
            def _readline():
                return self._proc.stdout.readline()
            while True:
                line = await loop.run_in_executor(None, _readline)
                if not line:
                    break
                logger.info(f"[sglang] {line.rstrip()}")

        asyncio.create_task(_pump_logs())

        self._base_url = f"http://{self._host}:{self._port}"
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=180)

        # Wait for readiness: /get_model_info (and/or /v1/models)
        await self._wait_ready()

        self._running = True
        logger.info(f"SGLang server is ready at {self._base_url}")

    async def _wait_ready(self, timeout_s: int = 300) -> None:
        assert self._client is not None
        deadline = asyncio.get_event_loop().time() + timeout_s
        last_err = None
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await self._client.get("/get_model_info", headers=self._auth_headers())
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e
            await asyncio.sleep(1.0 + random.random() * 0.5)
        raise RuntimeError(f"SGLang server failed readiness check: {last_err}")

    def _auth_headers(self):
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _create_raw_request(self, request_id: str, path: str) -> Request:
        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"x-request-id", request_id.encode() if request_id else b"")],
            "query_string": b"",
        }
        return Request(scope)

    async def _post_openai(
        self, path: str, payload: dict, stream: bool = False
    ) -> Union[dict, AsyncGenerator[str, None]]:
        assert self._client is not None
        if stream:
            # SSE stream passthrough
            async def gen():
                async with self._client.stream("POST", path, json=payload, headers=self._auth_headers()) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line:
                            # Pass through raw "data: ..." lines so upstream
                            # parser behaves like vLLM streaming.
                            yield line
            return gen()
        else:
            resp = await self._client.post(path, json=payload, headers=self._auth_headers())
            # Convert non-200s to the Ray ErrorResponse shape
            if resp.status_code != 200:
                try:
                    err = resp.json()
                except Exception:
                    err = {"error": {"message": resp.text}}
                return {"__error__": err}
            return resp.json()

    async def chat(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[Union[str, ChatCompletionResponse, ErrorResponse], None]:
        payload = request.model_dump()
        is_stream = bool(payload.get("stream"))
        result = await self._post_openai("/v1/chat/completions", payload, stream=is_stream)

        if is_stream:
            assert isinstance(result, AsyncGenerator)
            async for chunk in result:
                # Expect "data: {json}\n" lines; forward verbatim
                yield chunk
        else:
            if isinstance(result, dict) and result.get("__error__"):
                yield ErrorResponse(**result["__error__"])
            else:
                yield ChatCompletionResponse(**result)

    async def completions(
        self, request: CompletionRequest
    ) -> AsyncGenerator[Union[str, CompletionResponse, ErrorResponse], None]:
        payload = request.model_dump()
        is_stream = bool(payload.get("stream"))
        result = await self._post_openai("/v1/completions", payload, stream=is_stream)

        if is_stream:
            assert isinstance(result, AsyncGenerator)
            async for chunk in result:
                yield chunk
        else:
            if isinstance(result, dict) and result.get("__error__"):
                yield ErrorResponse(**result["__error__"])
            else:
                yield CompletionResponse(**result)

    async def embeddings(
        self, request: EmbeddingRequest
    ) -> AsyncGenerator[Union[EmbeddingResponse, ErrorResponse], None]:
        payload = request.model_dump()
        result = await self._post_openai("/v1/embeddings", payload, stream=False)
        if isinstance(result, dict) and result.get("__error__"):
            yield ErrorResponse(**result["__error__"])
        else:
            yield EmbeddingResponse(**result)

    async def resolve_lora(self, disk_lora_model: DiskMultiplexConfig):
        """
        Dynamically load a LoRA adapter into the running SGLang server.
        Requires server launched with --enable-lora and (ideally) max_lora_rank / lora_target_modules set.
        """
        assert self._client is not None
        payload = {
            "name": disk_lora_model.model_id,
            "path": disk_lora_model.local_path,
        }
        resp = await self._client.post("/load_lora_adapter", json=payload, headers=self._auth_headers())
        if resp.status_code != 200:
            raise ValueError(f"Failed to load LoRA adapter: {resp.text}")

    async def check_health(self) -> None:
        assert self._client is not None
        r = await self._client.get("/get_model_info", headers=self._auth_headers())
        r.raise_for_status()

    # Optional: if your runner lifecycle needs cleanup, you can add:
    def __del__(self):
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
