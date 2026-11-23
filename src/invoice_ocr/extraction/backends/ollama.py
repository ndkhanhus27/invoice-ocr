"""Ollama REST adapter. No torch import, SDK dependency or remote cloud fallback."""
from __future__ import annotations

import base64
from io import BytesIO
import json
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from ...contracts.invoice import invoice_json_schema


class OllamaError(RuntimeError):
    """An actionable error safe to display without leaking URL credentials."""


class OllamaClient:
    def __init__(self, args):
        self.args = args
        self.url = args.ollama_url.rstrip('/')
        parts = urlsplit(self.url)
        if parts.scheme not in {'http', 'https'} or not parts.hostname or parts.username or parts.password:
            raise ValueError('Ollama URL must be http(s)://host:port, without credentials.')
        # Local requests must not pass invoice images through a machine-wide proxy.
        self.opener = build_opener(ProxyHandler({}))

    def request(self, path, payload=None, *, timeout=None):
        data = json.dumps(payload).encode('utf-8') if payload is not None else None
        request = Request(self.url + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=timeout or self.args.ollama_timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            code = error.code
            error.close()
            if code == 404:
                raise OllamaError('Model/endpoint not found. Run: ollama pull ' + self.args.ollama_model) from None
            raise OllamaError(f'Ollama HTTP {code}. Check model compatibility, RAM/VRAM and Ollama logs.') from None
        except (TimeoutError, socket.timeout):
            raise OllamaError('Ollama timed out. Increase INVOICE_OLLAMA_TIMEOUT or reduce image size/context.') from None
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise OllamaError('Ollama timed out. Increase INVOICE_OLLAMA_TIMEOUT.') from None
            raise OllamaError('Cannot connect to Ollama. Start Ollama and check INVOICE_OLLAMA_URL.') from None
        except (ValueError, UnicodeError):
            raise OllamaError('Ollama returned invalid JSON. Check the server URL and Ollama version.') from None
        if not isinstance(result, dict) or result.get('error'):
            raise OllamaError('Ollama returned an error. Check RAM/VRAM and Ollama logs.')
        return result

    def check_model(self):
        tags = self.request('/api/tags', timeout=10)
        selected = self.args.ollama_model
        canonical = selected if ':' in selected.rsplit('/', 1)[-1] else selected + ':latest'
        entry = next((m for m in tags.get('models', [])
                      if m.get('name') in {selected, canonical} or m.get('model') in {selected, canonical}), None)
        if entry is None:
            raise OllamaError('Model is not installed. Run: ollama pull ' + selected)
        info = self.request('/api/show', {'model': selected}, timeout=30)
        if 'vision' not in info.get('capabilities', []):
            raise OllamaError('Selected model does not support images. Use qwen3-vl:2b or another vision model.')
        return {'model': selected, 'digest': entry.get('digest'), 'capabilities': info['capabilities']}

    def generate(self, frames, prompt):
        images = []
        for frame in frames:
            with BytesIO() as buffer:
                frame.save(buffer, format='PNG')
                images.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
        options = {'temperature': 0, 'seed': 42, 'num_predict': self.args.max_new_tokens,
                   'num_ctx': self.args.ollama_num_ctx}
        if self.args.ollama_device == 'cpu':
            options['num_gpu'] = 0
        if self.args.ollama_num_threads:
            options['num_thread'] = self.args.ollama_num_threads
        return self.request('/api/chat', {
            'model': self.args.ollama_model, 'stream': False,
            'messages': [{'role': 'user', 'content': f'{prompt}\nThere are {len(images)} images in page order.',
                          'images': images}],
            'format': invoice_json_schema(), 'options': options,
            'keep_alive': self.args.ollama_keep_alive,
        })


def doctor(args):
    info = OllamaClient(args).check_model()
    print(json.dumps({'backend': 'ollama', 'requested_device': args.ollama_device, **info}))
    print('Ollama ready. During inference, use ollama ps to check actual CPU/GPU allocation.')
    return 0
