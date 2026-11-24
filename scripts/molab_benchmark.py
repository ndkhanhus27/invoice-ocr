"""GPU invoice benchmark. Downloaded by the thin Molab notebook from GitHub."""

def stream_command(args, log_path, emit, *, cwd, env, timeout=None,
                   heartbeat_seconds=15, emit_interval=0):
    """Drain merged stdout/stderr live; keep heartbeat and clean up on interruption."""
    import queue
    import re
    import subprocess
    import threading
    import time

    events = queue.Queue()
    process = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", bufsize=1)

    def read_output():
        try:
            for line in process.stdout:
                events.put(line)
        finally:
            events.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started = time.monotonic()
    last_heartbeat = started
    last_emit = float("-inf")
    pending = None
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            while True:
                now = time.monotonic()
                if timeout is not None and now - started > timeout:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    item = events.get(timeout=min(1, heartbeat_seconds))
                except queue.Empty:
                    item = ""
                if item is None:
                    if pending:
                        emit(f"[{log_path.name}] {pending}")
                    break
                if item:
                    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", item).strip()
                    log.write(clean + "\n")
                    if clean:
                        pending = clean
                now = time.monotonic()
                if pending and now - last_emit >= emit_interval:
                    emit(f"[{log_path.name}] {pending}")
                    pending = None
                    last_emit = now
                if now - last_heartbeat >= heartbeat_seconds:
                    emit(f"[{log_path.name}] vẫn đang chạy — {now - started:.0f}s")
                    last_heartbeat = now
            code = process.wait(timeout=15)
        emit(f"[{log_path.name}] kết thúc, exit code = {code}")
        return code
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=2)
        process.stdout.close()


def run_benchmark(config, progress):
    """Isolate downloaded project imports in subprocesses, not the notebook kernel."""
    import csv
    import hashlib
    import json
    import os
    import platform
    import re
    import shutil
    import socket
    import subprocess
    import sys
    import tarfile
    import time
    import traceback
    import uuid
    import zipfile
    from datetime import datetime, timezone
    from pathlib import Path
    from urllib.parse import quote
    from urllib.request import Request, build_opener, ProxyHandler, urlopen

    from huggingface_hub import HfApi, snapshot_download
    import zstandard

    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("Notebook này cần Molab Linux x86_64 và NVIDIA GPU.")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", config["github_repo"]):
        raise ValueError("GitHub repo cần có dạng owner/repo.")
    if not re.fullmatch(r"[\w./:-]+", config["model"]) or "cloud" in config["model"].lower():
        raise ValueError("Chọn tên model Ollama local, không dùng model cloud.")
    if config["max_tokens"] >= config["num_ctx"]:
        raise ValueError("Output token limit phải nhỏ hơn context để còn chỗ cho ảnh và prompt.")
    version = config["ollama_version"].strip().removeprefix("v")
    if version != "latest" and not re.fullmatch(r"\d+\.\d+\.\d+(?:[-.\w]*)?", version):
        raise ValueError("Ollama version phải là latest hoặc số release, ví dụ 0.x.y.")
    hf_token = config["hf_token"].strip() or os.environ.get("HF_TOKEN")
    gh_token = config["github_token"].strip() or os.environ.get("GITHUB_TOKEN")
    if not hf_token:
        raise ValueError("Cần HF_TOKEN có quyền đọc dataset private.")
    secrets = [value for value in (hf_token, gh_token) if value]

    def redact(value):
        text = str(value)
        for secret in secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    cache = Path.home() / ".cache" / "invoice-molab"
    cache.mkdir(parents=True, exist_ok=True)
    run = cache / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    checkout = run / "code"
    output = run / "results"
    output.mkdir()
    manifest = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
                "config": {k: v for k, v in config.items() if k not in ("hf_token", "github_token")},
                "python": sys.version, "platform": platform.platform()}
    server = None
    server_log = None
    started = time.perf_counter()
    metrics = []

    def save_manifest():
        (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def stage(message):
        message = redact(message)
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        message = f"[{stamp}] {message}"
        with (output / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        progress(message)

    # Credentials are deliberately absent from model and CLI subprocess environments.
    child_env = {k: v for k, v in os.environ.items()
                 if not k.startswith("INVOICE_") and k not in ("HF_TOKEN", "GITHUB_TOKEN", "GH_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    child_env["PYTHONUNBUFFERED"] = "1"

    def command(args, log_name, *, env=None, timeout=None):
        def relay(message):
            stage(message)
            if log_name == "extract.log" and "] error " in message:
                prediction_path = Path(message.split("] error ", 1)[1])
                if prediction_path.parent.resolve() == (output / "predictions").resolve() and prediction_path.is_file():
                    try:
                        prediction = json.loads(prediction_path.read_text(encoding="utf-8-sig"))
                        stage(f"LỖI hóa đơn {prediction.get('document_id')}: {prediction.get('error', 'unknown error')}")
                    except (ValueError, OSError):
                        pass  # Raw subprocess output is still available if diagnostics cannot be read.
        return stream_command(args, output / log_name, relay,
                              cwd=checkout if checkout.exists() else run,
                              env=env or child_env, timeout=timeout,
                              emit_interval=1 if log_name == "model-pull.log" else 0)

    def download(url, destination, headers=None):
        temporary = destination.with_suffix(destination.suffix + ".part")
        stage(f"Bắt đầu tải {destination.name}")
        with urlopen(Request(url, headers=headers or {}), timeout=120) as response, temporary.open("wb") as target:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            last_report = time.monotonic()
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                received += len(chunk)
                if time.monotonic() - last_report >= 10:
                    suffix = f" / {total / 2**20:.0f} MiB ({100 * received / total:.1f}%)" if total else " MiB"
                    stage(f"{destination.name}: {received / 2**20:.0f}" + suffix)
                    last_report = time.monotonic()
        temporary.replace(destination)
        stage(f"Đã tải {destination.name}: {received / 2**20:.1f} MiB")

    local_http = build_opener(ProxyHandler({}))

    def api(path, body=None, timeout=60):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
        with local_http.open(request, timeout=timeout) as response:
            return json.load(response)

    try:
        stage(f"Thư mục log/kết quả: {output}")
        stage("1/6 — Kiểm tra GPU")
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        manifest["gpu"] = gpu.stdout.strip()
        stage(gpu.stdout.strip())
        rows = list(csv.reader(gpu.stdout.splitlines()))
        if not rows:
            raise RuntimeError("Không tìm thấy NVIDIA GPU.")
        # Runtime visibility may remap GPU indices; preserve CUDA_VISIBLE_DEVICES from Molab.
        if max(float(row[-1]) for row in rows) < 44000 and config["model"] == "gemma4:31b-it-q8_0":
            raise RuntimeError("Model Q8 mặc định cần nhiều VRAM; hiện không có GPU trống tối thiểu 44.000 MiB. Chọn runtime lớn hơn hoặc model nhỏ hơn.")

        stage("2/6 — Lấy code GitHub và tải FULL dataset")
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "invoice-ocr-molab"}
        if gh_token:
            headers["Authorization"] = "Bearer " + gh_token
        repo_url = "https://api.github.com/repos/" + config["github_repo"]
        with urlopen(Request(repo_url + "/commits/" + quote(config["github_ref"], safe=""), headers=headers), timeout=60) as response:
            code_sha = json.load(response)["sha"]
        manifest["github_commit"] = code_sha
        stage("GitHub commit: " + code_sha)
        archive = run / "code.zip"
        download(repo_url + "/zipball/" + code_sha, archive, headers)
        checkout.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                parts = Path(member.filename).parts[1:]
                if not parts:
                    continue
                target = checkout.joinpath(*parts)
                if not target.resolve().is_relative_to(checkout.resolve()) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Unsafe path/link in GitHub archive")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("wb") as dest:
                        shutil.copyfileobj(source, dest)
        info = HfApi(token=hf_token).repo_info(config["dataset"], repo_type="dataset", revision=config["dataset_ref"])
        manifest["dataset_commit"] = info.sha
        stage("HF dataset commit: " + info.sha + "; bắt đầu snapshot_download (xem progress bên dưới)")
        dataset = Path(snapshot_download(repo_id=config["dataset"], repo_type="dataset", revision=info.sha, token=hf_token))
        stage("Đã tải FULL dataset: " + str(dataset))
        for name in ("data", "data_split", "groundtruth"):
            source = dataset / name
            if not source.exists() and name != "data_split":
                raise ValueError("Dataset thiếu thư mục " + name)
            if source.exists():
                if (checkout / name).exists():
                    raise ValueError("GitHub chứa thư mục dữ liệu " + name + "; cần tách data sang HF.")
                (checkout / name).symlink_to(source, target_is_directory=True)
        source_gt = dataset / "groundtruth" / "reviewed" / (config["split"] + ".jsonl")
        # Keep original JSON lines verbatim: Decimal labels must not round through float.
        lines = [line for line in source_gt.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        records = [json.loads(line) for line in lines]
        if not records or any(row.get("review_status", "reviewed") != "reviewed" or "editor_draft" in row or "annotation" not in row for row in records):
            raise ValueError("Tập này chưa có groundtruth reviewed hợp lệ. Hãy chọn valid đã review.")
        ids = [row["document_id"] for row in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Groundtruth bị trùng document_id.")
        count = int(config["limit"]) or len(lines)
        selected_lines = lines[:count]
        manifest.update(reviewed_available=len(lines), evaluated_documents=len(selected_lines), document_ids=ids[:count])
        annotations = dataset / "groundtruth" / "annotations" / (config["split"] + ".jsonl")
        if annotations.exists():
            manifest["split_annotation_count"] = sum(bool(line.strip()) for line in annotations.read_text(encoding="utf-8-sig").splitlines())
        gt = output / "groundtruth_selected.jsonl"
        gt.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
        # Do not pass annotations to extraction, only document IDs and ordered image paths.
        input_manifest = output / "input_images.jsonl"
        input_manifest.write_text("".join(json.dumps({"document_id": r["document_id"], "images": r["images"]}, ensure_ascii=False) + "\n"
                                         for r in records[:count]), encoding="utf-8")
        child_env["PYTHONPATH"] = str(checkout / "src")
        env_file = run / "empty.env"
        env_file.write_text("# Explicit CLI settings only\n", encoding="utf-8")
        if command([sys.executable, "-m", "invoice_ocr", "plan", "--env-file", str(env_file),
                    "--input", str(input_manifest), "--output", str(output / "predictions")], "plan.log"):
            raise RuntimeError("Ảnh/manifest không hợp lệ; xem plan.log trong ZIP.")
        stage(f"Reviewed: {len(lines)}; lần này chấm {len(selected_lines)} hóa đơn.")

        stage("3/6 — Chuẩn bị Ollama trong cache Molab")
        install = cache / "ollama" / version
        binary = install / "bin" / "ollama"
        if not binary.is_file() or not (install / ".ready").is_file():
            install.mkdir(parents=True, exist_ok=True)
            package = install / "ollama.tar.zst"
            url = ("https://ollama.com/download/ollama-linux-amd64.tar.zst" if version == "latest" else
                   f"https://github.com/ollama/ollama/releases/download/v{version}/ollama-linux-amd64.tar.zst")
            download(url, package)
            stage("Đang giải nén Ollama")
            with package.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    tar.extractall(install, filter="data")
            binary.chmod(binary.stat().st_mode | 0o111)
            (install / ".ready").write_text("ok\n", encoding="utf-8")
            package.unlink()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        server_env = dict(child_env, OLLAMA_HOST=f"127.0.0.1:{port}",
                          OLLAMA_MODELS=str(cache / "models"), OLLAMA_NUM_PARALLEL="1",
                          OLLAMA_CONTEXT_LENGTH=str(config["num_ctx"]), OLLAMA_KEEP_ALIVE="30m",
                          OLLAMA_NO_CLOUD="1")
        server_log = (output / "ollama-server.log").open("w", encoding="utf-8")
        server = subprocess.Popen([str(binary), "serve"], env=server_env, stdout=server_log, stderr=subprocess.STDOUT)
        for attempt in range(120):
            if server.poll() is not None:
                raise RuntimeError("Ollama không khởi động được; xem ollama-server.log.")
            try:
                manifest["ollama_version"] = api("/api/version", timeout=2)["version"]
                break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError("Ollama chưa sẵn sàng sau 120 giây.")
        stage("4/6 — Tải model và kiểm tra GPU (lần đầu có thể mất nhiều phút)")
        if command([str(binary), "pull", config["model"]], "model-pull.log", env=server_env):
            raise RuntimeError("Tải model thất bại; xem model-pull.log.")
        shown = api("/api/show", {"model": config["model"]})
        if "vision" not in shown.get("capabilities", []):
            raise ValueError("Model đã chọn không hỗ trợ ảnh.")
        manifest["model_info"] = shown
        manifest["installed_models"] = api("/api/tags")
        stage("Đang nạp model vào GPU; tối đa 1800 giây. Log server: " + str(output / "ollama-server.log"))
        api("/api/generate", {"model": config["model"], "prompt": "", "stream": False,
                              "keep_alive": "30m", "options": {"num_ctx": int(config["num_ctx"])}}, timeout=1800)
        loaded = api("/api/ps").get("models", [])
        manifest["gpu_preflight"] = loaded
        if not loaded or not any(m.get("size_vram", 0) > 0 for m in loaded):
            raise RuntimeError("Model không được load lên GPU. Dừng trước khi benchmark.")
        manifest["status"] = "extracting"
        manifest["prompt_sha256"] = hashlib.sha256((checkout / "prompts/invoice.txt").read_bytes()).hexdigest()
        save_manifest()
        stage("5/6 — Trích xuất; log từng hóa đơn sẽ hiện trực tiếp bên dưới")
        extraction_started = time.perf_counter()
        exit_code = command([sys.executable, "-m", "invoice_ocr", "extract", "--env-file", str(env_file),
                             "--backend", "ollama", "--input", str(input_manifest), "--output", str(output / "predictions"),
                             "--ollama-url", base_url, "--ollama-model", config["model"], "--ollama-device", "auto",
                             "--ollama-timeout", str(config["timeout"]), "--ollama-num-ctx", str(config["num_ctx"]),
                             "--ollama-keep-alive", "30m", "--max-edge", str(config["max_edge"]),
                             "--max-new-tokens", str(config["max_tokens"])], "extract.log")
        manifest["extraction_exit_code"] = exit_code
        manifest["extraction_wall_seconds"] = round(time.perf_counter() - extraction_started, 3)
        prediction_dir = output / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        predictions = [json.loads(p.read_text(encoding="utf-8-sig")) for p in prediction_dir.glob("*.json")]
        if exit_code != 0 and not predictions:
            raise RuntimeError("Trích xuất không khởi chạy được; xem extract.log trong ZIP.")
        manifest["successful"] = sum(p.get("status") == "ok" for p in predictions)
        manifest["failed_or_missing"] = len(selected_lines) - manifest["successful"]
        manifest["inference_seconds"] = round(sum(p.get("inference_seconds", 0) for p in predictions), 3)
        for prediction in predictions:
            if prediction.get("status") != "ok":
                stage(f"LỖI hóa đơn {prediction.get('document_id')}: {prediction.get('error', 'unknown error')}")
        stage("6/6 — Chấm điểm (hóa đơn lỗi/thiếu vẫn nằm trong mẫu chấm)")
        if command([sys.executable, "-m", "invoice_ocr", "evaluate", "--groundtruth", str(gt),
                    "--predictions", str(prediction_dir), "--output", str(output / "evaluation")], "evaluate.log"):
            raise RuntimeError("Chấm điểm thất bại; xem evaluate.log.")
        with (output / "evaluation/metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            metrics = list(csv.DictReader(stream))
        manifest["status"] = "completed" if not manifest["failed_or_missing"] and exit_code == 0 else "completed_with_errors"
        stage("Đã chấm xong. Tải ZIP bên dưới trước khi đóng runtime.")
    except Exception as error:
        manifest.update(status="failed", error=redact(f"{type(error).__name__}: {error}"))
        stage("Dừng: " + manifest["error"])
        stage(redact(traceback.format_exc()))
        for detail in (getattr(error, "stdout", None), getattr(error, "stderr", None)):
            if detail:
                stage(str(detail))
        (output / "traceback.log").write_text(redact(traceback.format_exc()), encoding="utf-8")
        for name in ("plan.log", "model-pull.log", "ollama-server.log", "extract.log", "evaluate.log"):
            path = output / name
            if path.is_file():
                stage(f"--- {name}: 15 dòng cuối ---\n" + "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        if server_log is not None:
            server_log.close()
        manifest["total_wall_seconds"] = round(time.perf_counter() - started, 3)
        save_manifest()
        # Results contain reviewed labels and predictions, but never images or tokens.
        for log in output.glob("*.log"):
            content = log.read_text(encoding="utf-8", errors="replace")
            log.write_text(redact(content), encoding="utf-8")
        bundle_path = Path(shutil.make_archive(str(run / "invoice-results"), "zip", output))
    return {"manifest": manifest, "metrics": metrics, "zip": bundle_path, "directory": output}


def install_dependencies():
    """Install runtime dependencies with visible logs, on explicit notebook call."""
    import os
    from pathlib import Path
    import platform
    import sys
    if sys.version_info < (3, 12) or platform.system() != "Linux":
        raise RuntimeError("Chọn Linux GPU runtime với Python >= 3.12.")
    packages = ["huggingface-hub>=0.34,<2", "Pillow>=10,<13", "zstandard>=0.23,<1"]
    code = stream_command([sys.executable, "-m", "pip", "install", *packages],
                          Path.cwd() / "molab-dependencies.log", print,
                          cwd=Path.cwd(), env=os.environ.copy())
    if code:
        raise RuntimeError("Cài thư viện thất bại; xem log phía trên.")


def run_from_notebook(config, github_token=""):
    """Prompt secrets at runtime; keep them out of notebook source and result files."""
    import getpass
    import os
    import traceback
    print("Đã nhận lệnh chạy benchmark.", flush=True)
    selected = dict(config)
    selected["hf_token"] = os.environ.get("HF_TOKEN", "")
    selected["github_token"] = github_token or os.environ.get("GITHUB_TOKEN", "")
    try:
        if not selected["hf_token"]:
            print("Đang chờ nhập HF token. Nếu không thấy ô nhập, đặt secret HF_TOKEN trong runtime rồi chạy lại cell.", flush=True)
            selected["hf_token"] = getpass.getpass("HF token (ẩn): ")
        print("Đã có HF token; bắt đầu kiểm tra GPU và tải dữ liệu.", flush=True)
        return run_benchmark(selected, lambda message: print(message, flush=True))
    except Exception:
        details = traceback.format_exc()
        for secret in (selected["hf_token"], selected["github_token"]):
            if secret:
                details = details.replace(secret, "[REDACTED]")
        print("KHÔNG KHỞI CHẠY ĐƯỢC:\n" + details, flush=True)
        return None
    finally:
        selected.pop("hf_token", None)
        selected.pop("github_token", None)


def show_results(result):
    import html
    import json
    from pathlib import Path
    import shutil
    from IPython.display import HTML, FileLink, display
    if result is None:
        print("Chưa có kết quả; xem lỗi ở cell chạy.")
        return
    manifest = result["manifest"]
    keys = ["status", "error", "github_commit", "dataset_commit", "ollama_version",
            "reviewed_available", "evaluated_documents", "successful", "failed_or_missing",
            "extraction_wall_seconds", "inference_seconds", "total_wall_seconds"]
    print(json.dumps({k: manifest[k] for k in keys if k in manifest}, ensure_ascii=False, indent=2))
    focus = {"purchase_order", "items_rows", "totals.total_payment", "invoice_all_fields", "schema_valid", "all_fields"}
    columns = ["field", "correct", "wrong", "missing", "extra", "evaluated", "exact_match_pct", "cer_pct"]
    for title, rows in [("Tiêu chí chính", [r for r in result["metrics"] if r["field"] in focus]),
                        ("Toàn bộ tiêu chí", result["metrics"])]:
        print(title)
        header = "".join("<th>" + html.escape(k) + "</th>" for k in columns)
        body = "".join("<tr>" + "".join("<td>" + html.escape(str(r.get(k, ""))) + "</td>" for k in columns) + "</tr>" for r in rows)
        display(HTML("<div style='overflow-x:auto'><table><tr>" + header + "</tr>" + body + "</table></div>"))
    downloads = Path.cwd() / "invoice-results"
    downloads.mkdir(exist_ok=True)
    target = downloads / (result["directory"].parent.name + ".zip")
    shutil.copy2(result["zip"], target)
    print("Tải ZIP hoặc lấy trong trình duyệt file:", target)
    display(FileLink(str(target.relative_to(Path.cwd()))))
