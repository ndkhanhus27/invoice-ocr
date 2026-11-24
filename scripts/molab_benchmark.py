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


def load_reviewed_groundtruth(dataset, split, emit):
    """Prefer exported reviewed labels; otherwise use explicitly reviewed annotations only."""
    import json
    from pathlib import Path
    if split not in ("train", "valid", "test"):
        raise ValueError("Split phải là train, valid hoặc test.")
    dataset = Path(dataset)
    reviewed = dataset / "groundtruth" / "reviewed" / (split + ".jsonl")
    annotations = dataset / "groundtruth" / "annotations" / (split + ".jsonl")
    if reviewed.is_file():
        source = reviewed
        lines = [line for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif annotations.is_file():
        source = annotations
        lines = []
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("review_status") == "reviewed" and "editor_draft" not in record:
                lines.append(line)  # Preserve numeric literals without float round-tripping.
        emit(f"Thiếu {reviewed.relative_to(dataset)}; lấy {len(lines)} hóa đơn có review_status=reviewed từ {source.relative_to(dataset)}.")
    else:
        available = ", ".join(str(p.relative_to(dataset)) for p in sorted((dataset / "groundtruth").rglob("*.jsonl"))) or "không có JSONL"
        raise FileNotFoundError(f"Dataset thiếu groundtruth cho {split}. Cần {reviewed.relative_to(dataset)} hoặc {annotations.relative_to(dataset)} có review_status=reviewed. Hiện có: {available}. Hãy upload groundtruth/reviewed/{split}.jsonl lên HF rồi chạy lại với dataset_ref=main.")
    if not lines:
        raise ValueError(f"{source.relative_to(dataset)} không có hóa đơn reviewed để chấm; cần review/export nhãn rồi cập nhật HF.")
    emit(f"Groundtruth: {source.relative_to(dataset)} — {len(lines)} hóa đơn reviewed.")
    return source, lines


def materialize_image_tree(source, destination, emit):
    """HF snapshot files point outside data/; create ordinary files without changing guards."""
    import os
    from pathlib import Path
    import shutil
    import time
    counts = {"hardlinked": 0, "copied": 0}
    last_report = time.monotonic()

    def link_or_copy(src, dst):
        nonlocal last_report
        resolved = Path(src).resolve(strict=True)
        try:
            os.link(resolved, dst)
            counts["hardlinked"] += 1
        except OSError:
            shutil.copyfile(resolved, dst)
            counts["copied"] += 1
        if time.monotonic() - last_report >= 15:
            emit(f"Đang chuẩn bị ảnh: {sum(counts.values())} file")
            last_report = time.monotonic()
        return dst

    emit("Chuẩn bị data/ từ HF cache: hardlink file thật; copy nếu khác filesystem.")
    shutil.copytree(source, destination, symlinks=False, copy_function=link_or_copy)
    emit(f"Đã chuẩn bị ảnh: {counts['hardlinked']} hardlink, {counts['copied']} copy.")
    return counts


def verify_image_files(root, records):
    """Report absent uploads separately from cache symlink containment errors."""
    from pathlib import Path, PurePosixPath
    missing = []
    for record in records:
        for image in record.get("images", []):
            name = image if isinstance(image, str) else image.get("file", "")
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "data" or "\\" in name:
                raise ValueError("Đường dẫn ảnh không hợp lệ trong groundtruth: " + name)
            if not (Path(root) / name).is_file():
                missing.append(name)
    if missing:
        raise FileNotFoundError(f"Dataset thiếu {len(missing)} ảnh được groundtruth tham chiếu. Ví dụ: {', '.join(missing[:5])}. Upload các ảnh này vào đúng data/<split>/<document_id>/ trên HF rồi chạy lại.")



def validate_bf16_config(config):
    if config.get("model") != "google/gemma-4-31B-it":
        raise ValueError("Notebook này dùng bản gốc google/gemma-4-31B-it. Import notebook BF16 mới, không dùng tên model Ollama cũ.")
    if config.get("weights_format", "bf16") != "bf16":
        raise ValueError("Đã chọn bản gốc BF16; không tự đổi sang checkpoint lượng tử hóa.")
    for key in ("max_tokens", "max_edge", "max_input_tokens", "vision_soft_tokens"):
        if int(config.get(key, {"max_input_tokens": 8192, "vision_soft_tokens": 1120}.get(key, 0))) < 1:
            raise ValueError(key + " phải là số nguyên dương.")
    if int(config.get("vision_soft_tokens", 1120)) not in (70, 140, 280, 560, 1120):
        raise ValueError("vision_soft_tokens không được hỗ trợ.")
    if int(config.get("limit", 0)) < 0:
        raise ValueError("limit phải >= 0.")


def runtime_probe():
    """Probe in a short child so notebook kernel never imports torch or owns model memory."""
    import json
    import subprocess
    import sys
    snippet = '''import json,torch
from pathlib import Path
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch không truy cập được CUDA; chọn runtime GPU trước.")
free,total=torch.cuda.mem_get_info(0)
info={"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
      "vram_total_gib":total/1024**3,"vram_free_gib":free/1024**3,"bf16":torch.cuda.is_bf16_supported()}
for name in ("memory.max","memory.current","memory.events"):
    path=Path("/sys/fs/cgroup")/name
    if path.exists(): info[name]=path.read_text().strip()
print(json.dumps(info))
'''
    process = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, timeout=90)
    if process.returncode:
        raise RuntimeError("CUDA probe thất bại: " + process.stderr[-3000:])
    return json.loads(process.stdout.strip().splitlines()[-1])


def require_bf16_hardware(hardware):
    if not hardware.get("bf16"):
        raise RuntimeError("GPU hiện tại không hỗ trợ BF16.")
    # Original 31B is about 70 GB before context/runtime headroom. Never CPU-offload on 32 GiB RAM.
    if hardware.get("vram_free_gib", 0) < 80:
        raise RuntimeError(f"Gemma 4 31B BF16 cần GPU lớn: yêu cầu tối thiểu 80 GiB VRAM trống trong cấu hình này; đang có {hardware.get('vram_free_gib', 0):.1f} GiB. 32 GiB RAM hệ thống khác với VRAM. Không tự offload CPU hoặc đổi sang 4-bit.")


def run_benchmark(config, progress):
    """Isolate downloaded project imports in subprocesses, not the notebook kernel."""
    import csv
    import hashlib
    import json
    import os
    import platform
    import re
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    import uuid
    import zipfile
    from datetime import datetime, timezone
    from pathlib import Path
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    from huggingface_hub import HfApi, snapshot_download

    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("Notebook này cần Molab Linux x86_64 và NVIDIA GPU.")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", config["github_repo"]):
        raise ValueError("GitHub repo cần có dạng owner/repo.")
    validate_bf16_config(config)
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
    output = Path.cwd() / "invoice-results" / run.name / "results"
    output.mkdir(parents=True)
    manifest = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
                "config": {k: v for k, v in config.items() if k not in ("hf_token", "github_token")},
                "python": sys.version, "platform": platform.platform()}
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
        manifest["last_progress"] = message
        save_manifest()
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
                              emit_interval=0)

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

    try:
        stage(f"Thư mục log/kết quả: {output}")
        stage("1/5 — Kiểm tra RAM, VRAM và BF16 trước khi tải model")
        hardware = runtime_probe()
        manifest["hardware"] = hardware
        stage(json.dumps(hardware, ensure_ascii=False))
        require_bf16_hardware(hardware)
        stage("2/5 — Lấy code GitHub và tải FULL dataset")
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
                if name == "data":
                    manifest["image_materialization"] = materialize_image_tree(source, checkout / name, stage)
                else:
                    (checkout / name).symlink_to(source, target_is_directory=True)
        source_gt, lines = load_reviewed_groundtruth(dataset, config["split"], stage)
        manifest["groundtruth_source"] = source_gt.relative_to(dataset).as_posix()
        manifest["groundtruth_selection"] = "explicit_reviewed_status" if source_gt.parent.name == "annotations" else "reviewed_export"
        records = [json.loads(line) for line in lines]
        if not records or any(row.get("review_status", "reviewed") != "reviewed" or "editor_draft" in row or "annotation" not in row for row in records):
            raise ValueError("Tập này chưa có groundtruth reviewed hợp lệ. Hãy chọn valid đã review.")
        ids = [row["document_id"] for row in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Groundtruth bị trùng document_id.")
        count = int(config["limit"]) or len(lines)
        selected_lines = lines[:count]
        verify_image_files(checkout, records[:count])
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

        stage("3/5 — Tải bản gốc Gemma 4 31B từ Hugging Face, BF16, không lượng tử hóa")
        model_info = HfApi(token=hf_token).model_info(config["model"], revision=config.get("model_ref", "main"), files_metadata=True)
        model_sha = model_info.sha
        manifest.update(model_id=config["model"], model_commit=model_sha, weights_format="bf16", backend="transformers")
        save_manifest()
        model_folder = cache / "models" / model_sha
        model_folder.mkdir(parents=True, exist_ok=True)
        needed = sum(max(0, (item.size or 0) - ((model_folder / item.rfilename).stat().st_size if (model_folder / item.rfilename).is_file() else 0))
                     for item in model_info.siblings if item.rfilename.endswith(".safetensors"))
        if shutil.disk_usage(model_folder).free < needed + 2 * 1024**3:
            raise RuntimeError(f"Thiếu disk để tải weights gốc: cần thêm khoảng {needed / 1024**3:.1f} GiB và 2 GiB dự phòng.")
        stage(f"Model commit: {model_sha}; cần tải thêm khoảng {needed / 1024**3:.1f} GiB weights.")
        model_path = Path(snapshot_download(repo_id=config["model"], repo_type="model", revision=model_sha,
                         token=hf_token, local_dir=str(model_folder),
                         allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"]))
        manifest["prompt_sha256"] = hashlib.sha256((checkout / "prompts/invoice.txt").read_bytes()).hexdigest()
        manifest["status"] = "extracting"
        save_manifest()
        stage("4/5 — Transformers nạp từng shard BF16 lên cuda:0, batch 1; theo dõi heartbeat và log trực tiếp")
        extraction_started = time.perf_counter()
        exit_code = command([sys.executable, "-m", "invoice_ocr", "extract", "--env-file", str(env_file),
                             "--backend", "transformers", "--input", str(input_manifest), "--output", str(output / "predictions"),
                             "--model", str(model_path), "--revision", model_sha, "--offline", "--device", "cuda:0",
                             "--weights-format", "bf16", "--max-input-tokens", str(config.get("max_input_tokens", 8192)),
                             "--vision-soft-tokens", str(config.get("vision_soft_tokens", 1120)),
                             "--max-edge", str(config["max_edge"]), "--max-new-tokens", str(config["max_tokens"])], "extract.log")
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
        stage("5/5 — Chấm điểm (hóa đơn lỗi/thiếu vẫn nằm trong mẫu chấm)")
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
        for name in ("plan.log", "extract.log", "evaluate.log"):
            path = output / name
            if path.is_file():
                stage(f"--- {name}: 15 dòng cuối ---\n" + "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
    finally:
        manifest["total_wall_seconds"] = round(time.perf_counter() - started, 3)
        save_manifest()
        # Results contain reviewed labels and predictions, but never images or tokens.
        for log in output.glob("*.log"):
            content = log.read_text(encoding="utf-8", errors="replace")
            log.write_text(redact(content), encoding="utf-8")
        bundle_path = Path(shutil.make_archive(str(output.parent / "invoice-results"), "zip", output))
    return {"manifest": manifest, "metrics": metrics, "zip": bundle_path, "directory": output}


def install_dependencies(requirements_url, github_token=""):
    """Keep Molab's working CUDA torch build, install the BF16 backend dependencies."""
    import os
    from pathlib import Path
    import platform
    import sys
    from urllib.request import Request, urlopen
    if sys.version_info < (3, 12) or platform.system() != "Linux":
        raise RuntimeError("Chọn Linux GPU runtime với Python >= 3.12.")
    print("Kiểm tra PyTorch CUDA hiện có; không tự thay CUDA wheel...", flush=True)
    print(runtime_probe(), flush=True)
    headers = {"User-Agent": "invoice-ocr-molab"}
    if github_token:
        headers["Authorization"] = "Bearer " + github_token
    print("Tải requirements Molab từ cùng commit GitHub...", flush=True)
    requirements = Path.cwd() / "molab-requirements.txt"
    with urlopen(Request(requirements_url, headers=headers), timeout=60) as response:
        requirements.write_bytes(response.read())
    code = stream_command([sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                          Path.cwd() / "molab-dependencies.log", print,
                          cwd=Path.cwd(), env=os.environ.copy())
    if code:
        raise RuntimeError("Cài thư viện thất bại; xem log phía trên.")
    print("Backend Transformers BF16 sẵn sàng.", flush=True)


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
    keys = ["status", "error", "github_commit", "dataset_commit", "model_id", "model_commit", "weights_format",
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
