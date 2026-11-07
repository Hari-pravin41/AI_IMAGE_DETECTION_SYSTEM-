from pathlib import Path
import sys

ROOT = Path.cwd()
TARGETS = {
    "server": ROOT / "aidetect" / "server.py",
    "html": ROOT / "aidetect" / "templates" / "index.html",
    "pdf": ROOT / "aidetect" / "pdf_report.py",
}
BACKUP_SUFFIX = ".before_openai_provenance.bak"

for name, path in TARGETS.items():
    if not path.is_file():
        print(f"STOP: Expected project file not found: {path}")
        sys.exit(1)

original_bytes = {name: path.read_bytes() for name, path in TARGETS.items()}
original_text = {
    name: blob.decode("utf-8").replace("\r\n", "\n")
    for name, blob in original_bytes.items()
}
backups = {name: path.with_name(path.name + BACKUP_SUFFIX) for name, path in TARGETS.items()}
if any(path.exists() for path in backups.values()):
    print("STOP: A backup file already exists. No changes made.")
    for path in backups.values():
        if path.exists():
            print(f"Existing backup: {path}")
    sys.exit(1)

# server.py imports
server = original_text["server"]
old_import_block = "import base64\nimport io\nimport json\nfrom pathlib import Path\nimport re\n"
new_import_block = (
    "import base64\n"
    "import io\n"
    "import json\n"
    "import os\n"
    "import secrets\n"
    "from datetime import datetime, timezone\n"
    "from pathlib import Path\n"
    "import re\n"
    "from urllib.error import HTTPError, URLError\n"
    "from urllib.request import Request, urlopen\n"
)
if server.count(old_import_block) != 1:
    print("STOP: Expected import block in server.py not found exactly once. No changes made.")
    sys.exit(1)
server = server.replace(old_import_block, new_import_block, 1)

# Optional route. Image is sent only when the button is clicked and confirmed in the UI.
route_anchor = '''    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/predict")
'''
route_replacement = r'''    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/verify-provenance")
    def verify_provenance():
        """Optional external check for supported OpenAI provenance signals."""
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return jsonify({"error": "OpenAI provenance verification is not configured on this server."}), 503

        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "Select one image to verify."}), 400

        max_bytes = min(int(cfg.max_upload_mb * 1024 * 1024), 50 * 1024 * 1024)
        data = uploaded.read(max_bytes + 1)
        if not data:
            return jsonify({"error": "The uploaded file is empty."}), 400
        if len(data) > max_bytes:
            return jsonify({"error": "The image exceeds the provenance verification size limit."}), 413

        name = Path(uploaded.filename or "upload").name
        try:
            loaded = load_image(data, cfg, name=name)
        except ImageError as exc:
            return jsonify({"error": f"Invalid image: {exc}"}), 400

        media = {
            "PNG": ("image/png", "upload.png"),
            "JPEG": ("image/jpeg", "upload.jpg"),
            "WEBP": ("image/webp", "upload.webp"),
        }.get(str(loaded.format or "").upper())
        if media is None:
            return jsonify({"error": "External provenance checks support PNG, JPEG, and WebP images only."}), 400

        mime_type, safe_filename = media
        boundary = "----aidetect-" + secrets.token_hex(16)
        prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("ascii")
        body = prefix + data + f"\r\n--{boundary}--\r\n".encode("ascii")

        req = Request(
            "https://api.openai.com/v1/content_provenance_checks",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=30) as response:
                response_bytes = response.read(2 * 1024 * 1024 + 1)
        except HTTPError as exc:
            app.logger.warning("OpenAI provenance API returned HTTP %s", exc.code)
            messages = {
                401: "OpenAI rejected the API key. Check the key in this server's terminal.",
                403: "The API key does not have permission for this operation.",
                404: "Content Provenance API access is unavailable for this account or endpoint.",
                429: "OpenAI provenance verification rate limit or quota reached.",
            }
            message = messages.get(exc.code, f"OpenAI provenance verification failed (HTTP {exc.code}).")
            return jsonify({"error": message}), 502
        except (URLError, TimeoutError, OSError) as exc:
            app.logger.warning("OpenAI provenance request failed: %s", type(exc).__name__)
            return jsonify({"error": "Could not reach OpenAI provenance verification. Check the network and try again."}), 502

        if len(response_bytes) > 2 * 1024 * 1024:
            return jsonify({"error": "Unexpectedly large response from the provenance service."}), 502
        try:
            payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return jsonify({"error": "OpenAI returned an unreadable provenance response."}), 502

        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            return jsonify({"error": "OpenAI returned an unexpected provenance response."}), 502

        clean_results = []
        for item in raw_results[:5]:
            if not isinstance(item, dict) or item.get("type") not in ("c2pa", "synthid"):
                continue
            outcome = item.get("outcome")
            if outcome not in ("detected", "not_detected"):
                continue
            signal = {"type": item["type"], "outcome": outcome}
            for field in ("validation_state", "issuer", "model", "generated_at"):
                value = item.get(field)
                if isinstance(value, str):
                    signal[field] = value[:160]
            clean_results.append(signal)

        confirmed = any(
            (r["type"] == "synthid" and r["outcome"] == "detected")
            or (
                r["type"] == "c2pa"
                and r["outcome"] == "detected"
                and r.get("validation_state") in ("trusted", "valid")
                and "openai" in r.get("issuer", "").lower()
            )
            for r in clean_results
        )
        if confirmed:
            status = "detected"
            summary = (
                "A supported OpenAI provenance signal was detected. This is evidence about origin, "
                "not a guarantee that all image content is unmodified."
            )
        elif clean_results and all(r["outcome"] == "not_detected" for r in clean_results):
            status = "not_detected"
            summary = (
                "No supported OpenAI provenance signal was detected. This does not prove the image "
                "is genuine or rule out generation by another AI system."
            )
        else:
            status = "inconclusive"
            summary = "The service did not return a conclusive supported provenance signal."

        return jsonify({
            "result": {
                "service": "OpenAI Content Provenance API",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "summary": summary,
                "results": clean_results,
            }
        })

    @app.post("/api/predict")
'''
if server.count(route_anchor) != 1:
    print("STOP: Expected GPU health/predict route anchor not found exactly once. No changes made.")
    sys.exit(1)
server = server.replace(route_anchor, route_replacement, 1)

# index.html: external verify button. Existing local prediction remains untouched.
html = original_text["html"]
old_ui = '''    downloads.append(download,pdfDownload);
    section.append(summary,dl,privacy,manipulation,downloads);
'''
new_ui = '''    if(gpuMode){
      const provenancePanel=el('div','note','Optional external check. Your original image is sent to OpenAI only after you click Verify and confirm.');
      const verifyButton=el('button','download','Verify OpenAI provenance (external)');
      verifyButton.type='button';
      verifyButton.addEventListener('click',async()=>{
        if(!sourceFile){provenancePanel.textContent='Original file is not available for verification.';return}
        if(!window.confirm('Send the original image to OpenAI for supported provenance checks? The image leaves this local application.'))return;
        verifyButton.disabled=true;
        verifyButton.textContent='Verifying provenance...';
        provenancePanel.textContent='Checking supported OpenAI provenance signals...';
        try{
          const form=new FormData();
          form.append('file',sourceFile,sourceFile.name);
          const response=await fetch('/api/verify-provenance',{method:'POST',body:form});
          const payload=await response.json();
          if(!response.ok)throw new Error(payload.error||('HTTP '+response.status));
          const result=payload.result;
          if(!result)throw new Error('The verification response was incomplete.');
          report.external_provenance_verification=result;
          provenancePanel.replaceChildren(
            el('strong',null,'External provenance result: '+String(result.status||'inconclusive').toUpperCase()),
            document.createElement('br'),
            document.createTextNode(result.summary||'No summary provided.')
          );
          (result.results||[]).forEach(signal=>{
            const details=[String(signal.type||'signal').toUpperCase(),String(signal.outcome||'unknown')];
            if(signal.validation_state)details.push('validation: '+signal.validation_state);
            if(signal.issuer)details.push('issuer: '+signal.issuer);
            if(signal.model)details.push('model: '+signal.model);
            const line=el('div','meta',details.join(' | '));
            provenancePanel.append(line);
          });
          provenancePanel.append(el('div','meta','The visual-model decision above remains separate and unchanged.'));
        }catch(error){
          provenancePanel.textContent='Provenance verification failed: '+String(error.message||error);
        }finally{
          verifyButton.disabled=false;
          verifyButton.textContent='Verify OpenAI provenance (external)';
        }
      });
      downloads.append(download,pdfDownload,verifyButton);
      section.append(summary,dl,privacy,manipulation,provenancePanel,downloads);
    }else{
      downloads.append(download,pdfDownload);
      section.append(summary,dl,privacy,manipulation,downloads);
    }
'''
if html.count(old_ui) != 1:
    print("STOP: Expected UI download section not found exactly once. No changes made.")
    sys.exit(1)
html = html.replace(old_ui, new_ui, 1)

# pdf_report.py: render the optional result when it has been added to JSON/PDF report.
pdf = original_text["pdf"]
old_pdf = '''    story.append(_table(metadata_rows, styles))
    story.extend([
        _p("Interpretation and limitations", styles["section"]),
'''
new_pdf = '''    story.append(_table(metadata_rows, styles))

    external = report.get("external_provenance_verification")
    if external is not None:
        if not isinstance(external, Mapping):
            raise InvalidReportError("external provenance verification must be an object")
        status = external.get("status")
        status_labels = {
            "detected": "SUPPORTED OPENAI PROVENANCE SIGNAL DETECTED",
            "not_detected": "NO SUPPORTED OPENAI PROVENANCE SIGNAL DETECTED",
            "inconclusive": "INCONCLUSIVE",
        }
        story.extend([
            _p("External provenance verification", styles["section"]),
            _p(status_labels.get(status, "INCONCLUSIVE"), styles["body"]),
            _p(str(external.get("summary", "No summary available."))[:600], styles["body"]),
        ])
        checked_at = external.get("checked_at")
        if isinstance(checked_at, str):
            story.append(_p("Checked at (UTC): " + checked_at[:80], styles["small"]))
        signals = external.get("results", [])
        if isinstance(signals, list):
            for signal in signals[:5]:
                if not isinstance(signal, Mapping):
                    continue
                parts = [
                    str(signal.get("type", "signal"))[:30],
                    str(signal.get("outcome", "unknown"))[:30],
                ]
                for field in ("validation_state", "issuer", "model", "generated_at"):
                    value = signal.get(field)
                    if isinstance(value, str) and value:
                        parts.append(field + ": " + value[:100])
                story.append(_p(" | ".join(parts), styles["small"]))

    story.extend([
        _p("Interpretation and limitations", styles["section"]),
'''
if pdf.count(old_pdf) != 1:
    print("STOP: Expected PDF section anchor not found exactly once. No changes made.")
    sys.exit(1)
pdf = pdf.replace(old_pdf, new_pdf, 1)

updated = {"server": server, "html": html, "pdf": pdf}
# Validate basic structural indicators before writing anything.
if "@app.post(\"/api/verify-provenance\")" not in server:
    print("STOP: Route validation failed. No changes made.")
    sys.exit(1)
if "Verify OpenAI provenance (external)" not in html:
    print("STOP: UI validation failed. No changes made.")
    sys.exit(1)
if 'External provenance verification' not in pdf:
    print("STOP: PDF validation failed. No changes made.")
    sys.exit(1)

# Create all backups first. Then write each updated file preserving original newline style.
try:
    for name, path in TARGETS.items():
        backups[name].write_bytes(original_bytes[name])
    for name, path in TARGETS.items():
        newline = "\r\n" if b"\r\n" in original_bytes[name] else "\n"
        content = updated[name].replace("\n", newline).encode("utf-8")
        path.write_bytes(content)
except Exception as exc:
    print(f"ERROR while writing patches: {exc}")
    print("Backups, if created, use the suffix .before_openai_provenance.bak")
    sys.exit(2)

print("SUCCESS: optional provenance verification feature installed.")
print("Backups created:")
for path in backups.values():
    print(f"  {path.relative_to(ROOT)}")
print("Modified files:")
for path in TARGETS.values():
    print(f"  {path.relative_to(ROOT)}")
print("No API key was written to a file. No model weights or datasets were changed.")
