from pathlib import Path
import sys

ROOT = Path.cwd()


def fail(message: str) -> None:
    print(f"STOP: {message}")
    print("No files were modified.")
    raise SystemExit(1)

html_path = ROOT / "aidetect" / "templates" / "index.html"
pdf_path = ROOT / "aidetect" / "pdf_report.py"
for p in (html_path, pdf_path):
    if not p.is_file():
        fail(f"Required file not found: {p}")

html = html_path.read_text(encoding="utf-8")
pdf = pdf_path.read_text(encoding="utf-8")

html_old = """          provenancePanel.append(el('div','meta','The visual-model decision above remains separate and unchanged.'));"""
html_new = """          const badge=card.querySelector('.badge');
          const visualDisplay=r.label==='uncertain'
            ? 'UNCERTAIN (leaning '+(r.prob_ai >= r.threshold ? 'AI' : 'real')+')'
            : (r.label==='ai_generated' ? 'AI-GENERATED' : r.label==='real' ? 'LIKELY REAL' : String(r.label).toUpperCase());
          if(result.status==='detected'){
            if(badge){badge.className='badge b-ai';badge.textContent='AI ORIGIN SIGNAL DETECTED';}
            provenancePanel.insertBefore(
              el('strong',null,'IMPORTANT: Visual classifier and provenance verification disagree.'),
              provenancePanel.firstChild
            );
          }
          provenancePanel.append(el('div','meta','Visual-model output: '+visualDisplay+'; AI-class score: '+pct(r.prob_ai)+'. The visual score and threshold have not been changed.'));"""

if html.count(html_old) != 1:
    fail("Expected provenance result display line was not found exactly once in index.html. The file may have changed.")

pdf_old = '''    conclusion = (
        f"Based on the submitted case report, the model decision is {_DECISIONS[decision]} "
        f"(AI-class score {_format_score(score)}; threshold {_format_score(threshold)}). "
        "This is a model estimate, not proof. The SHA-256 identifies the uploaded bytes; "
        "manipulation detection is not implemented."
    )'''
pdf_new = '''    external_for_conclusion = report.get("external_provenance_verification")
    if isinstance(external_for_conclusion, Mapping) and external_for_conclusion.get("status") == "detected":
        conclusion = (
            "EXTERNAL PROVENANCE FINDING: a supported OpenAI provenance signal was detected. "
            f"The separate visual-model decision is {_DECISIONS[decision]} "
            f"(AI-class score {_format_score(score)}; threshold {_format_score(threshold)}). "
            "These findings disagree. Do not interpret the visual model's label as confirmation "
            "that the image is real; the provenance finding is separate evidence about origin. "
            "The SHA-256 identifies the uploaded bytes; manipulation detection is not implemented."
        )
    else:
        conclusion = (
            f"Based on the submitted case report, the model decision is {_DECISIONS[decision]} "
            f"(AI-class score {_format_score(score)}; threshold {_format_score(threshold)}). "
            "This is a model estimate, not proof. The SHA-256 identifies the uploaded bytes; "
            "manipulation detection is not implemented."
        )'''

if pdf.count(pdf_old) != 1:
    fail("Expected forensic conclusion block was not found exactly once in pdf_report.py. The file may have changed.")

html_backup = html_path.with_name("index.html.before_provenance_conflict_display.bak")
pdf_backup = pdf_path.with_name("pdf_report.py.before_provenance_conflict_display.bak")
if html_backup.exists() or pdf_backup.exists():
    fail("A backup destination already exists. No files were modified.")

# Prepare all output content before writing any files.
updated_html = html.replace(html_old, html_new, 1)
updated_pdf = pdf.replace(pdf_old, pdf_new, 1)

# Create backups first, then write only the two targeted changes.
html_backup.write_bytes(html_path.read_bytes())
pdf_backup.write_bytes(pdf_path.read_bytes())
html_path.write_text(updated_html, encoding="utf-8", newline="")
pdf_path.write_text(updated_pdf, encoding="utf-8", newline="")

print("SUCCESS: Provenance conflict display updated.")
print(f"Backup created: {html_backup.relative_to(ROOT)}")
print(f"Backup created: {pdf_backup.relative_to(ROOT)}")
print("Updated: aidetect/templates/index.html")
print("Updated: aidetect/pdf_report.py")
print("When a supported OpenAI provenance signal is detected, the UI badge will reflect that finding and explain the disagreement.")
print("The visual-model score, threshold, and model weights remain unchanged.")
