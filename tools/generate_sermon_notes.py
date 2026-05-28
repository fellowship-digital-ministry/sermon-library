#!/usr/bin/env python3
"""
Generate a condensed, catalog-style record for each sermon from its transcript:
an Introduction (a short description), an Outline, and a few subject Themes.

Think of it as the way a librarian catalogs a book: descriptive metadata that
helps someone find and orient to the sermon, NOT a substitute for the sermon
itself. We deliberately do not generate "Conclusion" or "Application" sections,
because those read as a verdict and would supplant a person wrestling with the
text. The frontend keeps these collapsed and de-emphasized, and labels them as
auto-generated.

The Introduction doubles as the sermon's "description", the human-readable line
shown when browsing/searching, so a result is evaluable without watching. The
Themes power keyword search alongside the description and outline.

Design (mirrors tools/generate_reference_summaries.py + the run_ingest.sh step):
  * Incremental/idempotent — a sermon is processed only if its metadata lacks a
    `notes` block or carries an older `notes_schema_version`. Weekly runs touch
    only the new sermon(s).
  * Input is the FULL transcript text (not retrieved chunks) so the outline
    reflects the whole message.
  * Output is validated before it is written: required fields present, outline
    is a list, introduction non-empty. On any validation failure NOTHING is
    written for that sermon, so the next run retries rather than persisting
    garbage.
  * Strict grounding — every section is derived only from the transcript, and
    Application captures only applications the preacher actually stated.

Phase-0 review (writes a samples file, does NOT touch metadata):
    python tools/generate_sermon_notes.py --video ID [--video ID ...] \
        --out tools/notes_samples.json

Real mode (merges the `notes` block into the metadata JSON in place):
    python tools/generate_sermon_notes.py --all --write
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import openai

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# --- paths (relative to repo root; this file lives at REPO/tools/) ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPT_DIR = os.path.join(REPO_ROOT, "transcription", "data", "transcripts")
METADATA_DIR = os.path.join(REPO_ROOT, "transcription", "data", "metadata")

NOTES_SCHEMA_VERSION = 2
DEFAULT_MODEL = os.environ.get("NOTES_MODEL", "anthropic/claude-sonnet-4.6")


SYSTEM_PROMPT = """You are cataloging a single sermon from an Independent \
Fundamental Baptist church, working ONLY from the verbatim transcript you are \
given. Like a librarian describing a book, you produce concise, descriptive \
metadata that helps someone find and orient to the sermon. You are NOT a \
commentator: you add no theology of your own, and you do NOT supply the \
sermon's conclusions, applications, or takeaways. The point is to help people \
find the sermon, not to spare them from engaging with it.

Hard rules:
1. Use ONLY what is actually said in this transcript. Never add outside \
   doctrine, illustrations, or conclusions the preacher did not state.
2. Do not invent dates, names, or scripture references. Only include a Bible \
   reference if the preacher actually preaches from or cites it.
3. NEVER use em dashes or en dashes. Use commas, periods, colons, or \
   parentheses instead. (Plain hyphens in verse ranges like "vv. 1-15" are \
   fine.)
4. Keep everything tight, plain, and descriptive. No emojis, no flowery \
   language, no exhortation. Describe what the sermon covers, do not tell the \
   reader what to do about it.
5. Introduction: 1 to 2 sentences, a standalone description of what the sermon \
   is about (it is shown when browsing). Name the main passage if the preacher \
   gives one. Descriptive, not a takeaway.
6. Outline: the sermon's actual main divisions, in the order he preached them, \
   as he framed them (keep his numbering or alliteration if he used it). You \
   may add a short descriptive clause after a colon. Capture only genuine \
   divisions; do not manufacture points to hit a number.
7. Themes: 3 to 6 short subject keywords or phrases describing what the sermon \
   is about (for cataloging and search), for example "prayer", "standing for \
   convictions", "the home". Descriptive subjects, not lessons or takeaways.

Return ONLY a JSON object with exactly this shape (no prose, no code fence):
{
  "introduction": "1-2 sentence standalone description; name the main passage if stated",
  "outline": ["I. ...", "II. ...", "..."],
  "themes": ["subject", "subject", "..."]
}"""


def build_user_prompt(title: str, transcript: str) -> str:
    return (
        f"SERMON TITLE: {title}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        "Produce the JSON catalog record now, following every hard rule. "
        "Describe the sermon; do not supply its conclusions or applications, "
        "and never use em dashes."
    )


def extract_json(raw: str) -> Dict[str, Any]:
    """Pull the JSON object out of a model response, tolerating stray prose or a
    ```json fence."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    if not raw.startswith("{"):
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
    return json.loads(raw)


def no_dashes(s: str) -> str:
    """Backstop the prompt rule: replace any en/em dash with a comma so none
    ever reach the page. Plain hyphens (verse ranges) are left alone."""
    return re.sub(r"\s*[—–]\s*", ", ", str(s)).strip()


def validate_and_clean(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Validate against the schema. Raises ValueError on anything we will not
    persist. Returns the cleaned `notes` block (wrapped with schema version)."""
    required = ["introduction", "outline", "themes"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if not isinstance(obj["outline"], list):
        raise ValueError("outline must be a list")
    if not isinstance(obj["themes"], list):
        raise ValueError("themes must be a list")

    notes = {
        "introduction": no_dashes(obj["introduction"]),
        "outline": [no_dashes(x) for x in obj["outline"] if str(x).strip()],
        "themes": [no_dashes(t).lower() for t in obj["themes"] if str(t).strip()],
    }
    if not notes["introduction"]:
        raise ValueError("introduction came back empty")
    return {"notes_schema_version": NOTES_SCHEMA_VERSION, "notes": notes}


def load_transcript(video_id: str) -> Optional[str]:
    path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return (json.load(f).get("text") or "").strip() or None


def load_metadata(video_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(METADATA_DIR, f"{video_id}_metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def needs_processing(meta: Dict[str, Any]) -> bool:
    return meta.get("notes_schema_version", 0) < NOTES_SCHEMA_VERSION


def generate_for(video_id: str, client: openai.OpenAI, model: str) -> Dict[str, Any]:
    meta = load_metadata(video_id)
    if meta is None:
        raise FileNotFoundError(f"no metadata for {video_id}")
    transcript = load_transcript(video_id)
    if not transcript:
        raise FileNotFoundError(f"no transcript text for {video_id}")
    title = meta.get("title", video_id)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(title, transcript)},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    cleaned = validate_and_clean(extract_json(resp.choices[0].message.content))
    cleaned["video_id"] = video_id
    cleaned["title"] = title
    usage = getattr(resp, "usage", None)
    if usage is not None:
        cleaned["_usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
    return cleaned


def select_videos(args) -> List[str]:
    if args.video:
        return args.video
    metas = []
    for p in glob.glob(os.path.join(METADATA_DIR, "*_metadata.json")):
        try:
            metas.append(json.load(open(p, encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    metas.sort(key=lambda m: str(m.get("publish_date") or ""), reverse=True)
    todo = [m["video_id"] for m in metas
            if m.get("video_id") and (not args.write or needs_processing(m))]
    if args.sample and len(todo) > args.sample:
        # Spread the sample across the corpus for variety, not just newest.
        step = len(todo) // args.sample
        todo = [todo[i * step] for i in range(args.sample)]
    return todo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", action="append", help="video_id (repeatable)")
    ap.add_argument("--sample", type=int, help="process N sermons spread across the corpus")
    ap.add_argument("--all", action="store_true", help="process every sermon needing notes")
    ap.add_argument("--write", action="store_true",
                    help="merge results into the metadata JSON in place (real mode)")
    ap.add_argument("--out", default="tools/notes_samples.json",
                    help="dry-run review file (when --write is not set)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not (args.video or args.sample or args.all):
        ap.error("specify --video, --sample N, or --all")

    if load_dotenv:
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1
    client = openai.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

    videos = select_videos(args)
    print(f"Model: {args.model}  |  {len(videos)} sermon(s)  |  "
          f"{'WRITE to metadata' if args.write else 'DRY-RUN to ' + args.out}\n")

    samples, ok, fail = [], 0, 0
    for vid in videos:
        try:
            result = generate_for(vid, client, args.model)
        except Exception as e:  # keep going; a failure just means retry next run
            print(f"  x {vid}: {e}")
            fail += 1
            continue
        ok += 1
        usage = result.pop("_usage", None)
        print(f"  ok {vid}  {result['title'][:60]}"
              + (f"   ({usage['prompt_tokens']}/{usage['completion_tokens']} tok)" if usage else ""))

        if args.write:
            meta = load_metadata(vid)
            meta["notes_schema_version"] = result["notes_schema_version"]
            meta["notes"] = result["notes"]
            path = os.path.join(METADATA_DIR, f"{vid}_metadata.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            samples.append(result)

    if not args.write:
        out_path = os.path.join(REPO_ROOT, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {len(samples)} sample(s) to {args.out} for review.")

    print(f"\nDone. {ok} ok, {fail} failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
