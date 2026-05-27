"""Generate a "point summary" footnote for each Bible reference.

For every reference in transcription/data/bible_references/*.json, fetch the
sermon transcript window around the citation timestamp and ask an LLM to
write a one-sentence study-Bible-style footnote describing the theological
or textual point being drawn from the verse in that sermon moment.

The result is stored as `point_summary` (and `point_summary_model`) on the
reference record itself, alongside `context`, `is_implicit`, etc. The
frontend reads it from the same per-book JSON file the API already serves.

Idempotent — only summarizes refs that don't already have a `point_summary`
field. Safe to run repeatedly as new sermons are ingested.

Usage:
  python tools/generate_reference_summaries.py
  python tools/generate_reference_summaries.py --book Romans.json --limit 10
  python tools/generate_reference_summaries.py --model anthropic/claude-haiku-4.5
"""
import os
import sys
import json
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


REFS_DIR = "transcription/data/bible_references"
TX_DIR = "transcription/data/transcripts"
META_DIR = "transcription/data/metadata"

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"

# ---- Prompt ----
# Calibrated against ~14 sample refs across famous + obscure + hard cases.
# Lead with theology not "the speaker", keep <=30 words, refuse to summarize
# when context is genuinely thin. The "doorway into the sermon" framing
# matters — these notes are study-Bible marginalia, not stand-alone takes.
SYSTEM_MESSAGE = (
    "You write study-Bible footnotes for a digital sermon library. "
    "Each note sits beside a verse and serves as a doorway into the sermon "
    "where that verse was preached. Write like an editor of a serious study "
    "Bible: lead with the theological or textual point being drawn from the "
    "verse, and let the sermon's framing show through. Be exact. Never put "
    "words in the preacher's mouth he did not say."
)


def build_user_message(citation, sermon_title, transcript_window_text):
    return f"""The verse below was cited or quoted in a sermon titled "{sermon_title}". A roughly 2-3 minute window of the actual transcript follows.

Write ONE sentence (max ~30 words) that captures the theological or textual point being drawn from {citation} in this sermon moment.

Rules:
- Do NOT begin with "The speaker", "The pastor", "The preacher", or "Pastor". Lead with the truth or theme itself.
- Stay specific to what is actually said in the transcript window. No doctrinal commentary that isn't voiced.
- If the verse is mentioned only in passing without development, write: "Brief mention; no exposition in this segment."
- Output ONLY the sentence. No quotes around it, no preface.

Verse: {citation}
Transcript window:
{transcript_window_text}
"""


def transcript_window(transcript, start_sec, secs_before=60, secs_after=90):
    """Concatenate segment text within [start - before, start + after] seconds."""
    if not transcript or "segments" not in transcript:
        return ""
    lo = start_sec - secs_before
    hi = start_sec + secs_after
    parts = []
    for seg in transcript["segments"]:
        s = seg.get("start", 0)
        e = seg.get("end", s)
        if e < lo or s > hi:
            continue
        text = (seg.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def citation_string(ref):
    book = ref.get("book") or ""
    chapter = ref.get("chapter")
    verse = ref.get("verse")
    s = book
    if chapter is not None and chapter != "":
        s += f" {chapter}"
        if verse is not None and verse != "":
            s += f":{verse}"
    return s.strip()


class SummaryGenerator:
    """Adds `point_summary` to refs that don't have one, in-place per book file."""

    def __init__(self, client, model, max_workers=4):
        self.client = client
        self.model = model
        self.max_workers = max_workers
        self.transcript_cache = {}
        self.title_cache = {}

    def get_transcript(self, video_id):
        if video_id not in self.transcript_cache:
            path = os.path.join(TX_DIR, f"{video_id}.json")
            try:
                self.transcript_cache[video_id] = json.load(open(path))
            except (FileNotFoundError, json.JSONDecodeError):
                self.transcript_cache[video_id] = None
        return self.transcript_cache[video_id]

    def get_title(self, video_id):
        if video_id not in self.title_cache:
            path = os.path.join(META_DIR, f"{video_id}_metadata.json")
            title = ""
            try:
                meta = json.load(open(path))
                title = meta.get("title") or ""
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            self.title_cache[video_id] = title
        return self.title_cache[video_id]

    def summarize(self, ref):
        """Return summary string, or None if we couldn't generate one."""
        video_id = ref.get("video_id")
        start = ref.get("start_time")
        if not video_id or start is None:
            return None
        transcript = self.get_transcript(video_id)
        if not transcript:
            return None
        window = transcript_window(transcript, start)
        if len(window) < 100:
            return None
        title = self.get_title(video_id) or "(untitled sermon)"
        citation = citation_string(ref)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": build_user_message(citation, title, window)},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            text = (response.choices[0].message.content or "").strip()
            # Models sometimes wrap output in quotes despite instructions.
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1].strip()
            return text or None
        except Exception as e:
            print(f"  error on {video_id}@{start}: {e}", file=sys.stderr)
            return None

    def process_book_file(self, file_path, limit=None):
        """Adds point_summary fields in-place. Saves periodically.

        Returns (n_new, n_kept, n_skipped).
        """
        with open(file_path) as f:
            try:
                refs = json.load(f)
            except json.JSONDecodeError:
                print(f"  skip (bad JSON): {file_path}", file=sys.stderr)
                return 0, 0, 0
        if not isinstance(refs, list):
            return 0, 0, 0

        to_process = [(i, r) for i, r in enumerate(refs) if not r.get("point_summary")]
        if limit:
            to_process = to_process[:limit]
        if not to_process:
            return 0, len(refs), 0

        book_name = os.path.basename(file_path).replace(".json", "")
        n_new = 0
        n_skipped = 0
        done = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self.summarize, r): i for i, r in to_process}
            with tqdm(total=len(futures), desc=f"  {book_name}", leave=False) as bar:
                for fut in as_completed(futures):
                    i = futures[fut]
                    summary = fut.result()
                    if summary:
                        refs[i]["point_summary"] = summary
                        refs[i]["point_summary_model"] = self.model
                        n_new += 1
                    else:
                        n_skipped += 1
                    done += 1
                    bar.update(1)
                    # Periodic checkpoint so a crash doesn't lose hours of work.
                    if done % 25 == 0:
                        self._save(file_path, refs)
        self._save(file_path, refs)
        return n_new, len(refs) - len(to_process), n_skipped

    @staticmethod
    def _save(file_path, refs):
        tmp = file_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(refs, f, indent=2, ensure_ascii=False)
        os.replace(tmp, file_path)

    def process_all(self, limit_per_book=None):
        files = sorted(glob.glob(os.path.join(REFS_DIR, "*.json")))
        files = [f for f in files if "processed_files" not in os.path.basename(f)]
        total_new = total_kept = total_skipped = 0
        for fp in files:
            n_new, n_kept, n_skipped = self.process_book_file(fp, limit=limit_per_book)
            if n_new or n_skipped:
                print(f"  {os.path.basename(fp):28s}  +{n_new} new, {n_kept} kept, {n_skipped} skipped")
            total_new += n_new
            total_kept += n_kept
            total_skipped += n_skipped
        print(f"\nTotal: {total_new} new summaries, {total_kept} already-summarized, "
              f"{total_skipped} couldn't be generated (missing transcript or thin window).")


def main():
    parser = argparse.ArgumentParser(description="Add AI footnote summaries to Bible references.")
    parser.add_argument("--model", default=os.environ.get("SUMMARY_MODEL", DEFAULT_MODEL),
                        help=f"OpenRouter model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel API requests (default: 4)")
    parser.add_argument("--book", help="Process only this book file (e.g. Romans.json)")
    parser.add_argument("--limit", type=int,
                        help="Limit number of refs to summarize per book in this run")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: OPENROUTER_API_KEY env var is required.")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    print(f"Model: {args.model}")
    print(f"Workers: {args.max_workers}\n")

    gen = SummaryGenerator(client, args.model, max_workers=args.max_workers)
    if args.book:
        path = os.path.join(REFS_DIR, args.book)
        if not os.path.exists(path):
            sys.exit(f"Not found: {path}")
        n_new, n_kept, n_skipped = gen.process_book_file(path, limit=args.limit)
        print(f"\n{args.book}: +{n_new} new, {n_kept} kept, {n_skipped} skipped")
    else:
        gen.process_all(limit_per_book=args.limit)


if __name__ == "__main__":
    main()
