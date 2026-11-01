"""A failed Whisper call must fail the sermon, never produce an empty or partial transcript."""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "transcription"))

import transcribe_audio as ta  # noqa: E402


def chunk(text):
    return {"text": text, "segments": [{"start": 0.0, "end": 1.0, "text": text}]}


class TranscribeChunkTests(unittest.TestCase):
    def test_api_error_returns_none(self):
        with mock.patch.object(ta, "OpenAI") as client, \
             mock.patch("builtins.open", mock.mock_open(read_data=b"x")):
            client.return_value.audio.transcriptions.create.side_effect = RuntimeError("429")
            self.assertIsNone(ta.transcribe_chunk("a.mp3", "whisper-1", "en"))


class TranscribeAudioTests(unittest.TestCase):
    def run_chunks(self, results):
        """Run transcribe_audio on an oversized file whose chunks return `results`."""
        with mock.patch.object(ta.os.path, "exists", return_value=True), \
             mock.patch.object(ta, "get_file_size_mb", return_value=ta.MAX_FILE_SIZE_MB + 1), \
             mock.patch.object(ta, "split_audio_file", return_value=[f"c{i}.mp3" for i in range(len(results))]), \
             mock.patch.object(ta, "transcribe_chunk", side_effect=results) as transcribe:
            return ta.transcribe_audio("big.mp3"), transcribe

    def test_all_chunks_failing_returns_none(self):
        result, _ = self.run_chunks([None, None])
        self.assertIsNone(result)

    def test_one_failed_chunk_fails_whole_file_and_stops_spending(self):
        result, transcribe = self.run_chunks([chunk("one"), None, chunk("three")])
        self.assertIsNone(result)
        self.assertEqual(transcribe.call_count, 2)

    def test_all_chunks_succeeding_merges(self):
        result, _ = self.run_chunks([chunk("one"), chunk("two")])
        self.assertIn("one", result["text"])
        self.assertIn("two", result["text"])


if __name__ == "__main__":
    unittest.main()
