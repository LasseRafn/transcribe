#!/usr/bin/env python3
"""Voice-to-Documentation CLI: narrate your work, get structured Jira-ready docs."""

import argparse
import math
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import anthropic
import numpy as np
import openai
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
WHISPER_MAX_BYTES = 25 * 1024 * 1024  # 25 MB
CHUNK_DURATION_SEC = 10 * 60  # 10-minute chunks

SUMMARIZE_PROMPT = """\
You are a documentation assistant. You will receive a raw transcript of someone \
narrating their customer-support / debugging work. The speaker references Jira \
ticket numbers (like PROJ-1234, DATA-567, etc.) as they move between tasks.

Your job:
1. Identify every Jira ticket number mentioned in the transcript.
2. Group the actions by ticket.
3. For each ticket, produce a clean, numbered step-by-step list of what was done.
4. If any actions don't reference a specific ticket, group them under "## General".
5. Use past tense, be concise, and preserve technical details (table names, field \
   names, IDs, values).

Output format (markdown):

## PROJ-1234
1. Step one
2. Step two

## OTHER-5678
1. Step one

Do NOT include any preamble or explanation — only the grouped steps.
"""


def record_audio(output_path: Path) -> None:
    """Record audio from microphone to a WAV file. Press Enter to stop."""
    print("Recording... press Enter to stop.\n")

    lock = threading.Lock()
    frames_written = 0
    stop_event = threading.Event()

    wf = wave.open(str(output_path), "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16-bit = 2 bytes
    wf.setframerate(SAMPLE_RATE)

    def callback(indata, frame_count, time_info, status):
        nonlocal frames_written
        if status:
            print(f"  ⚠ {status}", file=sys.stderr)
        wf.writeframes(indata.tobytes())
        with lock:
            frames_written += frame_count

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    )
    stream.start()

    # Elapsed time display in a background thread
    def show_elapsed():
        start = time.monotonic()
        while not stop_event.is_set():
            elapsed = int(time.monotonic() - start)
            mins, secs = divmod(elapsed, 60)
            with lock:
                current_frames = frames_written
            file_mb = (current_frames * 2) / (1024 * 1024)
            print(f"\r  Elapsed: {mins:02d}:{secs:02d}  |  ~{file_mb:.1f} MB", end="", flush=True)
            stop_event.wait(timeout=1.0)

    timer_thread = threading.Thread(target=show_elapsed, daemon=True)
    timer_thread.start()

    input()  # Block until Enter

    stop_event.set()
    stream.stop()
    stream.close()
    wf.close()

    with lock:
        total_frames = frames_written
    duration = total_frames / SAMPLE_RATE
    mins, secs = divmod(int(duration), 60)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n\nRecording saved: {output_path} ({mins}m{secs}s, {file_size_mb:.1f} MB)")


def chunk_wav(wav_path: Path) -> list[Path]:
    """Split a WAV into chunks of ~CHUNK_DURATION_SEC if it exceeds WHISPER_MAX_BYTES."""
    file_size = wav_path.stat().st_size
    if file_size <= WHISPER_MAX_BYTES:
        return [wav_path]

    chunks = []
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        total_frames = wf.getnframes()
        frames_per_chunk = CHUNK_DURATION_SEC * framerate
        n_chunks = math.ceil(total_frames / frames_per_chunk)

        print(f"File exceeds 25 MB — splitting into {n_chunks} chunks...")

        for i in range(n_chunks):
            chunk_path = wav_path.with_suffix(f".chunk{i}.wav")
            frames_to_read = min(frames_per_chunk, total_frames - i * frames_per_chunk)
            data = wf.readframes(frames_to_read)

            with wave.open(str(chunk_path), "wb") as chunk_wf:
                chunk_wf.setnchannels(n_channels)
                chunk_wf.setsampwidth(sampwidth)
                chunk_wf.setframerate(framerate)
                chunk_wf.writeframes(data)

            chunks.append(chunk_path)

    return chunks


def transcribe(audio_path: Path) -> str:
    """Transcribe audio via Whisper API, chunking if necessary."""
    client = openai.OpenAI()
    chunks = chunk_wav(audio_path)
    transcripts = []

    for i, chunk_path in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  Transcribing chunk {i + 1}/{len(chunks)}...")
        else:
            print("Transcribing...")

        with open(chunk_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        transcripts.append(result.text)

        # Clean up chunk files (but not the original)
        if chunk_path != audio_path:
            chunk_path.unlink()

    full_transcript = " ".join(transcripts)
    print(f"Transcription complete ({len(full_transcript)} characters).\n")
    return full_transcript


def summarize(transcript: str) -> str:
    """Send transcript to Claude for structured summarization."""
    print("Summarizing with Claude...")
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": f"{SUMMARIZE_PROMPT}\n\n---\n\nTRANSCRIPT:\n{transcript}",
            }
        ],
    )
    summary = message.content[0].text
    print("Summary complete.\n")
    return summary


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard via pbcopy (macOS)."""
    try:
        proc = subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def save_log(transcript: str, summary: str | None) -> Path:
    """Save transcript and summary to ./logs/ with timestamp."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"{timestamp}.md"

    parts = [f"# Audio Log — {timestamp}\n"]
    parts.append("## Raw Transcript\n")
    parts.append(transcript)
    if summary:
        parts.append("\n\n## Summary\n")
        parts.append(summary)

    log_path.write_text("\n".join(parts), encoding="utf-8")
    return log_path


def main():
    parser = argparse.ArgumentParser(
        description="Voice-to-Documentation: narrate your work, get Jira-ready docs."
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Record and transcribe only — skip Claude summarization.",
    )
    parser.add_argument(
        "--transcript-only",
        metavar="FILE",
        help="Transcribe an existing audio file instead of recording.",
    )
    args = parser.parse_args()

    # Determine audio path
    if args.transcript_only:
        audio_path = Path(args.transcript_only)
        if not audio_path.exists():
            print(f"Error: file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)
    else:
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        audio_path = recordings_dir / f"{timestamp}.wav"
        record_audio(audio_path)

    # Transcribe
    transcript = transcribe(audio_path)
    print("=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)
    print(transcript)
    print()

    # Summarize
    summary = None
    if not args.no_summary:
        summary = summarize(transcript)
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(summary)
        print()

        if copy_to_clipboard(summary):
            print("Summary copied to clipboard.")
        else:
            print("Could not copy to clipboard (pbcopy not available).")

    # Save log
    log_path = save_log(transcript, summary)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
