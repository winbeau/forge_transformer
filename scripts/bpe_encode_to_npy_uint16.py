#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
from pathlib import Path
from typing import BinaryIO, List, Tuple

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, TimeoutError

from forge_transformer import BASE_DIR
from forge_transformer.bpe import Tokenizer


# -----------------------------
# Module 1: Chunk boundary finder
# -----------------------------
def find_chunk_boundaries_after_token(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    mini_chunk_size: int = 4096,
    include_newline: bool = True,
) -> List[int]:
    assert isinstance(split_special_token, bytes)
    tok = split_special_token
    L = len(tok)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0 or desired_num_chunks <= 1:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[0] = 0
    boundaries[-1] = file_size

    for bi in range(1, len(boundaries) - 1):
        guess = boundaries[bi]
        start_pos = max(0, guess - (L - 1))
        file.seek(start_pos)

        carry = b""
        while True:
            data = file.read(mini_chunk_size)
            if data == b"":
                boundaries[bi] = file_size
                break

            window = carry + data
            idx = window.find(tok)
            if idx != -1:
                data_start = file.tell() - len(data)
                token_end = data_start - len(carry) + idx + L

                if include_newline and token_end < file_size:
                    file.seek(token_end)
                    nxt = file.read(2)
                    if nxt.startswith(b"\r\n"):
                        token_end += 2
                    elif nxt.startswith(b"\n"):
                        token_end += 1

                boundaries[bi] = min(token_end, file_size)
                break

            if len(window) >= (L - 1):
                carry = window[-(L - 1):]
            else:
                carry = window

    cleaned = [0]
    for b in boundaries[1:]:
        if b > cleaned[-1]:
            cleaned.append(b)
    if cleaned[-1] != file_size:
        cleaned.append(file_size)
    return cleaned


# -----------------------------
# Module 2: Multiprocessing init + probe
# -----------------------------
_tokenizer = None

def init_worker(vocab_path: str, merges_path: str, special_tokens: List[str]):
    global _tokenizer
    _tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

def worker_ping(_):
    # 用于确认 worker 已启动并完成 init
    return (os.getpid(), time.time())


# -----------------------------
# Module 3: Span processing helpers
# -----------------------------
def _read_span_bytes(input_path: str, start: int, end: int) -> bytes:
    with open(input_path, "rb") as f:
        f.seek(start)
        return f.read(end - start)

def _encode_text_linewise(text: str) -> List[int]:
    """
    关键优化：把 chunk 再按行切开，复用你单行脚本的快路径
    """
    global _tokenizer
    out: List[int] = []
    # keepends=True，确保换行不丢（虽然一般不重要，但和原文件一致性更好）
    for line in text.splitlines(True):
        if line.strip():
            out.extend(_tokenizer.encode(line))
    return out


# -----------------------------
# Module 4: Worker jobs (pass1 count / pass2 write)
# -----------------------------
def worker_count(args) -> Tuple[int, int, int, int, float, int]:
    span_i, input_path, start, end = args
    global _tokenizer

    t0 = time.time()
    span_bytes = end - start
    pid = os.getpid()

    data = _read_span_bytes(input_path, start, end)
    text = data.decode("utf-8", errors="strict")

    ids = _encode_text_linewise(text)
    max_id = int(max(ids)) if ids else -1

    dt = time.time() - t0
    return (span_i, len(ids), max_id, span_bytes, dt, pid)

def worker_encode_write(args) -> Tuple[int, int, int, float, int]:
    span_i, input_path, start, end, output_npy_path, offset, expected_count = args
    global _tokenizer

    t0 = time.time()
    span_bytes = end - start
    pid = os.getpid()

    data = _read_span_bytes(input_path, start, end)
    text = data.decode("utf-8", errors="strict")

    ids = _encode_text_linewise(text)
    if len(ids) != expected_count:
        raise RuntimeError(
            f"[span {span_i}] token count mismatch: got {len(ids)} != expected {expected_count}"
        )

    arr = np.load(output_npy_path, mmap_mode="r+")
    arr[offset : offset + expected_count] = np.asarray(ids, dtype=np.uint16)
    arr.flush()

    dt = time.time() - t0
    return (span_i, expected_count, span_bytes, dt, pid)


# -----------------------------
# Module 5: CLI + Orchestration
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-txt-path", type=Path, required=True)
    p.add_argument("--output-npy-path", type=Path, required=True)
    p.add_argument("--vocab-path", type=Path, default=BASE_DIR / "bpe_model/vocab.json")
    p.add_argument("--merges-path", type=Path, default=BASE_DIR / "bpe_model/merges.txt")
    p.add_argument("--special-tokens", type=str, default="<|endoftext|>")
    p.add_argument("--num-workers", type=int, default=max(1, cpu_count() - 2))

    p.add_argument("--desired-chunks", type=int, default=0, help="0 = auto (num_workers*40)")
    p.add_argument("--mini-chunk-size", type=int, default=4096)
    p.add_argument("--no-include-newline", action="store_true")

    p.add_argument("--heartbeat-seconds", type=float, default=15.0)
    p.add_argument("--imap-chunksize", type=int, default=1)
    p.add_argument("--probe-workers", action="store_true", help="probe worker init immediately")
    return p.parse_args()

def _bytes_to_mib(b: int) -> float:
    return b / 1024 / 1024

def main():
    args = parse_args()

    input_path = args.input_txt_path
    output_path = args.output_npy_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    special_tokens = [s for s in str(args.special_tokens).split(",") if s] or ["<|endoftext|>"]

    num_workers = int(args.num_workers)
    desired_chunks = int(args.desired_chunks) if int(args.desired_chunks) > 0 else max(num_workers * 40, num_workers)
    include_newline = not args.no_include_newline
    heartbeat = float(args.heartbeat_seconds)
    imap_chunksize = max(1, int(args.imap_chunksize))

    split_tok = special_tokens[0].encode("utf-8")
    file_size_bytes = input_path.stat().st_size

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries_after_token(
            f,
            desired_num_chunks=desired_chunks,
            split_special_token=split_tok,
            mini_chunk_size=int(args.mini_chunk_size),
            include_newline=include_newline,
        )

    spans_raw = [(i, str(input_path), boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    spans = [s for s in spans_raw if s[2] < s[3]]

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Workers: {num_workers}, spans: {len(spans)} (desired_chunks={desired_chunks})")
    print(f"File size: {_bytes_to_mib(file_size_bytes):.1f} MiB, avg span: {_bytes_to_mib(file_size_bytes/max(len(spans),1)):.2f} MiB")

    # -------- Pass1 --------
    counts = [0] * len(spans)
    max_id_seen = -1
    done_spans = 0
    done_bytes = 0
    running_tokens = 0
    t_start = time.time()

    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(str(args.vocab_path), str(args.merges_path), special_tokens),
    ) as pool:

        if args.probe_workers:
            t0 = time.time()
            # 让你立刻看到 worker 是否完成初始化
            pings = pool.map(worker_ping, range(num_workers))
            dt = time.time() - t0
            pids = [pid for pid, _ in pings]
            print(f"[probe] {num_workers} workers ready in {dt:.2f}s. pids={pids[:5]}{'...' if len(pids)>5 else ''}")

        it = pool.imap_unordered(worker_count, spans, chunksize=imap_chunksize)

        with tqdm(
            total=file_size_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Pass1: counting",
            mininterval=0.5,
        ) as pbar:
            while done_spans < len(spans):
                try:
                    span_i, cnt, max_id, span_bytes, dt, pid = it.next(timeout=heartbeat)
                except TimeoutError:
                    tqdm.write(f"[heartbeat] Pass1 still running... finished {done_spans}/{len(spans)} spans so far.")
                    continue

                counts[span_i] = cnt
                if max_id > max_id_seen:
                    max_id_seen = max_id

                done_spans += 1
                done_bytes += span_bytes
                running_tokens += cnt

                pbar.update(span_bytes)
                elapsed = time.time() - t_start
                pbar.set_postfix(
                    spans=f"{done_spans}/{len(spans)}",
                    tokens=f"{running_tokens:,}",
                    span_s=f"{dt:.1f}",
                    pid=str(pid),
                    MiBps=f"{_bytes_to_mib(done_bytes)/max(elapsed,1e-6):.1f}",
                )

    total_tokens = int(sum(counts))
    if total_tokens == 0:
        raise RuntimeError("No tokens produced. Check input file / tokenizer.")

    if max_id_seen >= 65536:
        raise RuntimeError(
            f"Token id {max_id_seen} exceeds uint16 range. Need uint32 or rebuild vocab."
        )

    print(f"Pass1 done. Total tokens: {total_tokens:,} (max_id={max_id_seen})")

    # allocate true .npy
    mmap_arr = np.lib.format.open_memmap(
        filename=str(output_path),
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )
    mmap_arr.flush()
    del mmap_arr

    # offsets
    offsets = [0] * len(counts)
    running = 0
    for i, c in enumerate(counts):
        offsets[i] = running
        running += c

    # -------- Pass2 --------
    write_jobs = []
    for i, (_, in_path, start, end) in enumerate(spans):
        write_jobs.append((i, in_path, start, end, str(output_path), offsets[i], counts[i]))

    done_spans = 0
    done_bytes = 0
    done_tokens = 0
    t_start = time.time()

    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(str(args.vocab_path), str(args.merges_path), special_tokens),
    ) as pool:
        it = pool.imap_unordered(worker_encode_write, write_jobs, chunksize=imap_chunksize)

        with tqdm(
            total=file_size_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Pass2: encode+write",
            mininterval=0.5,
        ) as pbar:
            while done_spans < len(write_jobs):
                try:
                    span_i, written, span_bytes, dt, pid = it.next(timeout=heartbeat)
                except TimeoutError:
                    tqdm.write(f"[heartbeat] Pass2 still running... finished {done_spans}/{len(write_jobs)} spans so far.")
                    continue

                done_spans += 1
                done_bytes += span_bytes
                done_tokens += written

                pbar.update(span_bytes)
                elapsed = time.time() - t_start
                pbar.set_postfix(
                    spans=f"{done_spans}/{len(write_jobs)}",
                    tokens=f"{done_tokens:,}",
                    span_s=f"{dt:.1f}",
                    pid=str(pid),
                    MiBps=f"{_bytes_to_mib(done_bytes)/max(elapsed,1e-6):.1f}",
                )

    print(f"Done! Saved uint16 tokens to: {output_path}")
    print("Load with: arr = np.load(path, mmap_mode='r')")


if __name__ == "__main__":
    main()

