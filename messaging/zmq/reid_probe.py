#!/usr/bin/env python3
"""
Diagnostic probe for the ReID event stream.

Reads 500 frames from the ZMQ stream and reports:
  - source_id distribution
  - embedding presence rate
  - embedding norm statistics
  - pairwise cosine similarity distribution (same-camera, to reveal embedding quality
    independent of the cross-camera requirement)

Usage:
    python3 reid_probe.py [--endpoint tcp://localhost:5555] [--frames 500]
"""
import argparse
import base64
import json
import sys
from collections import Counter

import numpy as np
import zmq


def decode_emb(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp://localhost:5555")
    ap.add_argument("--frames", type=int, default=500)
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.connect(args.endpoint)
    print(f"[probe] connected to {args.endpoint}, collecting {args.frames} frames …\n")

    source_ids   = Counter()
    total_dets   = 0
    with_emb     = 0
    without_emb  = 0
    norms        = []
    embeddings   = []   # (source_id, tid, emb)

    frames = 0
    while frames < args.frames:
        try:
            raw = sock.recv_string()
        except zmq.Again:
            print("[probe] timeout waiting for frames — is the pipeline running?")
            sys.exit(1)

        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue

        src = ev.get("source_id", -1)
        source_ids[src] += 1
        frames += 1

        for det in ev.get("detections", []):
            if det.get("class") != 0:   # person only
                continue
            total_dets += 1
            if "emb" not in det:
                without_emb += 1
                continue
            with_emb += 1
            emb = decode_emb(det["emb"])
            norms.append(float(np.linalg.norm(emb)))
            embeddings.append((src, det.get("tid"), emb))

    sock.close()
    ctx.term()

    # ── Report ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SOURCE ID DISTRIBUTION")
    print("=" * 60)
    for sid, cnt in sorted(source_ids.items()):
        print(f"  source_id={sid}  frames={cnt}")
    unique_sources = len(source_ids)
    print(f"\n  → {unique_sources} unique source(s) seen")
    if unique_sources < 2:
        print("  ⚠  Only 1 source detected. Cross-camera matching is impossible.")
        print("     Enable a second stream in configs/default.yaml.")

    print()
    print("=" * 60)
    print("EMBEDDING PRESENCE")
    print("=" * 60)
    print(f"  Person detections : {total_dets}")
    print(f"  With embedding    : {with_emb}  ({100*with_emb/max(total_dets,1):.1f}%)")
    print(f"  Without embedding : {without_emb}")
    if without_emb == total_dets:
        print("  ⚠  No embeddings received at all.")
        print("     Check that the SGIE is running and output-tensor-meta=1 is set.")

    if not norms:
        print("\n[probe] No embeddings to analyse. Exiting.")
        sys.exit(0)

    print()
    print("=" * 60)
    print("EMBEDDING NORM STATISTICS")
    print("=" * 60)
    norms = np.array(norms)
    print(f"  min={norms.min():.3f}  max={norms.max():.3f}  "
          f"mean={norms.mean():.3f}  std={norms.std():.3f}")
    print(f"  (L2-normalised embeddings should have norm ≈ 1.0)")
    if norms.mean() < 0.9 or norms.mean() > 1.1:
        print("  ⚠  Norms are not near 1.0 — embeddings may not be L2-normalised.")

    print()
    print("=" * 60)
    print("PAIRWISE COSINE SIMILARITY")
    print("=" * 60)
    # Normalise all embeddings
    embs_arr = np.stack([e for _, _, e in embeddings])
    embs_arr = embs_arr / (np.linalg.norm(embs_arr, axis=1, keepdims=True) + 1e-8)
    keys = [(s, t) for s, t, _ in embeddings]

    # Split pairs into intra-ID (same source+tid) and inter-ID (different tid)
    rng = np.random.default_rng(0)
    n = len(embs_arr)
    all_idx_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_idx_pairs)

    intra, inter = [], []
    for i, j in all_idx_pairs:
        sim = float(embs_arr[i] @ embs_arr[j])
        if keys[i] == keys[j]:      # same camera + same tracking ID
            intra.append(sim)
        else:
            inter.append(sim)
        if len(intra) >= 200 and len(inter) >= 200:
            break

    def _stats(label, sims):
        if not sims:
            print(f"  {label}: no pairs")
            return
        a = np.array(sims)
        print(f"  {label} ({len(a)} pairs):")
        print(f"    min={a.min():.3f}  max={a.max():.3f}  mean={a.mean():.3f}  std={a.std():.3f}")

    _stats("Intra-ID (same person, stability check)", intra)
    _stats("Inter-ID (different persons, discrimination)", inter)

    # Threshold breakdown on all sampled pairs combined
    all_sims = np.array(intra + inter)
    print(f"\n  Threshold breakdown (all {len(all_sims)} sampled pairs):")
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    for t in thresholds:
        frac = (all_sims >= t).mean()
        bar = "█" * int(frac * 40)
        print(f"    ≥{t:.2f}  {frac:5.1%}  {bar}")

    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if intra and inter:
        intra_mean = np.mean(intra)
        inter_mean = np.mean(inter)
        gap = intra_mean - inter_mean
        print(f"  Intra-ID mean : {intra_mean:.3f}  (want > 0.80)")
        print(f"  Inter-ID mean : {inter_mean:.3f}  (want < 0.50)")
        print(f"  Discriminative gap : {gap:.3f}  (want > 0.30)")
        if gap >= 0.30 and inter_mean < 0.60:
            print("  ✓ Embeddings are discriminative — model looks good.")
            print("  → For cross-camera matching, threshold 0.65–0.75 is a good starting point.")
        elif gap >= 0.15:
            print("  ~ Moderate discrimination. Matches may work but expect some false positives.")
            print("  → Lower threshold to 0.60–0.65 and check precision.")
        else:
            print("  ✗ Poor discrimination — intra/inter gap too small.")
            print("  → The model collapses identities into a narrow region.")
            print("  → Consider a different checkpoint or verify SGIE crops are correct size.")
    elif inter:
        print(f"  Only inter-ID pairs available. Mean={np.mean(inter):.3f}")
        print("  Cannot assess intra-ID stability without multiple embeddings per TID.")
    else:
        print("  Only intra-ID pairs found — need more unique tracking IDs for discrimination test.")


if __name__ == "__main__":
    main()
