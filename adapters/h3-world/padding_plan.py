#!/usr/bin/env python3
"""Derive exact finite-vocabulary H3 packed-length bounds without model weights.

The optional official check imports the pinned CPU processor/packed builder;
it never constructs a model, loads a checkpoint, or invokes CUDA inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

from build_campaign import load_parent, SCENE_PROMPT, PARENT_SHA
from h3_controls import ACTION_KEYS, H3_COMMIT, DIFFSYNTH_COMMIT, compile_controls, profile

BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_metadata(processor_root, selection_path):
    selection = json.loads(Path(selection_path).read_text())
    if selection["repo_id"] != "MiniMaxAI/MiniMax-H3" or selection["revision"] != BASE_REVISION:
        raise ValueError("wrong official metadata inventory")
    verified = {}
    for name in ("tokenizer.json", "preprocessor_config.json"):
        rel = "FL2VA/processor/" + name
        expected = next(r for r in selection["files"] if r["rfilename"] == rel)
        body = (Path(processor_root) / name).read_bytes()
        if len(body) != expected["size"]:
            raise ValueError("processor metadata size changed: " + name)
        digest = hashlib.sha256(body).hexdigest()
        official_lfs = expected.get("lfs", {}).get("sha256")
        blob_id = hashlib.sha1(("blob " + str(len(body)) + "\0").encode() + body).hexdigest()
        if official_lfs and digest != official_lfs:
            raise ValueError("processor metadata LFS SHA changed: " + name)
        if not official_lfs and blob_id != expected["blobId"]:
            raise ValueError("processor metadata git blob changed: " + name)
        verified[name] = {"sha256": digest, "official_git_blob_id": expected["blobId"], "bytes": len(body)}
    return verified


def build_plan(processor_root, selection_path, parent_manifest):
    from tokenizers import Tokenizer

    metadata = verify_metadata(processor_root, selection_path)
    config = json.loads((Path(processor_root) / "preprocessor_config.json").read_text())
    if config["image_processor_type"] != "Qwen2VLImageProcessorFast" or config["processor_class"] != "Qwen3VLProcessor":
        raise ValueError("unexpected processor implementation")
    width, height = 832, 480
    patch, merge, temporal = config["patch_size"], config["merge_size"], config["temporal_patch_size"]
    factor = patch * merge
    # Qwen smart_resize returns the same dimensions in this exact fixed-size
    # branch: both sides already factor-aligned and area within min/max.
    if width % factor or height % factor or not config["size"]["shortest_edge"] <= width * height <= config["size"]["longest_edge"]:
        raise ValueError("fixed-size smart_resize branch no longer applies; obtain official CPU image grid first")
    if (patch, merge, temporal) != (16, 2, 2):
        raise ValueError("unexpected image patch geometry")
    grid = [1, height // patch, width // patch]
    image_tokens = grid[0] * grid[1] * grid[2] // (merge ** 2)
    tokenizer = Tokenizer.from_file(str(Path(processor_root) / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    image_prefix_ids = encode("<Picture 1>: ")
    prompt_ids = encode(SCENE_PROMPT)
    negative_prompt_ids = encode(" ")
    head = len(image_prefix_ids) + 2 + image_tokens + len(prompt_ids)
    negative_head = len(image_prefix_ids) + 2 + image_tokens + len(negative_prompt_ids)
    tables = {mode: {action: {"text": text, "tokens": len(encode(text)), "ids": encode(text)}
                     for action, text in profile(mode)["action_text"].items()}
              for mode in ("native_keys", "camera_text")}
    common_max_tokens = max(v["tokens"] for t in tables.values() for v in t.values())
    common_noop_tokens = max(t["NOOP"]["tokens"] for t in tables.values())
    min_sentence_tokens = min(v["tokens"] for t in tables.values() for v in t.values())
    frame_rows = (height // 16 // 2) * (width // 16 // 2)
    all_horizons = {}
    for n in range(1, 15):
        frames, latent_t = 17 * n + 5, 5 * n + 2
        audio_t = round(frames / 24.0 * 40.0)  # exact native Python round operation
        annotation_upper = 2 * common_noop_tokens + 5 * n * common_max_tokens
        text_upper = max(head, negative_head) + annotation_upper
        real_upper = text_upper + frame_rows + 2 * audio_t + latent_t * frame_rows
        pad = ((real_upper + 63) // 64) * 64
        # Last action-script token starts at source-frame 17*N+1, scaled by
        # 5/3 for RoPE. The annotation origin must remain after the head.
        rope_min = Fraction((17 * n + 1) * 5, 3) + 1
        rope_min_integer = (rope_min.numerator + rope_min.denominator - 1) // rope_min.denominator
        minimum_annotation_tokens = latent_t * min_sentence_tokens
        if minimum_annotation_tokens < rope_min_integer:
            raise ValueError(f"action sentences too short for H3 RoPE at N={n}; padding cannot repair this")
        all_horizons[str(n)] = {
            "num_units": n, "num_frames": frames, "latent_t": latent_t,
            "audio_latent_t": audio_t, "frame_rows": frame_rows,
            "condition_rows": frame_rows, "audio_rows": 2 * audio_t,
            "video_rows": latent_t * frame_rows,
            "annotation_token_upper_bound": annotation_upper,
            "text_token_upper_bound": text_upper,
            "real_used_upper_bound": real_upper, "action_pad_used_to": pad,
            "seq_len": pad, "cu_seqlens": [0, pad, pad],
            "padding_alignment_rows": pad - real_upper,
            "rope_annotation_minimum_exact": str(rope_min),
            "rope_annotation_minimum_integer": rope_min_integer,
            "annotation_token_lower_bound": minimum_annotation_tokens,
            "rope_minimum_margin_tokens": minimum_annotation_tokens - rope_min_integer,
        }
    parent = load_parent(parent_manifest)
    formal = sorted({len(row["semantic_blocks"]) for row in parent})
    if formal != [5, 8, 11, 14]:
        raise ValueError("parent horizons changed")
    stats = {}
    for mode, table in tables.items():
        by_n = {n: [] for n in formal}
        for row in parent:
            n = len(row["semantic_blocks"])
            controls = compile_controls(row["semantic_blocks"], mode)
            length_map = {v["text"]: v["tokens"] for v in table.values()}
            action_tokens = sum(length_map[text] for text in controls["action_script"])
            h = all_horizons[str(n)]
            if controls["action_pad_used_to"] != h["action_pad_used_to"]:
                raise ValueError("compiled action padding differs from verified finite-vocabulary bound")
            if profile(mode)["packing"]["action_pad_used_to_by_units"][str(n)] != h["action_pad_used_to"]:
                raise ValueError("profile padding table differs from metadata proof")
            used = head + action_tokens + h["condition_rows"] + h["audio_rows"] + h["video_rows"]
            if used > h["action_pad_used_to"]:
                raise ValueError("actual manifest exceeds derived bound")
            by_n[n].append(used)
        stats[mode] = {str(n): {"records": len(values), "actual_positive_real_used_min": min(values),
                               "actual_positive_real_used_max": max(values),
                               "fixed_pad": all_horizons[str(n)]["action_pad_used_to"]}
                       for n, values in by_n.items()}
    global_pad = max(all_horizons[str(n)]["action_pad_used_to"] for n in formal)
    result = {"schema": "h3-padding-plan-v1", "metadata": metadata,
              "base_revision": BASE_REVISION, "parent_manifest_sha256": PARENT_SHA,
              "h3_commit": H3_COMMIT, "diffsynth_commit": DIFFSYNTH_COMMIT,
              "profile_sha256": {mode: profile(mode)["profile_sha256"] for mode in tables},
              "width": width, "height": height, "image_grid_thw": grid,
              "image_tokens": image_tokens, "image_prefix_tokens": len(image_prefix_ids),
              "vision_boundary_tokens": 2, "prompt_tokens": len(prompt_ids),
              "negative_prompt_tokens": len(negative_prompt_ids),
              "head_tokens": head, "negative_head_tokens": negative_head,
              "scene_prompt": SCENE_PROMPT, "action_tokens": tables,
              "common_action_token_upper_bound": common_max_tokens,
              "common_noop_tokens_upper_bound": common_noop_tokens,
              "bound_type": "mathematical upper bound over fixed finite vocabulary and all 1..14-unit action sequences, including NOOP; not guessed shortest actual length",
              "all_horizons": all_horizons, "formal_horizons": formal,
              "recommended_strategy": "common per-horizon 64-aligned bound shared by both profiles; run one horizon per worker process for strongest compile isolation",
              "formal_action_pad_used_to": {str(n): all_horizons[str(n)]["action_pad_used_to"] for n in formal},
              "main_flex_shapes_per_profile_if_mixed_horizons": len(formal),
              "tail_padding_segment_length": 0,
              "global_alternative_action_pad_used_to": global_pad,
              "global_vs_horizon_row_count_ratio": {str(n): global_pad / all_horizons[str(n)]["action_pad_used_to"] for n in formal},
              "overhead_interpretation": "Ratios describe QKV/FFN/activation row counts, not measured latency or an assumed quadratic attention cost. Padded rows are masked but not free.",
              "actual_manifest_lengths": stats,
              "official_cpu_builder_verified": False, "gpu_compile_verified": False}
    return result


def verify_official_cpu(plan, source, processor_root):
    source = Path(source).resolve()
    diff = source / "DiffSynth-Studio-h3-v2"
    for repo, expected in ((source, H3_COMMIT), (diff, DIFFSYNTH_COMMIT)):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if actual != expected:
            raise ValueError("official source commit differs")
    sys.path.insert(0, str(diff))
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    import diffsynth.models.minimax_h3_text_encoder as te
    import diffsynth.pipelines.minimax_h3_audio_video as pipe_module
    if not Path(te.__file__).resolve().is_relative_to(diff) or not Path(pipe_module.__file__).resolve().is_relative_to(diff):
        raise ValueError("wrong diffsynth import root")
    if "action_pad_used_to" not in inspect.getsource(pipe_module) or "action_real_used" not in inspect.getsource(pipe_module):
        raise ValueError("official action padding patch is absent")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    processor = AutoProcessor.from_pretrained(str(processor_root), local_files_only=True)
    _, grid, counts = te.image_token_counts(processor, [Image.new("RGB", (832, 480))])
    if counts != [plan["image_tokens"]] or grid.tolist() != [plan["image_grid_thw"]]:
        raise ValueError("official processor grid differs from metadata derivation")
    heads = {}
    for label, prompt in (("positive", SCENE_PROMPT), ("negative", " ")):
        ids, _ = te.presentation_fl2va(processor.tokenizer, prompt, counts)
        heads[label] = int(ids.numel())
    if heads != {"positive": plan["head_tokens"], "negative": plan["negative_head_tokens"]}:
        raise ValueError("official text head length differs")
    for table in plan["action_tokens"].values():
        for entry in table.values():
            ids, _ = te.presentation_t2va(processor.tokenizer, entry["text"])
            if ids.tolist() != entry["ids"]:
                raise ValueError("official tokenizer wrapper differs from tokenizer.json probe")
    builder = pipe_module.MiniMaxH3Unit_PackedSequenceBuilder()
    checked = 0
    for mode, table in plan["action_tokens"].items():
        length_map = {entry["text"]: entry["tokens"] for entry in table.values()}
        for n in range(1, 15):
            horizon = plan["all_horizons"][str(n)]
            # Cover every direction as the longest repeated run, plus NOOP;
            # mixed sequences lie under the proved per-token bound.
            for action in ACTION_KEYS:
                controls = compile_controls([action] * n, mode)
                if controls["action_pad_used_to"] != horizon["action_pad_used_to"]:
                    raise ValueError("compiled per-horizon padding differs from CPU plan")
                for branch, script in (("positive", controls["action_script"]), ("negative", controls["negative_action_script"])):
                    cursor, spans = heads[branch], []
                    for text in script:
                        end = cursor + length_map[text]
                        spans.append((cursor, end))
                        cursor = end
                    packed = builder._build_packed_fl2va(cursor, horizon["latent_t"], 30, 52,
                                horizon["audio_latent_t"], [0], action_text_spans=spans,
                                pad_used_to=horizon["action_pad_used_to"])
                    if packed["seq_len"] != horizon["seq_len"] or packed["cu_seqlens"].tolist() != horizon["cu_seqlens"]:
                        raise ValueError("official builder produces extra attention shapes")
                    if packed["action_real_used"] > horizon["action_pad_used_to"]:
                        raise ValueError("official packed length exceeds bound")
                    if packed["img_position_ids"].device.type != "cpu":
                        raise ValueError("CPU probe unexpectedly used an accelerator")
                    checked += 1
    plan["official_cpu_builder_verified"] = True
    plan["official_cpu_builder_checks"] = checked
    plan["official_cpu_probe_scope"] = "2 profiles x 14 horizons x 7 actions x positive/negative = 392 CPU-only packed layouts; no model instantiation"
    plan["official_image_grid_thw"] = grid.tolist()
    plan["official_cpu_versions"] = {"torch": torch.__version__}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processor-root", required=True, type=Path)
    ap.add_argument("--base-selection", required=True, type=Path)
    ap.add_argument("--parent-manifest", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--verify-official", action="store_true")
    ap.add_argument("--source", type=Path)
    args = ap.parse_args()
    plan = build_plan(args.processor_root, args.base_selection, args.parent_manifest)
    if args.verify_official:
        if args.source is None:
            ap.error("--verify-official requires --source")
        verify_official_cpu(plan, args.source, args.processor_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: plan[k] for k in ("head_tokens", "negative_head_tokens", "common_action_token_upper_bound", "formal_action_pad_used_to", "official_cpu_builder_verified", "gpu_compile_verified")}, indent=2))


if __name__ == "__main__":
    main()
