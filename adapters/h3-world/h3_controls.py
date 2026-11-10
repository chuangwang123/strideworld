"""Deterministic StrideWorld-to-H3 action compiler; no inference dependencies.

native_keys preserves the released character/follow-camera vocabulary.
camera_text is explicitly an experimental language adapter, not native keys.
Neither profile claims metric translation or angle control.
"""
from __future__ import annotations

import hashlib
import json

H3_COMMIT = "e46d64f1e62a1514ccdb797d4014ed04ab9735cc"
DIFFSYNTH_COMMIT = "300e3e4da76e881d5e6bd97d897810c18f6e4893"
KEYS = ("W", "A", "S", "D", "I", "J", "K", "L", "F")
ACTION_KEYS = {"NOOP": (), "MOVE_FORWARD": ("W",), "MOVE_BACKWARD": ("S",),
               "MOVE_LEFT": ("A",), "MOVE_RIGHT": ("D",),
               "YAW_LEFT": ("J",), "YAW_RIGHT": ("L",)}
NATIVE_TEXT = {
    "NOOP": "the man stands still, camera holds steady",
    "MOVE_FORWARD": "the man walks forward, camera follows him",
    "MOVE_BACKWARD": "the man walks backward, camera follows him",
    "MOVE_LEFT": "the man strafes left, camera follows him",
    "MOVE_RIGHT": "the man strafes right, camera follows him",
    "YAW_LEFT": "the man stands still, camera pans left slowly",
    "YAW_RIGHT": "the man stands still, camera pans right slowly",
}
CAMERA_TEXT = {
    "NOOP": "the environment is stationary, camera holds steady",
    "MOVE_FORWARD": "the environment is stationary, camera moves forward",
    "MOVE_BACKWARD": "the environment is stationary, camera moves backward",
    "MOVE_LEFT": "the environment is stationary, camera moves left",
    "MOVE_RIGHT": "the environment is stationary, camera moves right",
    "YAW_LEFT": "the environment is stationary, camera pans left slowly",
    "YAW_RIGHT": "the environment is stationary, camera pans right slowly",
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def packed_length_bound(num_units):
    """Proved finite-vocabulary bound; verified by padding_plan.py on CPU.

    413 head tokens (one 390-token image), two 8-token NOOP slots,
    at most 10 tokens/action latent, native video/audio row counts.
    Padding to 64 avoids a nonempty short final attention segment.
    """
    if type(num_units) is not int or not 1<=num_units<=16:
        raise ValueError('invalid packing horizon')
    audio_t=round((17*num_units+5)/24.0*40.0)
    used=413+16+50*num_units+390+2*audio_t+390*(5*num_units+2)
    return ((used+63)//64)*64


def profile(mode):
    if mode not in ("native_keys", "camera_text", "camera_text_half"):
        raise ValueError("unknown adapter mode")
    p = {"schema": "strideworld-h3-profile-v1", "mode": mode,
         "profile_id": "h3_world_" + mode + "_unit17f_slow_v1",
         "h3_commit": H3_COMMIT, "diffsynth_commit": DIFFSYNTH_COMMIT,
         "action_text": NATIVE_TEXT if mode == "native_keys" else CAMERA_TEXT,
         "action_keys": {k:list(v) for k,v in ACTION_KEYS.items()} if mode == "native_keys" else None,
         "supported_actions": list(ACTION_KEYS), "fps": 24,
         "frames_per_unit": 17, "latent_tokens_per_unit": 5,
         "condition_frames": 1, "terminal_noop_frames": 4,
         "height": 480, "width": 832, "steps": 50, "cfg_scale": 1.0,
         "translation_distance_per_unit": None, "yaw_degrees_per_unit": None,
         "amplitude_contract": "fixed native exposure time; actual motion measured from RGB",
         "native_translation_is_character_follow": mode == "native_keys",
         "custom_language_adapter": mode == "camera_text",
         "packing": {"strategy":"fixed 64-aligned upper bound per horizon",
                     "tokenizer_sha256":"a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
                     "head_token_upper_bound":413,"action_sentence_token_upper_bound":10,
                     "noop_sentence_tokens":8,
                     "action_pad_used_to_by_units":{str(n):packed_length_bound(n) for n in range(1,17)}},
         "acceptance": "candidate_requires_gpu_smoke", "official_reproduction": False}
    if mode == "camera_text_half":
        p.update(profile_id="h3_world_camera_text_half_unit8p5f_slow_v2",
                 frames_per_unit=8.5, latent_tokens_per_unit=2.5,
                 frames_per_unit_pattern=[8,9], latent_tokens_per_unit_pattern=[2,3],
                 terminal_noop_frames=None,
                 terminal_noop_policy="4 frames for even units; 13 for odd units to close native horizon",
                 custom_language_adapter=True,
                 amplitude_contract="half action exposure for even unit counts; actual RGB motion is not guaranteed half")
        p['packing']['action_pad_used_to_by_units']={str(n):packed_length_bound((n+1)//2) for n in range(1,17)}
    p["profile_sha256"] = canonical_hash(p)
    return p


def frame_spans(latent_t):
    result, start = [], 0
    for i in range(latent_t):
        end = start + (1, 4, 4, 4, 4)[i % 5]
        result.append([start, end])
        start = end
    return result


def compile_controls(actions, mode="native_keys"):
    if not isinstance(actions, (list, tuple)) or not 1 <= len(actions) <= 16:
        raise ValueError("expected 1..16 semantic action units")
    if any(a not in ACTION_KEYS for a in actions):
        raise ValueError("unsupported action: pitch is not vertical translation")
    p = profile(mode)
    n = len(actions)
    divisor = 2 if mode == "camera_text_half" else 1
    physical_units = (n+divisor-1)//divisor
    labels = ["NOOP"] + [a for i,a in enumerate(actions)
                            for _ in range(5*(i+1)//divisor-5*i//divisor)]
    labels += ["NOOP"]*(5*physical_units+2-len(labels))
    spans = frame_spans(len(labels))
    num_frames = 17*physical_units+5
    final_action_frame = 17*n//divisor
    assert spans[-1][1] == num_frames
    timeline = []
    for i, action in enumerate(actions):
        timeline.append({"unit": i+1, "action": action, "latent_start": 1+5*i//divisor,
                         "latent_end_exclusive": 1+5*(i+1)//divisor, "frame_start": 1+17*i//divisor,
                         "frame_end_exclusive": 1+17*(i+1)//divisor, "checkpoint_frame": 17*(i+1)//divisor})
    c = {"profile_id": p["profile_id"], "profile_sha256": p["profile_sha256"],
         "mode": mode, "semantic_blocks": list(actions), "num_frames": num_frames,
         "latent_t": len(labels), "fps": 24, "action_script": [p["action_text"][a] for a in labels],
         "action_pad_used_to": packed_length_bound(physical_units),
         "negative_action_script": [p["action_text"]["NOOP"]]*len(labels),
         "keys9": [[int(k in ACTION_KEYS[a]) for k in KEYS] for a in labels] if mode == "native_keys" else None,
         "latent_frame_spans": spans, "unit_timeline": timeline,
         "condition_frame_index": 0, "final_action_frame": final_action_frame,
         "terminal_noop_span": [final_action_frame+1, num_frames],
         "dose_checkpoint_frames": {str(d): 17*d//divisor for d in (1,2,3,5,8) if d <= n},
         "postprocessing": "save every native output frame; no prepend, duplicate, or interpolation"}
    c["controls_sha256"] = canonical_hash(c)
    return c
