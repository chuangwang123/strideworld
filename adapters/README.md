# Adapters

One adapter per model turns a StrideWorld request (a start view, an ordered list of action units, a seed) into one
call of the model's released inference code. Every model keeps its own control interface, resolution and frame rate.

`model_settings.json` is the reference for what was run for the paper. For each of the 15 models it lists the
upstream code and its commit, the weights and their revision, frames per action unit, frame rate, resolution, the
command of one action unit, the sampler, and how the start image is brought to the model's resolution. It matches
the table of model control interfaces in the appendix of the paper (Table A.2).

## Shared rules

* The model sees the start image and its fixed prompt. It never sees a later frame.
* Action units are applied in order, each in the current camera frame (x right, y down, z forward).
* The seed of the request is set again for every video.
* A single-action video and a route video use the same settings. A route is one rollout, never a concatenation of
  clips.
* Frames after the last action unit are not scored. Some models generate a few more frames to fill their native
  block.

## Layout

| Folder | Model | What the files do |
|---|---|---|
| `common/` | all camera-pose models | `strideworld_actions.py` holds the six actions and the camera path they command (`sample_c2w`, `pixel_intrinsics`). `build_model_inputs.py` builds the inputs of Matrix-Game 3.5, SolarWM and EVOKE from the requests, and `model_profiles.json` holds their command per action unit. |
| `matrix-game-3/`, `matrix-game-2/` | Matrix-Game 3.0, 2.0 | `matrix_game_*.json` maps each action to the keyboard vector and the mouse delta. `action_interface/` compiles a request into per-frame controls. `run_matrix_game_*_sequential.py` calls the official inference once per video. |
| `sana-wm/` | SANA-WM | `sana_wm.json` and `sana_wm_native.py` build the native action sequence. `run_sana_wm_streaming_persistent_worker.py` runs the official streaming pipeline. |
| `dreamx-world/` | DreamX-World | `run_dreamx_world_d16.py` (single actions) and `run_dreamx_world_routes.py` (routes) build the action tokens and call the official inference. The patch resets the random state before every video, so a video does not depend on its place in a batch. |
| `hunyuan-world-1.5/` | HY-World 1.5 | The worker turns a request into the official pose string and makes one pipeline call per video. |
| `hy-world-2/` | HY-World 2.0 | `hyworld2_camera_controls.py` builds the camera path, `hyworld2_rgbd_render.py` renders the condition frames from the depth of the start view, `hyworld2_worker.py` calls the official model. `upstream_local_changes.patch` lists the local changes to the official code (offline loading). |
| `lyra2/` | Lyra 2.0 | `prepare_lyra2_adapter.py` and `build_camera_profile.py` build the per-frame camera path, 10 frames per action unit. `run_lyra2.py` and `lyra2_persistent_batch.py` run the official sampler. |
| `solarwm-h3/` | SolarWM | Inputs come from `common/build_model_inputs.py`. `CONTROL_PROFILE.json` holds the command and `infer_config.json` the configuration passed to the official `solarwm infer`. |
| `worldcrafter-fast/` | WorldCrafter | `worldcrafter_schedule.py` builds the camera array and the command line of the official `inference.py`. |
| `lingbot-world-v2/` | LingBot-World v2 | The worker turns a request into the official camera and intrinsics files and makes one generation call per video. |
| `hy-worldplay/` | HY-WorldPlay | The worker turns a request into the official pose string. `OFFICIAL_CONTROL_CHECK.json` records the translation and yaw of one official tick. |
| `evoke-turbo/` | EVOKE | `build_evoke_full_inputs.py` builds the camera file, `evoke_full.py` runs the official inference, and `argv_sample.jsonl` shows the full command line of one video. |
| `matrix-game-3.5/` | Matrix-Game 3.5 | Inputs come from `common/build_model_inputs.py`. `matrix_persistent_worker.py` and `run_matrix_2gpu.py` run the official distilled inference with the two configuration files. |
| `alaya-world/` | AlayaWorld | `common.py` holds the command and builds the camera sequence in the official case layout. `run_full_worker.py` calls the official engine. |
| `h3-world/` | H3-World | `h3_controls.py` holds the fixed sentence of every action. `worker.py` generates each video in one call. |

## Running an adapter

The files are the drivers used for the paper with paths replaced by placeholders (`<path>/...`). To run one, check
out the upstream code at the commit in `model_settings.json`, download the weights at the listed revision, and point
the path constants at the top of the driver to them. The drivers check the hashes of the code and weights they were
run with, so a different revision stops with an error before generation. Scripts that only submit jobs to our
scheduler, resume interrupted jobs or audit finished outputs are not part of this release.

The requests are in `manifests/` of this repository and the start views in the data release. The recovered camera
trajectories and scores of all generated videos are in the data release, so every table of the paper can be rebuilt
without generating a video (see `scoring/`).
