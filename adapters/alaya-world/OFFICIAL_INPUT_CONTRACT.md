# AlayaWorld official input contract

This adapter is pinned to AlayaWorld commit `ea03cfbb2e4c4e9102ed8ea8562e0b5370ca9b79`.
The upstream inference entry consumes one case triplet:

```
<prefix>_image.png
<prefix>_camera.pt
<prefix>_prompt.txt
```

The camera metadata must contain `cam_c2w` with shape `[F,4,4]` and `intrinsic` with shape `[3,3]`. The official action conversion samples `cam_c2w` every 8 pixel frames and computes consecutive relative SE(3) poses, dividing the six-vector by the config action scale:

```
0.1337, 0.2769, 2.021, 0.00442, 0.01026, 0.00448
```

The official bundled `playground/case1/case1_camera.pt` has a forward step of `0.635` at the 8-frame latent stride. The smoke adapter therefore uses the official native8 baseline:

```
forward/backward:  ±0.635 per unit
left/right:        ±0.0423 per unit
yaw:               ±1 degree per unit
```

This yields approximately `0.314` forward, `0.316` lateral, and `1.695` yaw in the official normalized action space. The earlier v6 2x profile is intentionally excluded because it is outside this official-strength contract.

Each atomic route contains 16 units (full dose 16). Dose 12 is only the first 12 units / first 97 decoded frames for inspection. The combination route is the canonical 14-unit `G4_003_A` action list. The generated metadata contains 128 history frames, the control trajectory, and a held tail so the official 4-chunk layout decodes to 129 frames at 24 fps.

The CPU audit checks the exact upstream `c2w_to_adaln_action_vectors` implementation against every saved camera file. GPU smoke is still required for semantic motion and direction; nonzero image flow alone is not treated as action correctness.
