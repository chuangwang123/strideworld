# Leaderboard

`index.html` is one static page with the leaderboard data and example descriptions embedded (`leaderboard.json` and `examples.json`). It uses system fonts and makes no external requests when opened.

```bash
python3 leaderboard/build_leaderboard.py --table <data>/results/table2.csv --f2 <data>/results/f2_ratio.csv \
    --out leaderboard/leaderboard.json
python3 leaderboard/build_page.py        # embeds both JSON files into page_template.html -> index.html
```

## Examples

The Examples section contains four tabs: Rigidity, Pace, Composition and Fine-tuning. Each tab pairs two annotated
short clips from `media/` (eight MP4 files and eight JPG posters in total), with labels, captions and alternative text
from `examples.json`. The clips retain their original encoding and annotations.

Only the selected pair loads and plays. The pair loops together, pauses offscreen, and respects the system
reduced-motion setting; native controls allow manual playback. Use the left/right arrow keys to switch tabs.
The videos stack on narrow screens. Keep `media/` beside `index.html` when serving or opening the page locally.
If the media is absent, Examples hides automatically; the leaderboard remains usable. No external assets are fetched.

## Submitting a model

To submit a model, open an issue in this repository or contact the corresponding authors at
`qianyu@buaa.edu.cn` or `liweijia@sz.tsinghua.edu.cn`. Send the per-video measurements of all 3,640 requests of
`manifests/requests.jsonl` in the release format (one directory per model: `scores.json.gz`, `main_per_video.csv`,
`unit_estimates.csv`, `initial_view_depths.csv`, `observations.csv.gz`, optionally `rigidity_perframe.csv.gz` /
`rigidity_videos.csv.gz`; see the data card), produced by `pipeline/` from your videos, together with: the model name
and version, its interface (discrete keys, camera path, text prompt), the command per step (translation and yaw per
action unit) and whether it is the release default, and the frame indices of the step boundaries. We score it with
`scoring/` and add the row. The leaderboard is a static release snapshot of Table 2.

Terminology: the paper’s rigid counterpart is the code’s rigid twin, and the paper’s Pace score is the code’s Hold score.
