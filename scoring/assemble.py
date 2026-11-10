"""Assemble the per-model measurement files of the data release into the layout the scoring scripts read.

usage: python3 scoring/assemble.py DATA_RELEASE_DIR WORK_DIR

DATA_RELEASE_DIR/measurements/<model>/   per-model files (see the data card)
WORK_DIR/base/                           the 13 benchmark models (one table per measurement, models concatenated)
WORK_DIR/extra/                          the models scored later (SolarWM, HY-WorldPlay), *_solar names
WORK_DIR/<model>/                        one directory per later model (registry_release.json)
WORK_DIR/sft/                            the fine-tuning study of HY-World 1.5 (Sec. 6)
WORK_DIR/deform/                         Rigidity per-frame residuals of all models (perframe_all*.csv.gz)
Row order inside every file is that of the release (models in the order below).
"""
import gzip, json, shutil, sys
from pathlib import Path

BASE = ['matrix-game-3.5', 'sana-wm', 'dreamx-world', 'hunyuan-world-1.5', 'hy-world-2', 'lingbot-world-v2', 'alaya-world',
        'h3-world', 'lyra2', 'evoke-turbo', 'worldcrafter-fast', 'matrix-game-2', 'matrix-game-3']
BASE_SCORES = ['sana-wm', 'dreamx-world', 'hunyuan-world-1.5', 'hy-world-2', 'lingbot-world-v2', 'alaya-world', 'h3-world',
               'lyra2', 'matrix-game-3.5', 'evoke-turbo', 'worldcrafter-fast', 'matrix-game-2', 'matrix-game-3']
EXTRA = ['hy-worldplay', 'solarwm-h3']
SFT = ['n0', 'j256', 'b2', 'hy15ref']
TABLES = ['main_per_video.csv', 'unit_estimates.csv', 'initial_view_depths.csv']


def cat_csv(srcs, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as out:
        for i, s in enumerate(srcs):
            lines = (gzip.open(s, 'rt', encoding='utf-8') if s.suffix == '.gz' else open(s, encoding='utf-8')).readlines()
            out.writelines(lines if i == 0 else lines[1:])


def cat_gz(srcs, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dst, 'wt', encoding='utf-8') as out:
        for i, s in enumerate(srcs):
            lines = gzip.open(s, 'rt', encoding='utf-8').readlines()
            out.writelines(lines if i == 0 else lines[1:])


def scores(srcs, dst):
    recs = []
    for s in srcs:
        recs += json.load(gzip.open(s, 'rt', encoding='utf-8'))
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(dst, 'w'))


def obs(src_dir, model, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(src_dir / 'observations.csv.gz', 'rt', encoding='utf-8') as f, \
            open(dst_dir / f'{model}.observations.csv', 'w', encoding='utf-8') as g:
        shutil.copyfileobj(f, g)


def main(rel, work):
    M = rel / 'measurements'
    work.mkdir(parents=True, exist_ok=True)
    b = work / 'base'
    for t in TABLES + ['per_checkpoint_scores.csv']:
        cat_csv([M / m / t for m in BASE], b / t)
    scores([M / m / 'scores.json.gz' for m in BASE_SCORES], b / 'all_scores.json')
    for m in BASE:
        obs(M / m, m, b / 'obs')
    x = work / 'extra'
    for t in TABLES:
        cat_csv([M / m / t for m in EXTRA], x / t.replace('.csv', '_solar.csv'))
    scores([M / m / 'scores.json.gz' for m in EXTRA], x / 'all_scores_solar.json')
    for m in EXTRA:
        obs(M / m, m, x / 'obs')
        d = work / m
        for t in TABLES:
            cat_csv([M / m / t], d / t)
        scores([M / m / 'scores.json.gz'], d / 'all_scores.json')
        obs(M / m, m, d / 'obs')
    s = work / 'sft'
    for t in TABLES:
        cat_csv([M / 'finetune' / m / t for m in SFT], s / t.replace('.csv', '_sft.csv'))
    scores([M / 'finetune' / m / 'scores.json.gz' for m in SFT], s / 'all_scores_sft.json')
    for m in SFT:
        obs(M / 'finetune' / m, m, s / 'obs')
    f1 = [m for m in BASE + EXTRA if (M / m / 'rigidity_perframe.csv.gz').exists()]
    cat_gz([M / m / 'rigidity_perframe.csv.gz' for m in f1], work / 'deform' / 'perframe_all.csv.gz')
    cat_gz([M / m / 'rigidity_videos.csv.gz' for m in f1], work / 'deform' / 'perframe_all_videos.csv.gz')
    print(f'assembled {len(BASE)} + {len(EXTRA)} models, {len(SFT)} fine-tuning runs, Rigidity frames of {len(f1)} models -> {work}')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
