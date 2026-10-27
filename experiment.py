"""Repeatable experiments: the same settings across several random seeds.

One lucky run proves little. This trains one independent run per seed (in
parallel), then tests every seed's champion on every track, including tracks
it never saw in training.

    python experiment.py --track flowing --seeds 1 2 3 4 5 --generations 30
    python experiment.py --track easy --seeds 1 2 3 --legacy-fitness

Results go to experiments/<timestamp>-<track>/:
    seed-<n>/          a normal training run folder per seed
    summary.md         tables you can paste into a README
    summary.json       the same numbers, machine-readable
    comparison.png     best fitness per seed + mean lap-completion rate
"""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import statistics

import pygame

from evolution import POPULATION_SIZE, write_json
from track import TRACK_NAMES, add_track_argument, make_track


def run_seed(job):
    """Worker: train one seed headlessly, then test its champion on every track."""
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    pygame.init()
    try:
        from training import TrainingSession
        from evaluate_tracks import evaluate_champion
        folder = Path(job['folder'])
        folder.mkdir(parents=True, exist_ok=True)
        session = TrainingSession(make_track(job['track']), folder, seed=job['seed'],
                                  generations=job['generations'],
                                  population_size=job['population'],
                                  progress='straight' if job['legacy'] else 'track',
                                  lap_bonus=not job['legacy'])
        while True:
            session.run_generation()
            if session.finished:
                break
            session.next_generation()
        _, tests = evaluate_champion(folder / 'champion.json')
        return dict(seed=job['seed'], folder=str(folder), history=session.history, tests=tests)
    finally:
        pygame.quit()


def summarise(result):
    history = result['history']
    first_lap = next((h['generation'] for h in history if h['completion_rate'] > 0), None)
    final = history[-1]
    laps = [h['generation_best_lap'] for h in history if h.get('generation_best_lap')]
    return dict(seed=result['seed'], first_lap_generation=first_lap,
                final_best=round(final['best'], 1), final_average=round(final['average'], 1),
                final_completion=round(100 * final['completion_rate'], 1),
                best_lap=round(min(laps), 2) if laps else None)


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return '-'
    if len(values) == 1:
        return f'{values[0]:.1f}'
    return f'{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}'


def write_report(folder, settings, results):
    rows = [summarise(r) for r in results]
    lapped = [r for r in rows if r['first_lap_generation']]
    lines = [f"# Experiment: {settings['track']} track, {len(rows)} seeds", '',
             f"Population {settings['population']}, {settings['generations']} generations, "
             f"fitness: {'v1.0 legacy' if settings['legacy'] else 'progress along track + lap-time bonus'}.",
             '', '## Training', '',
             '| Seed | First lap (generation) | Final best | Final average | Final lap completion | Best lap |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lap = f"{r['best_lap']:.2f}s" if r['best_lap'] else '-'
        lines.append(f"| {r['seed']} | {r['first_lap_generation'] or 'never'} | {r['final_best']} | "
                     f"{r['final_average']} | {r['final_completion']}% | {lap} |")
    lines += ['', f"**{len(lapped)} of {len(rows)} seeds** produced a driver that completed a lap. "
              f"First lap: {mean_sd([r['first_lap_generation'] for r in rows])} generations "
              f"(seeds that lapped). Final lap completion: {mean_sd([r['final_completion'] for r in rows])}%. "
              f"Best lap: {mean_sd([r['best_lap'] for r in rows])} s.", '',
              '## Champions on every track', '',
              'Each seed\'s champion driven once on each track. Unseen = never used in training.', '',
              '| Track | Seen in training? | Laps completed | Mean lap % reached |',
              '| --- | --- | --- | --- |']
    for name in TRACK_NAMES:
        tests = [t for r in results for t in r['tests'] if t['track'] == name]
        done = sum(t['status'] == 'completed lap' for t in tests)
        lines.append(f"| {name} | {'yes' if name == settings['track'] else 'no (unseen)'} | "
                     f"{done}/{len(tests)} | {statistics.mean(t['lap_percent'] for t in tests):.1f}% |")
    (folder / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(folder / 'summary.json', dict(settings=settings, seeds=rows,
                                             tests={r['seed']: r['tests'] for r in results}))
    return '\n'.join(lines)


def save_comparison(folder, settings, results):
    from progress_graph import draw_progress_graph
    palette = [(255, 200, 80), (110, 190, 255), (255, 110, 130), (190, 140, 255),
               (240, 240, 240), (90, 220, 210), (255, 150, 60), (170, 210, 90)]
    count = min(len(r['history']) for r in results)
    series = [(f"seed {r['seed']}", palette[i % len(palette)],
               [h['best'] for h in r['history'][:count]], 'fitness')
              for i, r in enumerate(results)]
    series.append(('mean lap completion %', (120, 220, 140),
                   [100 * statistics.mean(r['history'][g]['completion_rate'] for r in results)
                    for g in range(count)], 'percent'))
    pygame.init()
    try:
        surface = pygame.Surface((1000, 520))
        draw_progress_graph(surface, surface.get_rect(), [], pygame.font.Font(None, 22),
                            f"Best fitness per seed: {settings['track']} track", series=series)
        pygame.image.save(surface, str(folder / 'comparison.png'))
    finally:
        pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_track_argument(parser, default='flowing')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--population', type=int, default=POPULATION_SIZE)
    parser.add_argument('--legacy-fitness', action='store_true')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--output', help='Folder for results (default experiments/<time>-<track>)')
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    folder = Path(args.output) if args.output else (
        Path(__file__).resolve().parent / 'experiments' / f'{stamp}-{args.track}')
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(track=args.track, seeds=args.seeds, generations=args.generations,
                    population=args.population, legacy=args.legacy_fitness)
    jobs = [dict(settings, seed=s, folder=str(folder / f'seed-{s}')) for s in args.seeds]
    print(f'Running {len(jobs)} seeds on {args.workers} worker(s) -> {folder}', flush=True)
    with multiprocessing.Pool(min(args.workers, len(jobs))) as pool:
        results = sorted(pool.map(run_seed, jobs), key=lambda r: r['seed'])
    report = write_report(folder, settings, results)
    save_comparison(folder, settings, results)
    print('\n' + report)
    print(f'\nSaved {folder / "summary.md"} and {folder / "comparison.png"}')
    return folder


if __name__ == '__main__':
    main()
