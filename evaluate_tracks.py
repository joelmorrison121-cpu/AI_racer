"""Test a saved champion on every track, including ones it never trained on.

    python evaluate_tracks.py                              # root champion.json
    python evaluate_tracks.py --champion training_runs/<run>/champion.json
    python evaluate_tracks.py --tracks easy technical

For each track it reports how the attempt ended, checkpoints passed, how far
round the lap the car got (measured along the centre line) and the lap time.
"""
import argparse
from pathlib import Path

import pygame
from evaluation import run_attempt
from evolution import load_champion
from track import TRACK_NAMES, make_track


def attempt_report(network, track_name):
    # Always judged the same way: progress along the route (the network's
    # inputs and driving don't depend on how fitness is scored).
    track = make_track(track_name)
    agent = run_attempt(network, track, progress='track', lap_bonus=True)
    gates = len(track.checkpoints)
    if agent.status == 'completed lap':
        passed, lap_fraction = gates, 1.0
    else:
        passed = agent.race.next_checkpoint
        lap_fraction = (passed + track.progress_fraction(agent.route_progress, passed)) / gates
    return dict(track=track_name, status=agent.status, checkpoints=f'{passed}/{gates}',
                lap_percent=round(100 * lap_fraction, 1), fitness=round(agent.fitness, 1),
                lap_time=agent.race.last_lap_time if agent.status == 'completed lap' else None)


def evaluate_champion(path, tracks=None):
    network, data = load_champion(path)
    trained_on = data.get('track', 'classic')
    rows = []
    for name in tracks or TRACK_NAMES:
        row = attempt_report(network, name)
        row['seen_in_training'] = name == trained_on
        rows.append(row)
    return trained_on, rows


def format_table(rows):
    lines = ['| Track | Trained on it? | Result | Checkpoints | Lap % | Lap time |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lap = f"{r['lap_time']:.2f}s" if r['lap_time'] else '-'
        lines.append(f"| {r['track']} | {'yes' if r['seen_in_training'] else 'no (unseen)'} | "
                     f"{r['status']} | {r['checkpoints']} | {r['lap_percent']}% | {lap} |")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--champion', default=str(Path(__file__).with_name('champion.json')))
    parser.add_argument('--tracks', nargs='+', choices=TRACK_NAMES, default=TRACK_NAMES)
    args = parser.parse_args(argv)
    pygame.init()
    try:
        trained_on, rows = evaluate_champion(args.champion, args.tracks)
    finally:
        pygame.quit()
    print(f'Champion: {args.champion} (trained on {trained_on})\n')
    print(format_table(rows))
    return rows


if __name__ == '__main__':
    main()
