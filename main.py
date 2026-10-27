"""Crash Course. Run with no options for the menu.

    python main.py                          # menu: freeplay, train, track select
    python main.py --track flowing          # skip the menu and drive a track
    python main.py --track flowing --ai     # ...starting with the champion driving
In game: Tab switches you / champion, R resets, T prints the AI's inputs, Esc = menu.
"""
import argparse
from pathlib import Path

import pygame
from car import Car
from track import make_track, TRACK_NAMES
from race import Race
from hud import HUD
from camera import Camera
from sensors import Sensors
from controller import observe, choose_action
from evolution import load_champion as read_champion
from settings import (
    WIDTH,
    HEIGHT,
    FPS,
    SIMULATION_DT,
    MAX_FRAME_TIME,
)


def read_controls(keys):
    accelerating = keys[pygame.K_w] or keys[pygame.K_UP]
    braking = keys[pygame.K_s] or keys[pygame.K_DOWN]
    steering = (int(keys[pygame.K_a] or keys[pygame.K_LEFT])
                - int(keys[pygame.K_d] or keys[pygame.K_RIGHT]))
    return accelerating, braking, steering


def reset_attempt(car, race):
    car.reset()
    race.reset_attempt()


def update_game(car, track, race, dt, accelerating, braking, steering):
    car.update(dt, accelerating, braking, steering)
    race.advance_timer(dt, car.speed)
    if track.collides(car):
        reset_attempt(car, race)
    else:
        race.check_progress(track, car.previous_position, car.position)


HERE = Path(__file__).resolve().parent


def champion_path(track_name):
    """The champion that goes with a track: champions/<track>.json, or the
    original root champion.json for the classic track."""
    candidate = HERE / 'champions' / f'{track_name}.json'
    if candidate.exists():
        return candidate
    return HERE / 'champion.json'


def load_champion(path=None, track_name='classic'):
    path = Path(path) if path else HERE / "champion.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} was not found. Run python train.py first.")
    network, data = read_champion(path)
    trained_on = data.get("track", "classic")
    print(f"Loaded champion from generation {data['generation']} "
          f"with fitness {data['fitness']:.1f} (trained on '{trained_on}')")
    if trained_on != track_name:
        print(f"Note: this champion was trained on '{trained_on}', not '{track_name}'.")
    return network


def champion_benchmark(network, track):
    """Drive the champion once, off-screen, to get the lap time you're racing."""
    from evaluation import run_attempt
    agent = run_attempt(network, track, progress='track', lap_bonus=True)
    if agent.status == 'completed lap':
        return dict(finished=True, lap=agent.race.last_lap_time)
    gates = len(track.checkpoints)
    reached = (agent.race.next_checkpoint
               + track.progress_fraction(agent.route_progress, agent.race.next_checkpoint)) / gates
    return dict(finished=False, percent=round(100 * reached))


def run_freeplay(screen, clock, track, network, start_ai=False, smoke_frames=None,
                 champion_label=''):
    """Drive the track yourself; Tab hands the wheel to the champion.

    Returns 'menu' (Esc) or 'quit' (window closed)."""
    car = Car(track.start_position, track.start_angle)
    race = Race(verbose=False)
    hud = HUD()
    camera = Camera(screen.get_size(), track.world_size)
    sensors = Sensors()
    benchmark = champion_benchmark(network, track)
    player_best = None          # best lap YOU drove (AI laps don't count)
    ai_enabled = start_ai
    frames = 0
    accumulator = 0.0
    banner_until = 0.0
    clock.tick()
    while True:
        frame_time = min(clock.tick(FPS) / 1000, MAX_FRAME_TIME)
        accumulator += frame_time
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return 'menu'
            if event.key == pygame.K_r:
                reset_attempt(car, race)
            elif event.key == pygame.K_t:
                sensors.update(car, track)
                observation = observe(network, car, track, race, sensors)
                accelerating, braking, steering = choose_action(network, observation)
                print("Inputs:", [round(x, 2) for x in observation])
                print(f"AI decision: accelerate={accelerating}, brake={braking}, "
                      f"steering={steering:.2f}")
            elif event.key == pygame.K_TAB:
                ai_enabled = not ai_enabled
                reset_attempt(car, race)   # each mode starts from a fresh attempt
                race.best_lap_time = race.last_lap_time = None
                race.laps_completed = 0

        keys = pygame.key.get_pressed()
        while accumulator >= SIMULATION_DT:
            laps_before = race.laps_completed
            if ai_enabled:
                sensors.update(car, track)
                accelerating, braking, steering = choose_action(
                    network, observe(network, car, track, race, sensors))
            else:
                accelerating, braking, steering = read_controls(keys)
            update_game(car, track, race, SIMULATION_DT, accelerating, braking, steering)
            if race.laps_completed > laps_before and not ai_enabled:
                lap = race.last_lap_time
                player_best = lap if player_best is None else min(player_best, lap)
                if benchmark['finished'] and lap < benchmark['lap']:
                    banner_until = frames + FPS * 3
            accumulator -= SIMULATION_DT

        sensors.update(car, track)
        camera.update(car.position)
        track.draw(screen, race.next_checkpoint, camera)
        sensors.draw(screen, camera, car)
        car.draw(screen, camera)
        hud.draw(screen, car, race, len(track.checkpoints))
        hud.draw_race(screen, track.name, ai_enabled, benchmark, player_best,
                      champion_label, frames < banner_until)
        pygame.display.flip()
        frames += 1
        if smoke_frames and frames >= smoke_frames:
            return 'quit'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--track', choices=TRACK_NAMES,
                        help='Skip the menu and drive this track straight away.')
    parser.add_argument("--champion", help="Champion JSON to play (default: the track's champion).")
    parser.add_argument("--ai", action="store_true", help="Start in AI mode (Tab still switches).")
    parser.add_argument("--smoke-frames", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Crash Course")
        clock = pygame.time.Clock()
        if args.track is None and args.champion is None:
            from menu import run_menu
            run_menu(screen, clock, smoke_frames=args.smoke_frames)
            return
        track = make_track(args.track or 'classic')
        path = Path(args.champion) if args.champion else champion_path(track.name)
        network = load_champion(path, track.name)
        run_freeplay(screen, clock, track, network, args.ai, args.smoke_frames,
                     champion_label=path.stem)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
