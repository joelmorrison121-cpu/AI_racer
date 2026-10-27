"""Drive manually or watch a saved champion.

    python main.py                                  # classic track, champion.json
    python main.py --track flowing --champion champions/flowing.json
Tab switches manual / AI, R resets, T prints the AI's inputs and decision.
"""
import argparse
from pathlib import Path

import pygame
from car import Car
from track import make_track, add_track_argument
from race import Race
from hud import HUD
from camera import Camera
from sensors import Sensors
from controller import get_observation, choose_action
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


def load_champion(path=None, track_name='classic'):
    path = Path(path) if path else Path(__file__).with_name("champion.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} was not found. Run python train.py first.")
    network, data = read_champion(path)
    trained_on = data.get("track", "classic")
    print(f"Loaded champion from generation {data['generation']} "
          f"with fitness {data['fitness']:.1f} (trained on '{trained_on}')")
    if trained_on != track_name:
        print(f"Note: this champion was trained on '{trained_on}', not '{track_name}'.")
    return network


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_track_argument(parser)
    parser.add_argument("--champion", help="Champion JSON to play (default champion.json).")
    parser.add_argument("--ai", action="store_true", help="Start in AI mode (Tab still switches).")
    parser.add_argument("--smoke-frames", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("My First AI Racer")
        clock = pygame.time.Clock()
        track = make_track(args.track)
        car = Car(track.start_position, track.start_angle)
        race = Race(verbose=True)
        hud = HUD()
        camera = Camera(screen.get_size(), track.world_size)
        sensors = Sensors()
        network = load_champion(args.champion, track.name)
        ai_enabled = args.ai
        frames = 0
        accumulator = 0.0
        running = True
        while running:
            frame_time = min(clock.tick(FPS) / 1000, MAX_FRAME_TIME)
            accumulator += frame_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    reset_attempt(car, race)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    sensors.update(car, track)

                    observation = get_observation(
                        car, track, race, sensors
                    )

                    accelerating, braking, steering = choose_action(
                        network, observation
                    )

                    print("Inputs:", [round(x, 2) for x in observation])
                    print(
                        f"AI decision: accelerate={accelerating}, "
                        f"brake={braking}, steering={steering:.2f}"
                    )
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    ai_enabled = not ai_enabled

                    # Start each mode from a fresh attempt.
                    reset_attempt(car, race)

                    mode = "AI" if ai_enabled else "Manual"
                    print(f"Control mode: {mode}")
            if not running:
                break

            keys = pygame.key.get_pressed()

            while accumulator >= SIMULATION_DT:
                if ai_enabled:
                    sensors.update(car, track)

                    observation = get_observation(
                        car, track, race, sensors
                    )

                    accelerating, braking, steering = choose_action(
                        network, observation
                    )
                else:
                    accelerating, braking, steering = read_controls(keys)

                update_game(
                    car,
                    track,
                    race,
                    SIMULATION_DT,
                    accelerating,
                    braking,
                    steering,
                )

                accumulator -= SIMULATION_DT

            # Draw once after processing the pending simulation steps.
            sensors.update(car, track)
            camera.update(car.position)

            mode = "AI" if ai_enabled else "Manual"
            pygame.display.set_caption(
                f"My First AI Racer | {mode} | Tab: switch mode"
            )

            track.draw(screen, race.next_checkpoint, camera)
            sensors.draw(screen, camera, car)
            car.draw(screen, camera)
            hud.draw(screen, car, race, len(track.checkpoints))

            pygame.display.flip()
            frames += 1
            if args.smoke_frames and frames >= args.smoke_frames:
                running = False
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()