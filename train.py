"""Fast headless training with exactly the same rules as live_train.py.

    python train.py                          # 20 generations, seed 42
    python train.py --generations 50 --seed 7
    python train.py --resume training_runs/<run> --generations 60
    python train.py --track flowing --generations 40
    python train.py --legacy-fitness           # v1.0 rules: reproduces champion.json
"""
import argparse
import pygame
from training import add_training_arguments, validate_training_arguments, create_session, track_for


def train(args):
    track = track_for(args)
    session = create_session(args, track)
    if session.generation_done:          # resumed: breed the saved population first
        session.next_generation()
    while True:
        session.run_generation()
        if session.finished:
            break
        session.next_generation()
    print(f'Champion: {session.output_directory / "champion.json"}')
    print(f'Graph:    {session.output_directory / "progress.png"}')
    return session


def main(argv=None):
    parser = add_training_arguments(argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter))
    args = parser.parse_args(argv)
    validate_training_arguments(parser, args)
    pygame.init()
    try:
        return train(args)
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
