"""TrainingSession: one evolutionary experiment, shared by live and headless training.

Each finished generation writes into the run folder:
    champion.json    best network so far (same format main.py loads)
    history.json     per-generation best/average fitness, outcomes, completion rate
    progress.png     graph of history.json
    checkpoint.json  everything needed to resume: population, fitnesses,
                     generation number and the random generator's exact state
"""
from collections import Counter
import json
import random
from pathlib import Path

import pygame

from agent import Agent
from evolution import (POPULATION_SIZE, GENERATIONS, new_population, breed, save_champion,
                       write_json, network_to_dict, network_from_dict,
                       rng_state_to_json, rng_state_from_json)
from progress_graph import save_progress_png
from settings import SIMULATION_DT
from track import add_track_argument

CHECKPOINT_VERSION = 1


def car_colour(index, population_size):
    colour = pygame.Color(0)
    colour.hsva = (index * 360 / population_size, 75, 100, 100)
    return tuple(colour)[:3]


class TrainingSession:
    def __init__(self, track, output_directory, seed=42, generations=GENERATIONS,
                 population_size=POPULATION_SIZE, save_graph=True, progress='track',
                 lap_bonus=True, _restoring=False):
        self.track = track
        self.progress = progress
        self.lap_bonus = lap_bonus
        self.output_directory = Path(output_directory)
        self.seed = seed
        self.generations = generations
        self.population_size = population_size
        self.save_graph = save_graph
        self.rng = random.Random(seed)
        self.generation = 1
        self.best_ever = -1.0
        self.best_lap = None
        self.history = []
        self.generation_done = False
        self.agents = [] if _restoring else self.make_agents(
            new_population(self.rng, population_size))

    # ----- population -------------------------------------------------------
    def make_agents(self, networks):
        return [Agent(network, self.track, i + 1, car_colour(i, len(networks)),
                      progress=self.progress, lap_bonus=self.lap_bonus)
                for i, network in enumerate(networks)]

    @property
    def finished(self):
        return self.generation_done and self.generation >= self.generations

    @property
    def leader(self):
        # Follow the strongest ACTIVE driver; when none remain, show the winner.
        active = [agent for agent in self.agents if agent.active]
        return max(active or self.agents, key=lambda agent: agent.fitness)

    def step(self):
        if self.generation_done:
            return
        for agent in self.agents:
            agent.step(SIMULATION_DT, self.track)
        if not any(agent.active for agent in self.agents):
            self.finish_generation()

    def run_generation(self):
        """Headless: simulate the current generation to completion."""
        while not self.generation_done:
            self.step()

    # ----- end of generation ------------------------------------------------
    def finish_generation(self):
        self.generation_done = True
        ranked = sorted(self.agents, key=lambda agent: agent.fitness, reverse=True)
        winner = ranked[0]
        if winner.fitness > self.best_ever:
            self.best_ever = winner.fitness
            save_champion(winner.network, winner.fitness, self.generation,
                          self.output_directory / 'champion.json',
                          track=getattr(self.track, 'name', 'classic'), seed=self.seed,
                          progress=self.progress, lap_bonus=self.lap_bonus)
        lap_times = [a.race.last_lap_time for a in self.agents
                     if a.status == 'completed lap']
        generation_lap = min(lap_times) if lap_times else None
        if generation_lap is not None:
            self.best_lap = (generation_lap if self.best_lap is None
                             else min(self.best_lap, generation_lap))
        entry = dict(generation=self.generation, best=winner.fitness,
                     average=sum(a.fitness for a in self.agents) / len(self.agents),
                     outcomes=dict(Counter(a.status for a in self.agents)),
                     completion_rate=len(lap_times) / len(self.agents),
                     generation_best_lap=generation_lap,
                     best_lap=self.best_lap)
        self.history.append(entry)
        write_json(self.output_directory / 'history.json', self.history)
        if self.save_graph:
            save_progress_png(self.history, self.output_directory / 'progress.png',
                              title=f'Training progress: {getattr(self.track, "name", "classic")}'
                                    f' track, seed {self.seed}')
        self.save_checkpoint()
        print(f'Generation {self.generation}: best {winner.fitness:.1f}, '
              f'average {entry["average"]:.1f}, completed {len(lap_times)}/{len(self.agents)}, '
              f'{entry["outcomes"]}', flush=True)

    def next_generation(self):
        if not self.generation_done or self.finished:
            return
        ranked = sorted(self.agents, key=lambda a: a.fitness, reverse=True)
        networks = breed([agent.network for agent in ranked], self.rng, self.population_size)
        self.generation += 1
        self.agents = self.make_agents(networks)
        self.generation_done = False

    # ----- save and resume --------------------------------------------------
    def save_checkpoint(self):
        """Saved after breeding inputs are final but BEFORE the next breed() call,
        so resuming replays exactly the same random draws."""
        write_json(self.output_directory / 'checkpoint.json', dict(
            version=CHECKPOINT_VERSION,
            track=getattr(self.track, 'name', 'classic'),
            seed=self.seed,
            population_size=self.population_size,
            progress=self.progress,
            lap_bonus=self.lap_bonus,
            generation=self.generation,
            generations=self.generations,
            best_ever=self.best_ever,
            best_lap=self.best_lap,
            history=self.history,
            rng_state=rng_state_to_json(self.rng),
            population=[dict(fitness=a.fitness, status=a.status, **network_to_dict(a.network))
                        for a in self.agents],
        ))

    @classmethod
    def resume(cls, run_directory, track, generations=None, save_graph=True):
        """Continue a run from its checkpoint.json. `generations` can extend the limit."""
        run_directory = Path(run_directory)
        data = json.loads((run_directory / 'checkpoint.json').read_text(encoding='utf-8'))
        if data.get('version') != CHECKPOINT_VERSION:
            raise ValueError('Unsupported checkpoint version')
        track_name = getattr(track, 'name', 'classic')
        if data['track'] != track_name:
            raise ValueError(f"This run was trained on the '{data['track']}' track, "
                             f"not '{track_name}'. Pass --track {data['track']}.")
        session = cls(track, run_directory, seed=data['seed'],
                      generations=generations or data['generations'],
                      population_size=data['population_size'], save_graph=save_graph,
                      progress=data.get('progress', 'track'),
                      lap_bonus=data.get('lap_bonus', True), _restoring=True)
        rng_state_from_json(session.rng, data['rng_state'])
        session.generation = data['generation']
        session.best_ever = data['best_ever']
        session.best_lap = data['best_lap']
        session.history = data['history']
        session.agents = session.make_agents(
            [network_from_dict(n) for n in data['population']])
        for agent, saved in zip(session.agents, data['population']):
            agent.fitness = saved['fitness']
            agent.status = saved['status']
        session.generation_done = True
        return session


# ----- command-line options shared by train.py and live_train.py -------------
def add_training_arguments(parser):
    parser.add_argument('--generations', type=int, default=GENERATIONS,
                        help=f'Generation limit (default {GENERATIONS}). With --resume, '
                             'a higher value extends a finished run.')
    parser.add_argument('--population', type=int, default=POPULATION_SIZE,
                        help=f'Drivers per generation (default {POPULATION_SIZE}, minimum 8).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed. Same seed + settings = same experiment.')
    parser.add_argument('--resume', metavar='RUN_FOLDER',
                        help='Continue a previous run from its checkpoint.json.')
    parser.add_argument('--legacy-fitness', action='store_true',
                        help='v1.0 scoring: straight-line progress and no lap-time bonus '
                             '(reproduces the shipped champion on the classic track).')
    add_track_argument(parser)
    return parser


def validate_training_arguments(parser, args):
    if args.generations < 1:
        parser.error('--generations must be at least 1')
    if args.population < 8:
        parser.error('--population must be at least 8 (2 elites, 6 parents, 4 new networks)')


def create_session(args, track, output=None, save_graph=True):
    """Start a fresh session or resume one, according to parsed arguments."""
    from evolution import create_run_directory
    if args.resume:
        generations = args.generations if args.generations != GENERATIONS else None
        session = TrainingSession.resume(args.resume, track, generations, save_graph)
        print(f'Resuming {args.resume} after generation {session.generation} '
              f'(limit {session.generations})', flush=True)
        return session
    output = output or create_run_directory()
    print(f'Saving this run in: {output}', flush=True)
    legacy = getattr(args, 'legacy_fitness', False)
    return TrainingSession(track, output, seed=args.seed, generations=args.generations,
                           population_size=args.population, save_graph=save_graph,
                           progress='straight' if legacy else 'track', lap_bonus=not legacy)


def track_for(args):
    """The track to use: the one saved in a resumed run, otherwise --track."""
    from track import make_track
    if args.resume:
        data = json.loads((Path(args.resume) / 'checkpoint.json').read_text(encoding='utf-8'))
        return make_track(data['track'])
    return make_track(args.track)
