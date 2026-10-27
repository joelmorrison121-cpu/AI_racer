"""Run with: py -m unittest test_live_training -v"""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import copy
from pathlib import Path
import random
import tempfile
import unittest
import pygame
from agent import Agent
from camera import Camera
from evaluation import evaluate
from evolution import breed, new_population
from live_train import TrainingSession, draw_world, draw_dashboard
from network import NeuralNetwork
from settings import SIMULATION_DT
from track import Track


def throttle_network():
    network = NeuralNetwork(seed=1)
    network.output_weights = [[0.0] * 8 for _ in range(3)]
    network.output_biases = [0, 1, -1]
    return network


class LiveTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((1000, 700))
        cls.track = Track()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_colours_and_interleaving_do_not_change_evaluation(self):
        nets = [NeuralNetwork(seed=42), throttle_network()]
        agents = [Agent(n, self.track, i, (10 + i * 100, 200, 80)) for i, n in enumerate(nets)]
        while any(a.active for a in agents):
            for a in agents:
                a.step(SIMULATION_DT, self.track)
        for n, a in zip(nets, agents):
            self.assertEqual((a.fitness, a.status), evaluate(n, self.track))
        self.assertEqual(agents[0].status, 'no progress')
        self.assertEqual(agents[1].status, 'crashed')

    def test_dead_agent_stops_and_drawing_preserves_mask(self):
        a = Agent(throttle_network(), self.track)
        while a.active:
            a.step(SIMULATION_DT, self.track)
        position = a.car.position.copy()
        mask_count = a.car.mask.count()
        alpha = a.car.image.get_alpha()
        a.step(SIMULATION_DT, self.track)
        a.draw(self.screen, Camera((1000, 700), self.track.world_size), True)
        self.assertEqual(a.car.position, position)
        self.assertEqual(a.car.mask.count(), mask_count)
        self.assertEqual(a.car.image.get_alpha(), alpha)

    def test_finishing_and_time_limit(self):
        for lap_bonus in (False, True):
            a = Agent(throttle_network(), self.track, lap_bonus=lap_bonus)
            a.race.next_checkpoint = 3
            a.car.position.update(499, 250)
            a.car.velocity.update(200, 0)
            a.step(SIMULATION_DT, self.track)
            self.assertEqual(a.status, 'completed lap')
            if lap_bonus:   # faster laps score more, but never 1000+ extra
                self.assertTrue(4000 < a.fitness < 5000)
            else:           # v1.0 scoring
                self.assertEqual(a.fitness, 4000)
        b = Agent(NeuralNetwork(seed=42), self.track)
        b.max_steps = 1
        b.step(SIMULATION_DT, self.track)
        self.assertEqual(b.status, 'time limit')

    def test_elites_are_preserved_and_independent(self):
        population = new_population(random.Random(42))
        before = copy.deepcopy(population[0].hidden_weights)
        children = breed(population, random.Random(99))
        self.assertEqual(len(children), 30)
        self.assertEqual(children[0].hidden_weights, before)
        children[0].hidden_weights[0][0] += 1
        self.assertEqual(population[0].hidden_weights, before)

    def test_generation_save_transition_and_render(self):
        with tempfile.TemporaryDirectory() as folder:
            session = TrainingSession(self.track, Path(folder))
            for a in session.agents:
                a.max_steps = 1
            session.step()
            self.assertTrue(session.generation_done)
            self.assertTrue((Path(folder) / 'champion.json').exists())
            self.assertTrue((Path(folder) / 'history.json').exists())
            draw_world(self.screen, Camera((1000, 545), self.track.world_size),
                       session, pygame.font.Font(None, 24), True)
            draw_dashboard(self.screen, session, pygame.font.Font(None, 24), False, 1, True)
            session.next_generation()
            self.assertEqual(session.generation, 2)
            self.assertTrue(all(a.active for a in session.agents))


if __name__ == '__main__':
    unittest.main()
