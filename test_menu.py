"""Menu flow and look-ahead inputs. Run: python -m unittest -v"""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import random
import unittest
from unittest import mock
import pygame
from agent import Agent
from controller import observe
from evolution import breed, new_population
from menu import Menu, TRACKS, GENERATION_PRESETS
from network import NeuralNetwork
from settings import SIMULATION_DT, WIDTH, HEIGHT
from track import make_track


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


class MenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        cls.menu = Menu(cls.screen, pygame.time.Clock())

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.menu.state, self.menu.track_index = 'title', 0

    def test_title_to_track_select_and_back(self):
        m = self.menu
        m.handle(key(pygame.K_RETURN))            # Freeplay is first
        self.assertEqual((m.state, m.mode), ('tracks', 'freeplay'))
        m.handle(key(pygame.K_RIGHT)); m.handle(key(pygame.K_RIGHT))
        self.assertEqual(TRACKS[m.track_index][0], 'flowing')
        for _ in range(10):
            m.handle(key(pygame.K_RIGHT))        # stops at the last track
        self.assertEqual(m.track_index, len(TRACKS) - 1)
        m.handle(key(pygame.K_ESCAPE))
        self.assertEqual(m.state, 'title')
        self.assertEqual(m.handle(key(pygame.K_ESCAPE)), 'quit')

    def test_train_flow_uses_chosen_generation_cap(self):
        m = self.menu
        m.title_index = 1
        m.handle(key(pygame.K_RETURN))
        self.assertEqual(m.mode, 'train')
        m.handle(key(pygame.K_RETURN))           # choose track -> generations
        self.assertEqual(m.state, 'generations')
        self.assertEqual(GENERATION_PRESETS[m.gen_index], 20)   # easy's recommendation
        m.gen_index = 0
        m.handle(key(pygame.K_RIGHT)); m.handle(key(pygame.K_RIGHT))
        captured = {}
        def fake_run(screen, clock, session, smoke_frames=None):
            captured['session'] = session
            return 'menu'
        with mock.patch('live_train.run_training', fake_run):
            m.handle(key(pygame.K_RETURN))
        self.assertEqual(captured['session'].generations, GENERATION_PRESETS[2])
        self.assertEqual(m.state, 'tracks')

    def test_every_screen_draws(self):
        m = self.menu
        m.draw_title(0.016, 1.0)
        m.state, m.mode = 'tracks', 'train'
        m.draw_tracks(0.016)
        m.draw_generations(0.016)


class LookaheadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    def test_lookahead_networks_get_13_inputs_and_keep_them(self):
        track = make_track('technical')
        net = NeuralNetwork(seed=3, inputs=13)
        agent = Agent(net, track)
        agent.sensors.update(agent.car, track)
        self.assertEqual(len(observe(net, agent.car, track, agent.race, agent.sensors)), 13)
        agent.step(SIMULATION_DT, track)
        children = breed(new_population(random.Random(1), 10, inputs=13), random.Random(2), 10)
        self.assertTrue(all(c.input_count == 13 for c in children))

    def test_default_networks_unchanged(self):
        # Same seed -> same 11-input weights as v1.0 (keeps old champions valid).
        self.assertEqual(NeuralNetwork(seed=42).hidden_weights,
                         NeuralNetwork(seed=42, inputs=11).hidden_weights)
        self.assertEqual(NeuralNetwork(seed=42).input_count, 11)


if __name__ == '__main__':
    unittest.main()
