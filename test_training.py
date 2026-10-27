"""Generation limit, progress graph and save/resume. Run: python -m unittest -v"""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import json
from pathlib import Path
import tempfile
import unittest
import pygame
from evolution import network_to_dict
from track import Track
from training import TrainingSession


def run(session, until):
    while True:
        session.run_generation()
        if session.finished or session.generation >= until:
            return session
        session.next_generation()


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.track = Track()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_generation_limit_history_and_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            session = run(TrainingSession(self.track, folder, seed=3, generations=2,
                                          population_size=8), 99)
            self.assertTrue(session.finished)
            self.assertEqual(session.generation, 2)
            history = json.loads((Path(folder) / 'history.json').read_text())
            self.assertEqual([h['generation'] for h in history], [1, 2])
            for key in ('best', 'average', 'completion_rate', 'outcomes'):
                self.assertIn(key, history[0])
            self.assertTrue((Path(folder) / 'progress.png').stat().st_size > 1000)
            session.next_generation()               # must not run past the limit
            self.assertEqual(session.generation, 2)

    def test_resume_continues_exactly(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            straight = run(TrainingSession(self.track, a, seed=5, generations=3,
                                           population_size=8, save_graph=False), 3)
            first = run(TrainingSession(self.track, b, seed=5, generations=2,
                                        population_size=8, save_graph=False), 2)
            self.assertTrue(first.finished)
            resumed = TrainingSession.resume(b, Track(), generations=3, save_graph=False)
            self.assertFalse(resumed.finished)
            resumed.next_generation()
            run(resumed, 3)
            self.assertEqual(resumed.history, straight.history)
            self.assertEqual([network_to_dict(x.network) for x in resumed.agents],
                             [network_to_dict(x.network) for x in straight.agents])
            self.assertEqual(resumed.rng.getstate(), straight.rng.getstate())

    def test_resume_rejects_a_different_track(self):
        with tempfile.TemporaryDirectory() as folder:
            run(TrainingSession(self.track, folder, generations=1, population_size=8,
                                save_graph=False), 1)
            other = Track()
            other.name = 'somewhere-else'
            with self.assertRaises(ValueError):
                TrainingSession.resume(folder, other)


if __name__ == '__main__':
    unittest.main()
