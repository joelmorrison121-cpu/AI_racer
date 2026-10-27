"""Curved tracks, angled checkpoints, progress along the route, lap reward.
Run: python -m unittest -v"""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import unittest
import pygame
from agent import Agent
from car import Car
from network import NeuralNetwork
from race import Race
from track import Checkpoint, TRACK_NAMES, make_track


def drive_along_centre(track, race, step=5.0, laps=1):
    """Move a point along the centre line; return the checkpoint order seen."""
    seen = []
    s, previous = 0.0, pygame.Vector2(track.start_position)
    while race.laps_completed < laps and s < track.centre_line.length * (laps + 1):
        s += step
        current, _ = track.centre_line.point_at(s)
        before = race.next_checkpoint
        race.check_progress(track, previous, current)
        if race.next_checkpoint != before or race.laps_completed:
            seen.append(before)
        previous = current
    return seen


class TrackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_diagonal_checkpoint_is_directional(self):
        gate = Checkpoint((0, 0), (100, 100), (1, -1))      # 45 degree gate
        self.assertTrue(gate.crossed(pygame.Vector2(30, 70), pygame.Vector2(70, 30)))
        self.assertFalse(gate.crossed(pygame.Vector2(70, 30), pygame.Vector2(30, 70)))  # backwards
        self.assertFalse(gate.crossed(pygame.Vector2(130, 170), pygame.Vector2(170, 130)))  # beside it

    def test_rect_checkpoints_match_v1(self):
        gate = Checkpoint.from_rect((1600, 680, 300, 40), (0, 1))
        self.assertEqual(gate.centre, pygame.Vector2(1750, 700))
        self.assertTrue(gate.crossed(pygame.Vector2(1601, 690), pygame.Vector2(1601, 710)))
        self.assertFalse(gate.crossed(pygame.Vector2(1900, 690), pygame.Vector2(1900, 710)))

    def test_every_track_starts_on_road_and_laps_in_order(self):
        for name in TRACK_NAMES:
            with self.subTest(track=name):
                track = make_track(name)
                car = Car(track.start_position, track.start_angle)
                self.assertFalse(track.collides(car), 'car starts on grass')
                race = Race()
                seen = drive_along_centre(track, race)
                self.assertEqual(race.laps_completed, 1)
                self.assertEqual(seen, list(range(len(track.checkpoints))))

    def test_progress_increases_along_route(self):
        for name in ('flowing', 'technical'):
            track = make_track(name)
            hint, last = None, -1.0
            for k in range(1, 400):
                point, _ = track.centre_line.point_at(k * track.centre_line.length / 400)
                s, hint = track.centre_line.locate(point, hint)
                self.assertGreater(s + 1e-6, last, f'{name}: progress went backwards at step {k}')
                last = s

    def test_hairpin_hint_prevents_jumping_across(self):
        # On the technical track's hairpin the two legs are close. With a hint,
        # a car near the end of the first leg stays on it.
        track = make_track('technical')
        line = track.centre_line
        for k in range(200):
            s_true = k * line.length / 200
            point, _ = line.point_at(s_true)
            _, hint = line.locate(point)
            s, _ = line.locate(point, hint)
            gap = abs((s - s_true + line.length / 2) % line.length - line.length / 2)
            self.assertLess(gap, 8)     # same place (arc length L is the start again)

    def test_progress_fraction_rewards_forward_driving_only(self):
        track = make_track('flowing')
        begin, end = track.segment_start(1), track.gate_progress(1)
        self.assertAlmostEqual(track.progress_fraction(begin + (end - begin) / 2, 1), 0.5)
        self.assertEqual(track.progress_fraction(begin - 50, 1), 0.0)     # behind = nothing

    def test_faster_lap_scores_higher(self):
        track = make_track('easy')
        scores = []
        for lap_time in (20.0, 30.0):
            agent = Agent(NeuralNetwork(seed=1), track)
            agent.race.last_lap_time = lap_time
            scores.append(agent.lap_fitness(track))
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], len(track.checkpoints) * 1000)


if __name__ == '__main__':
    unittest.main()
