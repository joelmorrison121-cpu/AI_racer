"""Run with: py -m unittest -v. Uses an off-screen SDL display."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import unittest
import pygame
from car import Car
from track import Track
from race import Race
from hud import HUD
from camera import Camera
from main import update_game


class GameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((1000, 700))

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.track = Track()
        self.car = Car(self.track.start_position)
        self.race = Race()

    def test_acceleration_and_motion(self):
        update_game(self.car, self.track, self.race, .02, True, False, 0)
        self.assertAlmostEqual(self.car.speed, 5)
        self.assertAlmostEqual(self.car.position.x, 500.1)
        self.assertAlmostEqual(self.race.lap_time, .02)

    def test_directional_order_and_records(self):
        # A finish-line crossing alone must not count.
        self.race.check_progress(self.track, pygame.Vector2(490, 250), pygame.Vector2(510, 250))
        self.assertEqual(self.race.next_checkpoint, 0)
        first = self.track.checkpoints[0]
        self.assertFalse(first.crossed(pygame.Vector2(1750, 700), pygame.Vector2(1750, 660)))
        for duration in (14.2, 15.1):
            self.race.lap_time = duration
            for checkpoint in self.track.checkpoints:
                centre = pygame.Vector2(checkpoint.rect.center)
                self.race.check_progress(self.track, centre - checkpoint.forward * 30,
                                         centre + checkpoint.forward * 30)
        self.assertEqual(self.race.laps_completed, 2)
        self.assertAlmostEqual(self.race.best_lap_time, 14.2)
        self.assertAlmostEqual(self.race.last_lap_time, 15.1)

    def test_crash_resets_attempt_preserves_best(self):
        self.race.best_lap_time = 12
        self.race.next_checkpoint = 2
        self.car.position.update(500, 700)  # On the central grass
        update_game(self.car, self.track, self.race, .02, False, False, 0)
        self.assertEqual(self.car.position, pygame.Vector2(500, 250))
        self.assertEqual(self.car.speed, 0)
        self.assertEqual(self.race.next_checkpoint, 0)
        self.assertEqual(self.race.lap_time, 0)
        self.assertEqual(self.race.best_lap_time, 12)
        self.assertFalse(self.track.collides(self.car))

    def test_grip_and_independent_cars(self):
        self.car.velocity.update(100, 50)
        self.car.update(.02, False, False, 0)
        self.assertLess(self.car.velocity.y, 50)
        other = Car((500, 250))
        self.assertEqual(other.velocity.length(), 0)
        self.car.reset()
        self.assertEqual(self.car.velocity.length(), 0)

    def test_render(self):
        camera = Camera(self.screen.get_size(), self.track.world_size)
        camera.update(self.car.position)

        self.track.draw(self.screen, 0, camera)
        self.car.draw(self.screen, camera)
        HUD().draw(self.screen, self.car, self.race, 4)

        self.assertEqual(
            tuple(self.screen.get_at((10, 10)))[:3],
            (45, 110, 55),
        )


if __name__ == '__main__':
    unittest.main()
