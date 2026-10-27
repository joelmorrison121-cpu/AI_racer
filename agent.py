"""One controller's attempt. Rendering does not change its fitness or physics.

Fitness (see README "How it learns"):
  * 1000 per checkpoint passed in order, plus 0-999 partial credit towards the next.
  * progress='track' (default) measures partial credit ALONG THE CENTRE LINE, so a
    car is rewarded for following the route through bends. progress='straight' is
    the v1.0 straight-line distance to the next checkpoint.
  * lap_bonus=True (default): a completed lap scores checkpoints*1000 plus up to
    LAP_TIME_BONUS more the faster it was, so finishers keep improving.
"""
import pygame
from car import Car
from race import Race
from sensors import Sensors
from controller import get_observation, choose_action
from settings import SIMULATION_DT

LAP_TIME_BONUS = 1000.0
PROGRESS_MODES = ('track', 'straight')


class Agent:
    def __init__(self, network, track, number=1, colour=(255, 165, 0),
                 progress='track', lap_bonus=True):
        if progress not in PROGRESS_MODES:
            raise ValueError(f'progress must be one of {PROGRESS_MODES}')
        self.network = network
        self.progress_mode = progress
        self.lap_bonus = lap_bonus
        self.number = number
        self.colour = colour
        self.car = Car(track.start_position, track.start_angle)
        # Recolour only the body; preserve the original image's transparency.
        pygame.draw.rect(self.car.base_image, colour, (0, 0, 44, 24), border_radius=5)
        pygame.draw.rect(self.car.base_image, 'lightblue', (28, 4, 9, 16))
        self.car.refresh_shape()
        self.race = Race()
        self.sensors = Sensors()
        self.status = 'driving'
        self.fitness = 0.0
        self.steps = 0
        self.last_improvement_step = 0
        self.max_steps = round(getattr(track, 'time_limit', 30.0) / SIMULATION_DT)
        self.patience_steps = round(8 / SIMULATION_DT)
        self.starting_distance = self.distance_to_target(track)
        self.route_progress, self.route_index = track.centre_line.locate(self.car.position)

    @property
    def active(self):
        return self.status == 'driving'

    def distance_to_target(self, track):
        centre = pygame.Vector2(track.checkpoints[self.race.next_checkpoint].centre)
        return max(self.car.position.distance_to(centre), 1.0)

    def step(self, dt, track):
        if not self.active:
            return
        # All callers use the same fixed timestep.
        if abs(dt - SIMULATION_DT) > 1e-12:
            raise ValueError('Agent.step requires SIMULATION_DT')
        step_index = self.steps
        self.steps += 1
        self.sensors.update(self.car, track)
        observation = get_observation(self.car, track, self.race, self.sensors)
        actions = choose_action(self.network, observation)
        self.car.update(dt, *actions)
        self.race.advance_timer(dt, self.car.speed)

        if track.collides(self.car):
            self.status = 'crashed'
            return

        old_checkpoint = self.race.next_checkpoint
        self.race.check_progress(track, self.car.previous_position, self.car.position)
        if self.race.laps_completed:
            self.fitness = self.lap_fitness(track)
            self.status = 'completed lap'
            return

        if self.progress_mode == 'track':
            self.route_progress, self.route_index = track.centre_line.locate(
                self.car.position, self.route_index)
            progress = min(track.progress_fraction(self.route_progress,
                                                   self.race.next_checkpoint), 0.99)
        else:  # v1.0: straight-line distance to the next checkpoint's centre
            target = pygame.Vector2(track.checkpoints[self.race.next_checkpoint].centre)
            if self.race.next_checkpoint != old_checkpoint:
                self.starting_distance = self.distance_to_target(track)
            distance = self.car.position.distance_to(target)
            progress = max(0.0, min(1 - distance / self.starting_distance, 0.99))
        score = self.race.next_checkpoint * 1000 + progress * 1000
        if score > self.fitness + 1.0:
            self.last_improvement_step = step_index
        self.fitness = max(self.fitness, score)
        if step_index - self.last_improvement_step >= self.patience_steps:
            self.status = 'no progress'
        elif self.steps >= self.max_steps:
            self.status = 'time limit'

    def lap_fitness(self, track):
        base = len(track.checkpoints) * 1000.0
        if not self.lap_bonus:
            return base
        lap_time = self.race.last_lap_time or 0.0
        limit = getattr(track, 'time_limit', 30.0)
        return base + LAP_TIME_BONUS * max(0.0, 1 - lap_time / limit)

    def draw(self, screen, camera, leader=False, font=None):
        rect = camera.apply_rect(self.car.rect)
        image = self.car.image
        if not self.active and self.status != 'completed lap':
            # Fade a COPY. Changing the collision image's alpha would corrupt masks.
            image = image.copy()
            image.set_alpha(65)
        screen.blit(image, rect)
        if leader:
            pygame.draw.circle(screen, (255, 230, 80), rect.center, 34, 3)
            if font:
                label = font.render(f'LEADER #{self.number}', True, (255, 230, 80))
                screen.blit(label, (rect.centerx - label.get_width() // 2, rect.top - 27))
