import pygame
from settings import MAX_SPEED


def get_observation(car, track, race, sensors):
    forward = pygame.Vector2(1, 0).rotate(-car.angle)
    sideways = pygame.Vector2(-forward.y, forward.x)

    # Velocity relative to the car's heading.
    forward_speed = car.velocity.dot(forward) / MAX_SPEED
    sideways_speed = car.velocity.dot(sideways) / MAX_SPEED

    # Direction from the car to the next checkpoint's centre.
    checkpoint = track.checkpoints[race.next_checkpoint]
    target_position = pygame.Vector2(checkpoint.centre)

    to_target = target_position - car.position

    # A zero-length vector cannot be normalised.
    if to_target.length_squared() > 0:
        to_target = to_target.normalize()

    target_forward = to_target.dot(forward)
    target_sideways = to_target.dot(sideways)

    return sensors.get_readings() + [
        forward_speed,
        sideways_speed,
        target_forward,
        target_sideways,
    ]


def choose_action(network, observation):
    steering, accelerator_signal, brake_signal = (
        network.forward(observation)
    )

    accelerating = accelerator_signal > 0
    braking = brake_signal > 0

    # Match the argument order expected by Car.update().
    return accelerating, braking, steering