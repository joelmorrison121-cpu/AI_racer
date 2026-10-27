import pygame
from settings import MAX_SPEED


def get_observation(car, track, race, sensors, lookahead=False):
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

    observation = sensors.get_readings() + [
        forward_speed,
        sideways_speed,
        target_forward,
        target_sideways,
    ]
    if lookahead:
        # Direction to the checkpoint AFTER the next one, so a hairpin is
        # visible before the car reaches it.
        after = track.checkpoints[(race.next_checkpoint + 1) % len(track.checkpoints)]
        to_after = pygame.Vector2(after.centre) - car.position
        if to_after.length_squared() > 0:
            to_after = to_after.normalize()
        observation += [to_after.dot(forward), to_after.dot(sideways)]
    return observation


def observe(network, car, track, race, sensors):
    """The observation this particular network was built for (11 or 13 inputs)."""
    return get_observation(car, track, race, sensors, lookahead=network.uses_lookahead)


def choose_action(network, observation):
    steering, accelerator_signal, brake_signal = (
        network.forward(observation)
    )

    accelerating = accelerator_signal > 0
    braking = brake_signal > 0

    # Match the argument order expected by Car.update().
    return accelerating, braking, steering