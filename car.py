"""Car state and physics. No keyboard or lap rules belong here."""
import pygame
import settings


class Car:
    def __init__(self, start_position, start_angle=0):
        self.start_position = tuple(start_position)
        self.start_angle = start_angle
        self.base_image = pygame.Surface((44, 24), pygame.SRCALPHA)
        pygame.draw.rect(self.base_image, "orange", (0, 0, 44, 24), border_radius=5)
        pygame.draw.rect(self.base_image, "lightblue", (28, 4, 9, 16))
        self.reset()

    def reset(self):
        self.position = pygame.Vector2(self.start_position)
        self.previous_position = self.position.copy()
        self.velocity = pygame.Vector2()
        self.angle = self.start_angle
        self.speed = 0.0
        self.refresh_shape()

    def update(self, dt, accelerating, braking_now, steering):
        old_forward = pygame.Vector2(1, 0).rotate(-self.angle)
        forward_speed = self.velocity.dot(old_forward)
        steering_strength = min(max(forward_speed, 0) / 100, 1)
        self.angle += steering * settings.TURN_SPEED * steering_strength * dt

        forward = pygame.Vector2(1, 0).rotate(-self.angle)
        sideways = pygame.Vector2(-forward.y, forward.x)
        forward_speed = self.velocity.dot(forward)
        sideways_speed = self.velocity.dot(sideways)
        if accelerating:
            forward_speed += settings.ACCELERATION * dt
        slowdown = 0.0
        if not accelerating:
            slowdown += settings.ROLLING_RESISTANCE * dt
        if braking_now:
            slowdown += settings.BRAKING * dt
        if forward_speed > 0:
            forward_speed = max(0, forward_speed - slowdown)
        elif forward_speed < 0:
            forward_speed = min(0, forward_speed + slowdown)
        sideways_speed *= 2 ** (-settings.GRIP * dt)
        self.velocity = forward * forward_speed + sideways * sideways_speed
        if self.velocity.length() > settings.MAX_SPEED:
            self.velocity.scale_to_length(settings.MAX_SPEED)
        self.speed = self.velocity.length()
        self.previous_position = self.position.copy()
        self.position += self.velocity * dt
        self.refresh_shape()

    def refresh_shape(self):
        # Image, rectangle and mask must always describe the same pose.
        self.image = pygame.transform.rotate(self.base_image, self.angle)
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image)

    def draw(self, screen, camera):
        screen_rect = camera.apply_rect(self.rect)
        screen.blit(self.image, screen_rect)
