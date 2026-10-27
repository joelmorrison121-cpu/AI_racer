import pygame


class Camera:
    def __init__(self, screen_size, world_size):
        self.screen_width, self.screen_height = screen_size
        self.world_width, self.world_height = world_size

        # World coordinates of the window's top-left corner.
        self.position = pygame.Vector2()

    def update(self, target_position):
        # Aim to place the target in the middle of the window.
        target_x = target_position.x - self.screen_width / 2
        target_y = target_position.y - self.screen_height / 2

        # Maximum scrolling distances.
        max_x = max(0, self.world_width - self.screen_width)
        max_y = max(0, self.world_height - self.screen_height)

        self.position.x = max(0, min(target_x, max_x))
        self.position.y = max(0, min(target_y, max_y))

    def apply_point(self, world_position):
        return pygame.Vector2(world_position) - self.position

    def apply_rect(self, world_rect):
        # Return a shifted copy; never move the original world rectangle.
        return world_rect.move(
            -round(self.position.x),
            -round(self.position.y),
        )