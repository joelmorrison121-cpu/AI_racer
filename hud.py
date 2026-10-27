"""Screen-space information. This will stay fixed when a camera is added."""
import pygame


def format_time(value):
    return "--" if value is None else f"{value:.2f}s"


class HUD:
    def __init__(self):
        self.font = pygame.font.Font(None, 28)

    def draw(self, screen, car, race, checkpoint_count):
        lines = [
            f"Speed: {car.speed:.0f} px/s",
            f"Laps completed: {race.laps_completed}",
            f"Next checkpoint: {race.next_checkpoint + 1}/{checkpoint_count}",
            f"Current lap: {race.lap_time:.2f}s",
            f"Last lap: {format_time(race.last_lap_time)}",
            f"Best lap: {format_time(race.best_lap_time)}",
        ]
        pygame.draw.rect(
            screen,
            (20, 30, 25),
            (15, 15, 300, 200),
            border_radius=10,
        )
        for index, line in enumerate(lines):
            text_image = self.font.render(line, True, (240, 240, 240))
            screen.blit(text_image, (30, 25 + index * 30))
