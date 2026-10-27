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

    def draw_race(self, screen, track_name, ai_enabled, benchmark, player_best,
                  champion_label='', beat_banner=False):
        """Top-right panel: who is driving and the champion's lap to beat."""
        small = pygame.font.Font(None, 26)
        driver = 'CHAMPION AI driving' if ai_enabled else 'YOU are driving'
        if benchmark['finished']:
            target = f"Champion lap: {benchmark['lap']:.2f}s"
        else:
            target = f"Champion: no lap ({benchmark['percent']}%)"
        yours = f"Your best: {format_time(player_best)}"
        lines = [(f"{track_name.upper()}  ·  {driver}", (238, 208, 154)),
                 (target, (240, 240, 240)), (yours, (240, 240, 240)),
                 ("Tab: switch   R: reset   Esc: menu", (170, 175, 180))]
        width = 330
        x = screen.get_width() - width - 15
        pygame.draw.rect(screen, (20, 30, 25), (x, 15, width, 132), border_radius=10)
        for i, (text, colour) in enumerate(lines):
            screen.blit(small.render(text, True, colour), (x + 14, 26 + i * 28))
        if beat_banner:
            big = pygame.font.Font(None, 64)
            label = big.render('You beat the champion!', True, (255, 220, 90))
            box = label.get_rect(center=(screen.get_width() // 2, 250)).inflate(40, 24)
            pygame.draw.rect(screen, (16, 15, 12), box, border_radius=12)
            pygame.draw.rect(screen, (214, 183, 123), box, 2, border_radius=12)
            screen.blit(label, label.get_rect(center=box.center))
