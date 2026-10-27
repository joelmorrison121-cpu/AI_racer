"""Training progress graph: best and average fitness plus lap-completion rate.

Drawn with pygame only (no extra dependency). Used by the live viewer (G key)
and saved as progress.png in each run folder after every generation.

    python progress_graph.py training_runs/<run>     # re-render a run's graph
"""
import json
import sys
from pathlib import Path

import pygame

BG = (16, 22, 30)
GRID = (44, 54, 66)
TEXT = (225, 233, 242)
BEST = (255, 200, 80)
AVERAGE = (110, 190, 255)
COMPLETION = (120, 220, 140)


def _nice_ceiling(value):
    if value <= 0:
        return 1.0
    magnitude = 10 ** (len(str(int(value))) - 1)
    for step in (1, 2, 2.5, 5, 10):
        if step * magnitude >= value:
            return step * magnitude
    return 10 * magnitude


def draw_progress_graph(surface, rect, history, font, title='Training progress',
                        series=None):
    """Draw onto `surface` inside `rect`.

    `history` is a list of dicts with generation, best, average and completion_rate.
    `series` optionally overrides lines: [(label, colour, values, axis)] where axis
    is 'fitness' or 'percent'.
    """
    rect = pygame.Rect(rect)
    pygame.draw.rect(surface, BG, rect, border_radius=8)
    pad_left, pad_right, pad_top, pad_bottom = 64, 58, 40, 46
    plot = pygame.Rect(rect.x + pad_left, rect.y + pad_top,
                       rect.w - pad_left - pad_right, rect.h - pad_top - pad_bottom)
    surface.blit(font.render(title, True, TEXT), (rect.x + 14, rect.y + 10))
    if series is None:
        series = [
            ('Best fitness', BEST, [h['best'] for h in history], 'fitness'),
            ('Average fitness', AVERAGE, [h['average'] for h in history], 'fitness'),
            ('Lap completion %', COMPLETION,
             [100 * h.get('completion_rate', 0) for h in history], 'percent'),
        ]
    count = max((len(values) for _, _, values, _ in series), default=0)
    fitness_values = [v for _, _, values, axis in series if axis == 'fitness' for v in values]
    top = _nice_ceiling(max(fitness_values, default=1))

    for i in range(5):
        y = plot.bottom - plot.h * i / 4
        pygame.draw.line(surface, GRID, (plot.left, y), (plot.right, y), 1)
        left = font.render(f'{top * i / 4:.0f}', True, TEXT)
        surface.blit(left, (plot.left - left.get_width() - 8, y - left.get_height() / 2))
        right = font.render(f'{25 * i}%', True, COMPLETION)
        surface.blit(right, (plot.right + 8, y - right.get_height() / 2))
    pygame.draw.rect(surface, GRID, plot, 1)

    def point(index, value, axis):
        x = plot.left + (plot.w * index / (count - 1) if count > 1 else plot.w / 2)
        scale = 100 if axis == 'percent' else top
        return x, plot.bottom - plot.h * min(max(value / scale, 0), 1)

    if count:
        step = max(1, round(count / 10))
        for index in range(0, count, step):
            label = font.render(str(index + 1), True, TEXT)
            x, _ = point(index, 0, 'fitness')
            surface.blit(label, (x - label.get_width() / 2, plot.bottom + 6))
    caption = font.render('Generation', True, TEXT)
    surface.blit(caption, (plot.centerx - caption.get_width() / 2, plot.bottom + 24))

    legend_x = rect.right - 14
    for label, colour, values, axis in reversed(series):
        text = font.render(label, True, colour)
        legend_x -= text.get_width()
        surface.blit(text, (legend_x, rect.y + 10))
        legend_x -= 22
        points = [point(i, v, axis) for i, v in enumerate(values)]
        if len(points) > 1:
            pygame.draw.lines(surface, colour, False, points, 2)
        for p in points[-1:]:
            pygame.draw.circle(surface, colour, (round(p[0]), round(p[1])), 4)


def save_progress_png(history, path, size=(900, 480), title='Training progress'):
    surface = pygame.Surface(size)
    font = pygame.font.Font(None, 22)
    draw_progress_graph(surface, surface.get_rect(), history, font, title)
    pygame.image.save(surface, str(path))


def main():
    if len(sys.argv) != 2:
        print('Usage: python progress_graph.py training_runs/<run folder>')
        return 1
    folder = Path(sys.argv[1])
    history = json.loads((folder / 'history.json').read_text(encoding='utf-8'))
    pygame.init()
    try:
        save_progress_png(history, folder / 'progress.png', title=f'Training progress: {folder.name}')
    finally:
        pygame.quit()
    print(f'Saved {folder / "progress.png"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
