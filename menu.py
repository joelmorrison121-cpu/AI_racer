"""Start menu: Freeplay or Train the AI, a sliding track picker and a
generation-cap picker. Run through `python main.py` (no options).

Keyboard: arrows to move, Enter to choose, Esc to go back. The mouse works too.
"""
import math
import random

import pygame

from camera import Camera
from evolution import create_run_directory
from settings import FPS
from track import make_track

# Bloons-style difficulty tabs, easiest first.
TRACKS = [
    ('easy', 'BEGINNER', 'A wide oval with gentle bends. A good first lap.'),
    ('classic', 'INTERMEDIATE', 'The original rectangle: four straights, four right angles.'),
    ('flowing', 'ADVANCED', 'Sweeping S-bends and diagonals. Smooth lines win.'),
    ('technical', 'EXPERT', 'Two hairpins, a chicane and a tight inner loop.'),
]
GENERATION_PRESETS = [10, 20, 30, 50, 100]
DEFAULT_GENERATIONS = 20
# Harder tracks need longer runs: in testing, first laps on technical came at
# generation 28 on average (18-35), so its default is 50.
RECOMMENDED_GENERATIONS = {'easy': 20, 'classic': 20, 'flowing': 30, 'technical': 50}

BG = (16, 15, 12)
PANEL = (26, 23, 18)
INK = (242, 236, 226)
MUTED = (170, 160, 145)
BRASS = (214, 183, 123)
BRASS_LIGHT = (238, 208, 154)
CARD_W, CARD_H = 560, 330


def font(size):
    return pygame.font.Font(None, size)


def ease(current, target, dt, rate=12.0):
    return current + (target - current) * (1 - math.exp(-rate * dt))


class Button:
    def __init__(self, label, center, size=(320, 62), text_size=46):
        self.label = label
        self.rect = pygame.Rect((0, 0), size)
        self.rect.center = center
        self.text_size = text_size
        self.glow = 0.0

    def hit(self, pos):
        return self.rect.collidepoint(pos)

    def draw(self, screen, selected, dt):
        self.glow = ease(self.glow, 1.0 if selected else 0.0, dt, 14)
        if self.glow > 0.02:
            glow = pygame.Surface(self.rect.inflate(24, 24).size, pygame.SRCALPHA)
            pygame.draw.rect(glow, (*BRASS, int(60 * self.glow)), glow.get_rect(), border_radius=16)
            screen.blit(glow, self.rect.inflate(24, 24).topleft)
        pygame.draw.rect(screen, PANEL, self.rect, border_radius=12)
        border = [int(MUTED[i] + (BRASS[i] - MUTED[i]) * self.glow) for i in range(3)]
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=12)
        colour = BRASS_LIGHT if selected else INK
        text = font(self.text_size).render(self.label, True, colour)
        screen.blit(text, text.get_rect(center=self.rect.center))


def track_thumbnail(track, size):
    """Whole-track picture scaled to fit `size`, on a dark card."""
    world = pygame.Surface(track.world_size)
    track.draw(world, -1, Camera(track.world_size, track.world_size))
    ratio = min(size[0] / track.world_size[0], size[1] / track.world_size[1])
    image = pygame.transform.smoothscale(
        world, (round(track.world_size[0] * ratio), round(track.world_size[1] * ratio)))
    card = pygame.Surface(size)
    card.fill(BG)
    card.blit(image, image.get_rect(center=card.get_rect().center))
    return card


def blurred_backdrop(image, size):
    small = pygame.transform.smoothscale(image, (max(1, size[0] // 10), max(1, size[1] // 10)))
    backdrop = pygame.transform.smoothscale(small, size)
    shade = pygame.Surface(size, pygame.SRCALPHA)
    shade.fill((10, 9, 7, 215))
    backdrop.blit(shade, (0, 0))
    return backdrop


def loading(screen, message):
    screen.fill(BG)
    text = font(40).render(message, True, MUTED)
    screen.blit(text, text.get_rect(center=screen.get_rect().center))
    pygame.display.flip()
    pygame.event.pump()


class Menu:
    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.w, self.h = screen.get_size()
        loading(screen, 'Loading tracks...')
        self.tracks = [make_track(name) for name, _, _ in TRACKS]
        self.thumbs = [track_thumbnail(t, (CARD_W, CARD_H - 70)) for t in self.tracks]
        self.backdrops = [blurred_backdrop(t, (self.w, self.h)) for t in self.thumbs]
        self.benchmarks = {}
        self.state = 'title'
        self.mode = None            # 'freeplay' or 'train'
        self.title_index = 0
        self.track_index = 0
        self.slide = 0.0            # animated card position
        self.gen_index = GENERATION_PRESETS.index(DEFAULT_GENERATIONS)
        cx = self.w // 2
        self.title_buttons = [Button('Freeplay', (cx, 360)), Button('Train the AI', (cx, 440)),
                              Button('Quit', (cx, 520))]
        self.back = Button('< Back', (90, 45), (130, 46), 32)
        self.start = Button('Start', (cx, 520), (260, 62))
        self.tab_rects = [pygame.Rect(0, 0, 200, 46) for _ in TRACKS]
        for i, rect in enumerate(self.tab_rects):
            rect.center = (cx + (i - 1.5) * 215, self.h - 60)
        self.arrow_left = pygame.Rect(0, 0, 60, 90)
        self.arrow_left.center = (cx - CARD_W // 2 - 70, 330)
        self.arrow_right = pygame.Rect(0, 0, 60, 90)
        self.arrow_right.center = (cx + CARD_W // 2 + 70, 330)
        self.chip_rects = [pygame.Rect(0, 0, 110, 70) for _ in GENERATION_PRESETS]
        for i, rect in enumerate(self.chip_rects):
            rect.center = (cx + (i - 2) * 130, 380)
        self.card_rect = pygame.Rect(0, 0, CARD_W, CARD_H)
        self.card_rect.center = (cx, 330)

    # ----- champion lap to show on each card --------------------------------
    def benchmark(self, index):
        name = TRACKS[index][0]
        if name not in self.benchmarks:
            from main import champion_path, champion_benchmark, load_champion
            loading_text = font(30).render('Timing the champion...', True, MUTED)
            self.screen.blit(loading_text, loading_text.get_rect(center=(self.w // 2, self.h - 115)))
            pygame.display.flip()
            network = load_champion(champion_path(name), name)
            self.benchmarks[name] = champion_benchmark(network, self.tracks[index])
        return self.benchmarks[name]

    # ----- actions ------------------------------------------------------------
    def choose_title(self, index):
        if index == 2:
            return 'quit'
        self.mode = 'freeplay' if index == 0 else 'train'
        self.state = 'tracks'
        return None

    def choose_track(self):
        if self.mode == 'train':
            recommended = RECOMMENDED_GENERATIONS.get(TRACKS[self.track_index][0], DEFAULT_GENERATIONS)
            self.gen_index = GENERATION_PRESETS.index(recommended)
            self.state = 'generations'
            return None
        from main import champion_path, load_champion, run_freeplay
        name = TRACKS[self.track_index][0]
        path = champion_path(name)
        loading(self.screen, 'Starting freeplay...')
        result = run_freeplay(self.screen, self.clock, self.tracks[self.track_index],
                              load_champion(path, name), champion_label=path.stem)
        return 'quit' if result == 'quit' else None

    def start_training(self):
        from live_train import run_training
        from training import TrainingSession
        loading(self.screen, 'Building the first generation...')
        session = TrainingSession(self.tracks[self.track_index], create_run_directory(),
                                  seed=random.randrange(1_000_000),
                                  generations=GENERATION_PRESETS[self.gen_index])
        result = run_training(self.screen, self.clock, session)
        self.state = 'tracks'
        return 'quit' if result == 'quit' else None

    def go_back(self):
        self.state = {'generations': 'tracks', 'tracks': 'title'}.get(self.state, 'title')

    def move_track(self, step):
        self.track_index = max(0, min(len(TRACKS) - 1, self.track_index + step))

    # ----- input --------------------------------------------------------------
    def handle(self, event):
        if event.type == pygame.QUIT:
            return 'quit'
        if event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_ESCAPE:
                if self.state == 'title':
                    return 'quit'
                self.go_back()
            elif self.state == 'title':
                if key in (pygame.K_UP, pygame.K_w):
                    self.title_index = (self.title_index - 1) % 3
                elif key in (pygame.K_DOWN, pygame.K_s):
                    self.title_index = (self.title_index + 1) % 3
                elif key in (pygame.K_RETURN, pygame.K_SPACE):
                    return self.choose_title(self.title_index)
            elif self.state == 'tracks':
                if key in (pygame.K_LEFT, pygame.K_a):
                    self.move_track(-1)
                elif key in (pygame.K_RIGHT, pygame.K_d):
                    self.move_track(1)
                elif key in (pygame.K_RETURN, pygame.K_SPACE):
                    return self.choose_track()
            elif self.state == 'generations':
                if key in (pygame.K_LEFT, pygame.K_a):
                    self.gen_index = max(0, self.gen_index - 1)
                elif key in (pygame.K_RIGHT, pygame.K_d):
                    self.gen_index = min(len(GENERATION_PRESETS) - 1, self.gen_index + 1)
                elif key in (pygame.K_RETURN, pygame.K_SPACE):
                    return self.start_training()
        if event.type == pygame.MOUSEMOTION and self.state == 'title':
            for i, button in enumerate(self.title_buttons):
                if button.hit(event.pos):
                    self.title_index = i
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            if self.state != 'title' and self.back.hit(pos):
                self.go_back()
            elif self.state == 'title':
                for i, button in enumerate(self.title_buttons):
                    if button.hit(pos):
                        return self.choose_title(i)
            elif self.state == 'tracks':
                if self.arrow_left.collidepoint(pos):
                    self.move_track(-1)
                elif self.arrow_right.collidepoint(pos):
                    self.move_track(1)
                elif self.card_rect.collidepoint(pos):
                    return self.choose_track()
                for i, rect in enumerate(self.tab_rects):
                    if rect.collidepoint(pos):
                        self.track_index = i
            elif self.state == 'generations':
                for i, rect in enumerate(self.chip_rects):
                    if rect.collidepoint(pos):
                        self.gen_index = i
                if self.start.hit(pos):
                    return self.start_training()
        if event.type == pygame.MOUSEWHEEL and self.state == 'tracks':
            self.move_track(-1 if event.y > 0 else 1)
        return None

    # ----- drawing ------------------------------------------------------------
    def draw_heading(self, title, subtitle=None, y=110):
        text = font(64).render(title, True, INK)
        self.screen.blit(text, text.get_rect(center=(self.w // 2, y)))
        if subtitle:
            sub = font(30).render(subtitle, True, MUTED)
            self.screen.blit(sub, sub.get_rect(center=(self.w // 2, y + 44)))

    def draw_title(self, dt, t):
        self.screen.blit(self.backdrops[int(t / 4) % len(TRACKS)], (0, 0))
        title = font(130).render('CRASH COURSE', True, BRASS_LIGHT)
        shadow = font(130).render('CRASH COURSE', True, (60, 48, 30))
        rect = title.get_rect(center=(self.w // 2, 165 + 4 * math.sin(t * 1.4)))
        self.screen.blit(shadow, rect.move(4, 5))
        self.screen.blit(title, rect)
        tag = font(32).render('Neural-network drivers that learn to race', True, MUTED)
        self.screen.blit(tag, tag.get_rect(center=(self.w // 2, 245)))
        for i, button in enumerate(self.title_buttons):
            button.draw(self.screen, i == self.title_index, dt)
        hint = font(24).render('Arrows + Enter, or click.  Esc quits.', True, MUTED)
        self.screen.blit(hint, hint.get_rect(center=(self.w // 2, self.h - 30)))

    def draw_card(self, index, center_x, scale, alpha):
        name, level, blurb = TRACKS[index]
        card = pygame.Surface((CARD_W, CARD_H), pygame.SRCALPHA)
        pygame.draw.rect(card, PANEL, card.get_rect(), border_radius=16)
        card.blit(self.thumbs[index], (0, 0))
        pygame.draw.rect(card, PANEL, (0, CARD_H - 70, CARD_W, 70))
        card.blit(font(44).render(name.capitalize(), True, INK), (20, CARD_H - 62))
        card.blit(font(24).render(blurb, True, MUTED), (20, CARD_H - 26))
        badge = font(24).render(level, True, BG)
        badge_rect = badge.get_rect().inflate(18, 10)
        badge_rect.topright = (CARD_W - 14, 14)
        pygame.draw.rect(card, BRASS, badge_rect, border_radius=8)
        card.blit(badge, badge.get_rect(center=badge_rect.center))
        pygame.draw.rect(card, BRASS if scale > 0.95 else MUTED, card.get_rect(), 2, border_radius=16)
        if scale != 1:
            card = pygame.transform.smoothscale(card, (round(CARD_W * scale), round(CARD_H * scale)))
        if alpha < 255:   # dim side cards (per-pixel alpha surfaces ignore set_alpha)
            shade = pygame.Surface(card.get_size(), pygame.SRCALPHA)
            shade.fill((10, 9, 7, 255 - alpha))
            card.blit(shade, (0, 0))
        self.screen.blit(card, card.get_rect(center=(center_x, 330)))

    def draw_tracks(self, dt):
        self.slide = ease(self.slide, self.track_index, dt, 10)
        self.screen.blit(self.backdrops[self.track_index], (0, 0))
        heading = 'Freeplay' if self.mode == 'freeplay' else 'Train the AI'
        hint = ('Pick a track, then race it. Tab swaps in the champion.'
                if self.mode == 'freeplay' else 'Pick a track for a new population to learn.')
        self.draw_heading(heading, hint, 80)
        # Neighbouring cards first (smaller, dimmer), then the current one on top.
        order = sorted(range(len(TRACKS)), key=lambda i: -abs(i - self.slide))
        for i in order:
            offset = i - self.slide
            if abs(offset) > 1.6:
                continue
            scale = 1 - 0.28 * min(abs(offset), 1)
            alpha = int(255 * (1 - 0.65 * min(abs(offset), 1)))
            self.draw_card(i, self.w // 2 + offset * (CARD_W * 0.88), scale, alpha)
        for rect, direction, enabled in ((self.arrow_left, -1, self.track_index > 0),
                                         (self.arrow_right, 1, self.track_index < len(TRACKS) - 1)):
            if not enabled:
                continue
            cx, cy = rect.center
            points = [(cx - 18 * direction, cy - 34), (cx + 22 * direction, cy), (cx - 18 * direction, cy + 34)]
            pygame.draw.polygon(self.screen, BRASS, points)
            pygame.draw.polygon(self.screen, (90, 70, 40), points, 3)
        if self.mode == 'freeplay':
            bench = self.benchmark(self.track_index)
            info = (f"Champion lap to beat: {bench['lap']:.2f}s" if bench['finished']
                    else f"Champion can't finish this one yet ({bench['percent']}% of a lap)")
        else:
            info = 'Press Enter or click the card to choose'
        text = font(30).render(info, True, BRASS_LIGHT)
        self.screen.blit(text, text.get_rect(center=(self.w // 2, 530)))
        for i, rect in enumerate(self.tab_rects):
            active = i == self.track_index
            pygame.draw.rect(self.screen, BRASS if active else PANEL, rect, border_radius=23)
            pygame.draw.rect(self.screen, BRASS, rect, 2, border_radius=23)
            label = font(28).render(TRACKS[i][1], True, BG if active else INK)
            self.screen.blit(label, label.get_rect(center=rect.center))
        self.back.draw(self.screen, False, dt)

    def draw_generations(self, dt):
        self.screen.blit(self.backdrops[self.track_index], (0, 0))
        name = TRACKS[self.track_index][0]
        self.draw_heading('How many generations?',
                          f'{name.capitalize()} track · 30 drivers per generation', 150)
        for i, rect in enumerate(self.chip_rects):
            active = i == self.gen_index
            pygame.draw.rect(self.screen, BRASS if active else PANEL, rect, border_radius=14)
            pygame.draw.rect(self.screen, BRASS, rect, 2, border_radius=14)
            label = font(48).render(str(GENERATION_PRESETS[i]), True, BG if active else INK)
            self.screen.blit(label, label.get_rect(center=rect.center))
        recommended = RECOMMENDED_GENERATIONS.get(name, DEFAULT_GENERATIONS)
        note = font(26).render(f'Recommended for this track: {recommended}. More generations = '
                               'longer to watch, more chance to learn.', True, MUTED)
        self.screen.blit(note, note.get_rect(center=(self.w // 2, 455)))
        self.start.draw(self.screen, True, dt)
        tip = font(24).render('While training: Space pause · 1/2/3 speed · F follow · G graph · Esc menu',
                              True, MUTED)
        self.screen.blit(tip, tip.get_rect(center=(self.w // 2, self.h - 40)))
        self.back.draw(self.screen, False, dt)

    def run(self, smoke_frames=None):
        t = 0.0
        frames = 0
        while True:
            dt = min(self.clock.tick(FPS) / 1000, 0.05)
            t += dt
            for event in pygame.event.get():
                if self.handle(event) == 'quit':
                    return
            if self.state == 'title':
                self.draw_title(dt, t)
            elif self.state == 'tracks':
                self.draw_tracks(dt)
            else:
                self.draw_generations(dt)
            pygame.display.flip()
            frames += 1
            if smoke_frames and frames >= smoke_frames:
                return


def run_menu(screen, clock, smoke_frames=None):
    Menu(screen, clock).run(smoke_frames)
