"""Tracks: geometry, collision mask, checkpoints and progress along the route.

Two kinds of track share one interface:

* ``Track`` / ``ClassicTrack`` - the original v1.0 rectangle (outer/inner rects).
  Kept bit-for-bit so the shipped champion and old results still reproduce.
* ``PathTrack`` - defined by a closed centre-line path plus a road width.
  The path is smoothed with a Catmull-Rom spline, so tracks can have curves,
  diagonals and hairpins. Checkpoints are generated perpendicular to the path,
  so they can sit at any angle.

Every track offers ``progress_along(position, hint)``: how far along the centre
line a position is (in pixels). Fitness uses this so drivers are rewarded for
following the route through bends rather than for straight-line closeness.
"""
import math

import pygame
from settings import GRASS_COLOUR, ROAD_COLOUR

KERB_COLOUR = (215, 215, 220)
CENTRE_LINE_COLOUR = (95, 95, 102)


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
class Checkpoint:
    """A directional gate: a line segment from ``start`` to ``end``.

    It counts only when a car crosses it travelling in the ``forward`` direction
    (forward must be perpendicular to the segment). Works at any angle.
    """

    def __init__(self, start, end, forward, centre=None):
        self.start = pygame.Vector2(start)
        self.end = pygame.Vector2(end)
        self.forward = pygame.Vector2(forward).normalize()
        self.centre = pygame.Vector2(centre) if centre is not None else (self.start + self.end) / 2
        along = self.end - self.start
        self.length = along.length()
        self.direction = along / self.length
        left, right = sorted((self.start.x, self.end.x))
        top, bottom = sorted((self.start.y, self.end.y))
        # Bounding box, kept for code and tests that use checkpoint.rect.center.
        self.rect = pygame.Rect(round(left), round(top),
                                max(1, round(right - left)), max(1, round(bottom - top)))

    @classmethod
    def from_rect(cls, rect, forward):
        """The v1.0 format: a thin rectangle plus a travel direction."""
        rect = pygame.Rect(rect)
        forward = pygame.Vector2(forward)
        if forward.x != 0:
            start, end = (rect.centerx, rect.top), (rect.centerx, rect.bottom)
        else:
            start, end = (rect.left, rect.centery), (rect.right, rect.centery)
        gate = cls(start, end, forward, centre=rect.center)
        gate.rect = rect
        return gate

    def crossed(self, previous, current):
        previous_side = (previous - self.centre).dot(self.forward)
        current_side = (current - self.centre).dot(self.forward)
        if not (previous_side < 0 <= current_side):
            return False
        fraction = -previous_side / (current_side - previous_side)
        crossing = previous.lerp(current, fraction)
        along = (crossing - self.start).dot(self.direction)
        return 0 <= along < self.length

    def draw(self, screen, active, camera):
        colour = (255, 220, 70) if active else (110, 110, 120)
        pygame.draw.line(screen, colour, camera.apply_point(self.start),
                         camera.apply_point(self.end), 3)


# --------------------------------------------------------------------------
# Shared centre-line maths
# --------------------------------------------------------------------------
class CentreLine:
    """A closed polyline with cumulative arc length, for measuring progress."""

    def __init__(self, points):
        self.points = [pygame.Vector2(p) for p in points]
        self.count = len(self.points)
        self.cumulative = [0.0]
        for i in range(self.count):
            a, b = self.points[i], self.points[(i + 1) % self.count]
            self.cumulative.append(self.cumulative[-1] + a.distance_to(b))
        self.length = self.cumulative[-1]

    def _project(self, position, i):
        a, b = self.points[i], self.points[(i + 1) % self.count]
        ab = b - a
        t = 0.0 if ab.length_squared() == 0 else max(0.0, min(1.0, (position - a).dot(ab) / ab.length_squared()))
        closest = a + ab * t
        return position.distance_squared_to(closest), self.cumulative[i] + t * ab.length()

    def locate(self, position, hint=None, window=24):
        """Return (arc_length, segment_index) of the nearest centre-line point.

        With a ``hint`` (the previous segment index) only nearby segments are
        searched, so a car on one side of a hairpin is never mistaken for being
        on the other side.
        """
        position = pygame.Vector2(position)
        indices = (range(self.count) if hint is None else
                   ((hint + k) % self.count for k in range(-window, window + 1)))
        best = None
        for i in indices:
            distance, s = self._project(position, i)
            if best is None or distance < best[0]:
                best = (distance, s, i)
        return best[1], best[2]

    def point_at(self, s):
        """Point and unit tangent at arc length s."""
        s %= self.length
        lo, hi = 0, self.count
        while lo < hi - 1:  # binary search for the segment
            mid = (lo + hi) // 2
            if self.cumulative[mid] <= s:
                lo = mid
            else:
                hi = mid
        a, b = self.points[lo], self.points[(lo + 1) % self.count]
        seg = self.cumulative[lo + 1] - self.cumulative[lo]
        t = 0 if seg == 0 else (s - self.cumulative[lo]) / seg
        tangent = (b - a)
        return a.lerp(b, t), tangent.normalize() if tangent.length_squared() else pygame.Vector2(1, 0)


class BaseTrack:
    name = 'base'
    description = ''
    time_limit = 30.0          # seconds an attempt may last on this track

    def collides(self, car):
        outside_world = not self.bounds.contains(car.rect)
        overlap = self.obstacle_mask.overlap(car.mask, car.rect.topleft)
        return outside_world or overlap is not None

    # Progress along the route ------------------------------------------------
    def gate_progress(self, index):
        """Arc length of checkpoint ``index`` along the centre line."""
        return self.gate_positions[index]

    def segment_start(self, next_checkpoint):
        """Arc length where the current segment (towards next_checkpoint) starts."""
        return self.start_progress if next_checkpoint == 0 else self.gate_positions[next_checkpoint - 1]

    def progress_fraction(self, s, next_checkpoint):
        """0..1 fraction of the way from the previous gate to the next one,
        measured along the centre line. Backwards driving scores 0."""
        length = self.centre_line.length
        begin = self.segment_start(next_checkpoint)
        span = (self.gate_positions[next_checkpoint] - begin) % length or length
        travelled = (s - begin) % length
        if travelled > span:  # behind the segment start, or beyond the gate
            travelled = 0.0 if travelled > (span + length) / 2 else span
        return travelled / span


# --------------------------------------------------------------------------
# v1.0 rectangle
# --------------------------------------------------------------------------
class Track(BaseTrack):
    """The original rectangular circuit (also available as ClassicTrack)."""
    name = 'classic'
    description = 'Original v1.0 rectangle: four straights and four right angles.'
    time_limit = 30.0

    def __init__(self):
        # The whole game world, independent of the visible window.
        self.world_size = (2000, 1400)
        self.bounds = pygame.Rect((0, 0), self.world_size)

        # Track geometry: x, y, width, height.
        self.outer = pygame.Rect(100, 100, 1800, 1200)
        self.inner = pygame.Rect(400, 400, 1200, 600)

        # Start on the top straight, facing right.
        self.start_position = (500, 250)
        self.start_angle = 0

        # Clockwise route: right → bottom → left → top.
        self.checkpoints = [
            Checkpoint.from_rect((1600, 680, 300, 40), (0, 1)),
            Checkpoint.from_rect((980, 1000, 40, 300), (-1, 0)),
            Checkpoint.from_rect((100, 680, 300, 40), (0, -1)),
            Checkpoint.from_rect((480, 100, 40, 300), (1, 0)),
        ]

        self.obstacle_mask = self.build_collision_mask()

        # Centre line of the road, clockwise from the start (used for progress).
        self.centre_line = CentreLine([(500, 250), (1750, 250), (1750, 1150),
                                       (250, 1150), (250, 250)])
        self.start_progress = 0.0
        self.gate_positions = [self.centre_line.locate(c.centre)[0] for c in self.checkpoints]
        if self.gate_positions[-1] < 1:     # the finish gate sits at the start: s = full lap
            self.gate_positions[-1] = self.centre_line.length

    def build_collision_mask(self):
        surface = pygame.Surface(self.world_size, pygame.SRCALPHA)
        surface.fill((255, 255, 255, 255))
        pygame.draw.rect(surface, (0, 0, 0, 0), self.outer)
        pygame.draw.rect(surface, (255, 255, 255, 255), self.inner)
        return pygame.mask.from_surface(surface)

    def draw(self, screen, next_checkpoint, camera):
        screen.fill(GRASS_COLOUR)
        pygame.draw.rect(screen, ROAD_COLOUR, camera.apply_rect(self.outer))
        pygame.draw.rect(screen, GRASS_COLOUR, camera.apply_rect(self.inner))
        for index, checkpoint in enumerate(self.checkpoints):
            checkpoint.draw(screen, index == next_checkpoint, camera)


ClassicTrack = Track


# --------------------------------------------------------------------------
# Path-based tracks
# --------------------------------------------------------------------------
def catmull_rom_loop(control_points, spacing=6.0):
    """Smooth closed curve through every control point, sampled every ~spacing px."""
    pts = [pygame.Vector2(p) for p in control_points]
    n = len(pts)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        steps = max(2, int(p1.distance_to(p2) / spacing))
        for k in range(steps):
            t = k / steps
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return out


class PathTrack(BaseTrack):
    """A circuit defined by a closed centre-line path and a road width."""

    def __init__(self, name, control_points, road_width, checkpoints=12,
                 time_limit=45.0, description='', margin=160, reverse=False):
        self.name = name
        self.description = description
        self.time_limit = time_limit
        self.road_width = road_width
        points = list(reversed(control_points)) if reverse else list(control_points)
        path = catmull_rom_loop(points)

        # Shift the world so the whole road sits inside with a grass margin.
        min_x = min(p.x for p in path) - road_width / 2 - margin
        min_y = min(p.y for p in path) - road_width / 2 - margin
        path = [p - pygame.Vector2(min_x, min_y) for p in path]
        max_x = max(p.x for p in path) + road_width / 2 + margin
        max_y = max(p.y for p in path) + road_width / 2 + margin
        self.world_size = (math.ceil(max_x), math.ceil(max_y))
        self.bounds = pygame.Rect((0, 0), self.world_size)
        self.centre_line = CentreLine(path)

        # Start on the centre line at s = 0, facing along the path.
        start, tangent = self.centre_line.point_at(0)
        self.start_progress = 0.0
        self.start_position = (start.x, start.y)
        self.start_angle = -math.degrees(math.atan2(tangent.y, tangent.x))

        # Evenly spaced gates; the last one is the start/finish line itself,
        # placed just behind the start so a lap needs the whole circuit.
        length = self.centre_line.length
        self.gate_positions = [length * (i + 1) / checkpoints for i in range(checkpoints)]
        self.gate_positions[-1] = length - 30.0
        self.checkpoints = [self._gate(s) for s in self.gate_positions]

        self.obstacle_mask, self.surface = self._build()

    def _gate(self, s):
        centre, tangent = self.centre_line.point_at(s)
        normal = pygame.Vector2(-tangent.y, tangent.x)
        half = self.road_width / 2
        return Checkpoint(centre - normal * half, centre + normal * half, tangent, centre)

    def _paint_road(self, surface, colour, width):
        radius = width / 2
        pts = self.centre_line.points
        for i, p in enumerate(pts):
            q = pts[(i + 1) % len(pts)]
            d = q - p
            if d.length_squared() == 0:
                continue
            n = pygame.Vector2(-d.y, d.x).normalize() * radius
            pygame.draw.polygon(surface, colour, [p + n, q + n, q - n, p - n])
            pygame.draw.circle(surface, colour, p, radius)

    def _build(self):
        # Collision: everything except the road is an obstacle.
        mask_surface = pygame.Surface(self.world_size, pygame.SRCALPHA)
        mask_surface.fill((255, 255, 255, 255))
        self._paint_road(mask_surface, (0, 0, 0, 0), self.road_width)
        mask = pygame.mask.from_surface(mask_surface)

        # Pre-rendered picture: grass, kerb edge, road, dashed centre line.
        surface = pygame.Surface(self.world_size)
        surface.fill(GRASS_COLOUR)
        self._paint_road(surface, KERB_COLOUR, self.road_width + 8)
        self._paint_road(surface, ROAD_COLOUR, self.road_width)
        pts = self.centre_line.points
        for i in range(0, len(pts), 8):
            pygame.draw.line(surface, CENTRE_LINE_COLOUR, pts[i], pts[(i + 4) % len(pts)], 2)
        start, tangent = self.centre_line.point_at(0)
        normal = pygame.Vector2(-tangent.y, tangent.x) * (self.road_width / 2)
        for k in range(10):  # chequered start line
            a = start - normal + normal * (2 * k / 10)
            b = start - normal + normal * (2 * (k + 1) / 10)
            pygame.draw.line(surface, (240, 240, 240) if k % 2 else (30, 30, 30), a, b, 6)
        return mask, surface

    def draw(self, screen, next_checkpoint, camera):
        screen.fill(GRASS_COLOUR)
        screen.blit(self.surface, (-round(camera.position.x), -round(camera.position.y)))
        for index, checkpoint in enumerate(self.checkpoints):
            checkpoint.draw(screen, index == next_checkpoint, camera)


# --------------------------------------------------------------------------
# Track library
# --------------------------------------------------------------------------
TRACK_DEFINITIONS = {
    'easy': dict(
        description='Wide rounded oval with gentle, constant-radius bends.',
        road_width=280, checkpoints=10, time_limit=40.0,
        control_points=[(400, 300), (1200, 260), (2000, 300), (2350, 700), (2000, 1100),
                        (1200, 1140), (400, 1100), (60, 700)]),
    'flowing': dict(
        description='Sweeping S-bends and a long diagonal; rewards smooth lines.',
        road_width=230, checkpoints=14, time_limit=50.0,
        control_points=[(300, 300), (1000, 220), (1500, 520), (2100, 300), (2600, 560),
                        (2550, 1150), (1950, 1350), (1450, 1050), (900, 1400), (350, 1250),
                        (120, 800)]),
    'technical': dict(
        description='Two hairpins, a chicane and a tight inner loop.',
        road_width=200, checkpoints=18, time_limit=60.0,
        control_points=[(300, 250), (1500, 250), (2300, 250), (2550, 520), (2300, 780),
                        (1600, 780), (1400, 980), (1600, 1180), (2350, 1200), (2550, 1450),
                        (2300, 1700), (1250, 1700), (950, 1480), (650, 1700), (250, 1650),
                        (100, 1250), (420, 1000), (180, 720), (100, 450)]),
}

TRACK_NAMES = ['classic'] + list(TRACK_DEFINITIONS)
_cache = {}


def make_track(name='classic'):
    """Build (and cache) a track by name: classic, easy, flowing or technical."""
    if name not in TRACK_NAMES:
        raise ValueError(f"Unknown track '{name}'. Choose from: {', '.join(TRACK_NAMES)}")
    if name not in _cache:
        if name == 'classic':
            _cache[name] = Track()
        else:
            _cache[name] = PathTrack(name, **TRACK_DEFINITIONS[name])
    return _cache[name]


def add_track_argument(parser, default='classic'):
    parser.add_argument('--track', choices=TRACK_NAMES, default=default,
                        help=f'Which circuit to use (default {default}).')
    return parser
