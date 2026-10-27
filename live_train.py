"""Watch evolution live.

    python live_train.py                         # 20 generations, seed 42
    python live_train.py --generations 40
    python live_train.py --resume training_runs/<run>
    python live_train.py --track technical
Controls are listed on the dashboard and in the README.
"""
import argparse
import pygame
from camera import Camera
from settings import WIDTH, HEIGHT, FPS, SIMULATION_DT, MAX_FRAME_TIME
from progress_graph import draw_progress_graph
from training import (TrainingSession, add_training_arguments,  # noqa: F401 (re-export)
                      validate_training_arguments, create_session,
                      track_for)


def draw_world(surface, camera, session, font, show_sensors):
    leader = session.leader
    session.track.draw(surface, leader.race.next_checkpoint, camera)
    if show_sensors:
        leader.sensors.update(leader.car, session.track)
        leader.sensors.draw(surface, camera, leader.car)
    # Dead cars behind live ones, leader on top. Cars never collide with each other.
    ordered = sorted(session.agents, key=lambda a: a.active)
    for agent in ordered:
        if agent is not leader:
            agent.draw(surface, camera)
    leader.draw(surface, camera, leader=True, font=font)


def draw_dashboard(screen, session, font, paused, speed, follow):
    top = HEIGHT - 155
    pygame.draw.rect(screen, (16, 22, 30), (0, top, WIDTH, 155))
    alive = sum(a.active for a in session.agents)
    completed = sum(a.status == 'completed lap' for a in session.agents)
    generation_best = max(a.fitness for a in session.agents)
    best = max(session.best_ever, generation_best)
    lap = '--' if session.best_lap is None else f'{session.best_lap:.2f}s'
    state = 'FINISHED' if session.finished else ('PAUSED' if paused else 'TRAINING')
    if session.generation_done and not session.finished:
        state = 'GENERATION COMPLETE'
    view = 'Follow leader' if follow else 'Whole track'
    leader = session.leader
    lines = [
        f'GENERATION {session.generation}/{session.generations}    {state}    Target speed: {speed}x',
        f'Active: {alive}/{len(session.agents)}    Finished laps: {completed}    '
        f'Best fitness: {best:.1f}    Best finished-gen lap: {lap}',
        f'Leader #{leader.number}: {leader.fitness:.1f} | {leader.status}    View: {view}',
        'Space: pause   1/2/3: 1x/2x/4x   F: camera   S: leader sensors   G: graph   Esc: quit',
        'Enter: next generation when complete. Champions save in this run\'s folder.',
    ]
    for i, text in enumerate(lines):
        screen.blit(font.render(text, True, (225, 233, 242)), (16, top + 12 + i * 27))


def main(args, smoke_frames=None):
    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('AI Racer | Live evolution')
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)
        track = track_for(args)
        session = create_session(args, track)
        camera = Camera((WIDTH, HEIGHT - 155), track.world_size)
        overview_camera = Camera(track.world_size, track.world_size)
        world_surface = pygame.Surface(track.world_size)
        ratio = min(WIDTH / track.world_size[0], (HEIGHT - 155) / track.world_size[1])
        overview_size = tuple(round(d * ratio) for d in track.world_size)
        overview_position = ((WIDTH - overview_size[0]) // 2,
                             (HEIGHT - 155 - overview_size[1]) // 2)
        paused = False
        follow = False
        show_sensors = False
        show_graph = False
        graph_font = pygame.font.Font(None, 22)
        speed = 1
        accumulator = 0.0
        generation_hold = 0.0
        running = True
        frames = 0
        while running:
            frame_time = min(clock.tick(FPS) / 1000, MAX_FRAME_TIME)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        accumulator = 0.0
                    elif event.key == pygame.K_f:
                        follow = not follow
                    elif event.key == pygame.K_s:
                        show_sensors = not show_sensors
                    elif event.key == pygame.K_g:
                        show_graph = not show_graph
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        speed = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 4}[event.key]
                    elif event.key == pygame.K_RETURN and session.generation_done:
                        session.next_generation()
                        generation_hold = 0.0
                        accumulator = 0.0
            if not running:
                break
            if not paused and not session.finished:
                if session.generation_done:
                    generation_hold += frame_time
                    if generation_hold >= 1.5:
                        session.next_generation()
                        generation_hold = 0.0
                        accumulator = 0.0
                else:
                    # Bound catch-up work so input remains responsive on slower machines.
                    accumulator = min(accumulator + frame_time * speed, SIMULATION_DT * 8)
                    while accumulator >= SIMULATION_DT and not session.generation_done:
                        session.step()
                        accumulator -= SIMULATION_DT
                    if session.generation_done:
                        accumulator = 0.0
                        generation_hold = 0.0
            if follow:
                camera.update(session.leader.car.position)
                draw_world(screen, camera, session, font, show_sensors)
            else:
                draw_world(world_surface, overview_camera, session, font, show_sensors)
                screen.fill((11, 16, 21))
                overview = pygame.transform.smoothscale(world_surface, overview_size)
                screen.blit(overview, overview_position)
            if show_graph:
                draw_progress_graph(screen, (40, 30, WIDTH - 80, HEIGHT - 155 - 60),
                                    session.history, graph_font,
                                    'Training progress (G to hide)')
            draw_dashboard(screen, session, font, paused, speed, follow)
            pygame.display.flip()
            frames += 1
            if smoke_frames is not None and frames >= smoke_frames:
                running = False
    finally:
        pygame.quit()


if __name__ == '__main__':
    parser = add_training_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--smoke-frames', type=int, default=None,
                        help='Exit after a few frames for an automated startup check.')
    arguments = parser.parse_args()
    validate_training_arguments(parser, arguments)
    main(arguments, arguments.smoke_frames)
