"""Per-driver lap progress and session records; independent of rendering."""


class Race:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.laps_completed = 0
        self.best_lap_time = None
        self.last_lap_time = None
        self.reset_attempt()

    def reset_attempt(self):
        # Keep completed laps and records when a run is interrupted.
        self.next_checkpoint = 0
        self.lap_time = 0.0
        self.timer_running = False

    def advance_timer(self, dt, speed):
        if not self.timer_running and speed > 0:
            self.timer_running = True
        if self.timer_running:
            self.lap_time += dt

    def check_progress(self, track, previous, current):
        checkpoint = track.checkpoints[self.next_checkpoint]
        if not checkpoint.crossed(previous, current):
            return
        self.next_checkpoint += 1
        if self.next_checkpoint == len(track.checkpoints):
            self.laps_completed += 1
            self.next_checkpoint = 0
            self.last_lap_time = self.lap_time
            if self.best_lap_time is None or self.lap_time < self.best_lap_time:
                self.best_lap_time = self.lap_time
            self.lap_time = 0.0
            self.timer_running = True
        elif self.verbose:
            print(f"Checkpoint {self.next_checkpoint}/{len(track.checkpoints)} reached")
