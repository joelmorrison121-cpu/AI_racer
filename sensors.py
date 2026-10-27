import pygame


class Sensors:
    def __init__(self, max_distance=350, step=4):
        self.max_distance = max_distance
        self.step = step

        # Relative to the car: left → ahead → right.
        self.angles = (90, 60, 30, 0, -30, -60, -90)

        self.distances = []
        self.endpoints = []

    def update(self, car, track):
        self.distances = []
        self.endpoints = []

        for relative_angle in self.angles:
            ray_angle = car.angle + relative_angle
            direction = pygame.Vector2(1, 0).rotate(-ray_angle)

            distance, endpoint = self.cast_ray(
                car.position, direction, track
            )

            self.distances.append(distance)
            self.endpoints.append(endpoint)

    def cast_ray(self, origin, direction, track):
        distance = 0

        while True:
            point = origin + direction * distance

            # Check bounds before accessing a pixel in the mask.
            inside_world = (
                0 <= point.x < track.world_size[0]
                and 0 <= point.y < track.world_size[1]
            )

            if not inside_world:
                return distance, point

            pixel = (int(point.x), int(point.y))

            # A set mask pixel represents grass/an obstacle.
            if track.obstacle_mask.get_at(pixel):
                return distance, point

            if distance >= self.max_distance:
                return self.max_distance, point

            distance = min(distance + self.step, self.max_distance)

    def get_readings(self):
        # Convert distances into values between 0 and 1.
        return [
            distance / self.max_distance
            for distance in self.distances
        ]

    def draw(self, screen, camera, car):
        start = camera.apply_point(car.position)

        for distance, endpoint in zip(
            self.distances, self.endpoints
        ):
            end = camera.apply_point(endpoint)

            if distance < self.max_distance:
                colour = (255, 100, 100)  # Obstacle detected
            else:
                colour = (100, 220, 255)  # Clear to sensor limit

            pygame.draw.line(screen, colour, start, end, 1)
            pygame.draw.circle(
                screen,
                colour,
                (round(end.x), round(end.y)),
                3,
            )