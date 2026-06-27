import time
from typing import Literal

import matplotlib
import numpy as np
import pygame
from environment import RevenueManagementEnv
from matplotlib.colors import Normalize

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
LIGHT_GRAY = (200, 200, 200)


class OptimisationVisualiser:
    def __init__(
        self,
        env: RevenueManagementEnv,
        model,
        day_xs: list[int],
        bucket_names: list[str],
        mode: Literal["online", "replay"] = "replay",
    ):
        self.env = env
        self.model = model
        self.day_xs = day_xs
        self.bucket_names = bucket_names
        self.mode = mode

        self.total_frames = 0

        self.history = {
            "demand_matrix": [],
            "availabilities": [],
            "revenue": [],
            "bookings": [],
            "rewards": [],
        }

        self._initialize_pygame()

    def run(self, deterministic: bool = True):
        if self.mode == "replay":
            self._collect_history(deterministic=deterministic)

        pygame.display.set_caption(f"Agent Decisions visualization (service_id: {self.env.service_id})")
        clock = pygame.time.Clock()
        while self.running:
            self._handle_events()
            if self.playing:
                self.current_frame = (self.current_frame + 1) % self.total_frames
            # Clear screen
            self.screen.fill(WHITE)
            self._draw_frame()
            pygame.display.flip()
            clock.tick(10)  # Limit to 10 FPS
        pygame.quit()

    def _initialize_pygame(self):
        """Initialize Pygame and set up the screen."""
        self.screen_width = 1200
        self.screen_height = 800
        self.slider_dragging = False
        self.running = True
        self.playing = True
        self.current_frame = 0

        self.left_key_held = False
        self.right_key_held = False

        self.grid_rows, self.grid_cols = self.env.demand_matrix.shape
        self.cell_width = self.screen_width // 2.1 // self.grid_cols
        self.cell_height = self.screen_height // 2.1 // self.grid_rows
        self.padding_x = 20
        self.padding_y = 20

        demand_min = np.min(self.env.demand_matrix)
        demand_max = np.max(self.env.demand_matrix)
        self.demand_norm = Normalize(vmin=demand_min, vmax=demand_max)
        self.demand_colormap = matplotlib.colormaps["RdYlGn_r"]

        revenue_min = np.min(self.env.revenue_matrix)
        revenue_max = np.max(self.env.revenue_matrix)
        self.revenue_norm = Normalize(vmin=revenue_min, vmax=revenue_max)
        self.revenue_colormap = matplotlib.colormaps["RdYlGn_r"]

        pygame.init()
        self.demand_font = pygame.font.Font(None, 24)
        self.revenue_font = pygame.font.Font(None, 16)
        self.small_font = pygame.font.Font(None, 24)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

    def _collect_history(self, deterministic: bool):
        """Collect the history of the agent's decisions."""

        def _add_to_history(obs: np.ndarray, reward: float):
            self.history["availabilities"].append(self.env.availabilities.copy())

            self.history["revenue"].append(self.env.revenue)
            self.history["bookings"].append(self.env.availabilities.sum())
            self.history["rewards"].append(reward)

        obs, _ = self.env.reset()
        _add_to_history(obs, reward=0)

        begin = time.monotonic()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action, _ = self.model.predict(
                obs,
                deterministic=deterministic,
            )
            obs, reward, terminated, truncated, _ = self.env.step(action)
            _add_to_history(obs, reward)
        print(f"Episode finished in {time.monotonic() - begin:.2f}s")

        self.total_frames = len(self.history["revenue"])

    def _handle_events(self):
        """Handle Pygame events."""
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    self.running = False
                case pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_down(event)
                case pygame.MOUSEBUTTONUP:
                    self.slider_dragging = False
                case pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event)
                case pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # Space bar for play/pause
                        self.playing = not self.playing
                    elif event.key == pygame.K_LEFT:  # Left arrow to step left
                        self.playing = False
                        self.left_key_held = True
                    elif event.key == pygame.K_RIGHT:  # Right arrow to step right
                        self.playing = False
                        self.right_key_held = True
                case pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        self.left_key_held = False
                    elif event.key == pygame.K_RIGHT:
                        self.right_key_held = False

        # Handle continuous stepping
        if self.left_key_held:
            self.current_frame = max(0, self.current_frame - 1)
        if self.right_key_held:
            self.current_frame = min(self.total_frames - 1, self.current_frame + 1)

    def _handle_mouse_down(self, event):
        """Handle mouse button down events."""
        slider_x, slider_y, slider_width, slider_height = self._get_slider_dimensions()
        if slider_x <= event.pos[0] <= slider_x + slider_width and slider_y <= event.pos[1] <= slider_y + slider_height:
            self.slider_dragging = True
            self.playing = False

        # Handle step left button
        if self._get_step_left_button_rect().collidepoint(event.pos):
            self.playing = False
            self.current_frame = max(0, self.current_frame - 1)

        # Handle step right button
        if self._get_step_right_button_rect().collidepoint(event.pos):
            self.playing = False
            self.current_frame = min(self.total_frames - 1, self.current_frame + 1)

        # Handle play/pause button
        if self._get_play_pause_button_rect().collidepoint(event.pos):
            self.playing = not self.playing

    def _handle_mouse_motion(self, event):
        """Handle mouse motion events."""
        if self.slider_dragging:
            slider_x, _, slider_width, _ = self._get_slider_dimensions()
            slider_pos = max(0, min(event.pos[0] - slider_x, slider_width)) - 1
            self.current_frame = int((slider_pos / slider_width) * self.total_frames)

    def _get_slider_dimensions(self):
        """Get the dimensions of the slider."""
        slider_width = 300
        slider_height = 20
        slider_x = (self.screen_width - slider_width) // 2
        slider_y = self.screen_height - 50
        return slider_x, slider_y, slider_width, slider_height

    def _draw_demand_matrix(self):
        """Draw the demand matrix"""
        demand_matrix = self.env.demand_matrix

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                rect = pygame.Rect(
                    self.padding_x + j * self.cell_width,
                    self.padding_y + i * self.cell_height,
                    self.cell_width,
                    self.cell_height,
                )
                demand_value = demand_matrix[i, j]
                color = self.demand_colormap(self.demand_norm(demand_value))  # Get RGBA color
                color = tuple(int(c * 255) for c in color[:3])  # Convert to RGB

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)  # Cell border

                # Draw demand value
                text = self.demand_font.render(str(demand_matrix[i, j]), True, WHITE)
                self.screen.blit(
                    text,
                    (
                        rect.x + self.cell_width // 2 - text.get_width() // 2,
                        rect.y + self.cell_height // 2 - text.get_height() // 2,
                    ),
                )

        # Draw day_xs labels (columns)
        for j, day_label in enumerate(self.day_xs):
            text = self.demand_font.render(str(day_label), True, BLACK)
            self.screen.blit(
                text,
                (
                    self.padding_x + j * self.cell_width + self.cell_width // 2 - text.get_width() // 2,
                    self.padding_y + self.grid_rows * self.cell_height + 5,
                ),
            )

        # Draw bucket_names labels (rows)
        for i, bucket_label in enumerate(self.bucket_names):
            text = self.demand_font.render(str(bucket_label), True, BLACK)
            self.screen.blit(
                text,
                (
                    self.padding_x - text.get_width() - 5,
                    self.padding_y + i * self.cell_height + self.cell_height // 2 - text.get_height() // 2,
                ),
            )

    def _draw_availabilities(self):
        """Draw the availabilities on top of revenue matrix."""
        revenue_matrix = self.env.revenue_matrix.astype(int)
        availabilities = self.history["availabilities"][self.current_frame]

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                rect = pygame.Rect(
                    self.screen_width // 2 + self.padding_x + j * self.cell_width,
                    self.padding_y + i * self.cell_height,
                    self.cell_width,
                    self.cell_height,
                )
                revenue_value = revenue_matrix[i, j]
                color = self.revenue_colormap(self.revenue_norm(revenue_value))  # Get RGBA color
                color = tuple(int(c * 255) for c in color[:3])  # Convert to RGB

                # Apply transparency for selected buckets
                if not availabilities[i, j]:
                    color = tuple(int(c * 0.5) for c in color)  # Darken the color

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)  # Cell border

                # if availability_value > 0:
                text = self.revenue_font.render(str(revenue_value), True, WHITE)
                self.screen.blit(
                    text,
                    (
                        rect.x + self.cell_width // 2 - text.get_width() // 2,
                        rect.y + self.cell_height // 2 - text.get_height() // 2,
                    ),
                )

        # Draw day_xs labels (columns)
        for j, day_label in enumerate(self.day_xs):
            text = self.demand_font.render(str(day_label), True, BLACK)
            self.screen.blit(
                text,
                (
                    self.screen_width // 2
                    + self.padding_x
                    + j * self.cell_width
                    + self.cell_width // 2
                    - text.get_width() // 2,
                    self.padding_y + self.grid_rows * self.cell_height + 5,
                ),
            )

        # Draw bucket_names labels (rows)
        for i, bucket_label in enumerate(self.bucket_names):
            text = self.demand_font.render(str(bucket_label), True, BLACK)
            self.screen.blit(
                text,
                (
                    self.screen_width // 2 + self.padding_x - text.get_width() - 5,
                    self.padding_y + i * self.cell_height + self.cell_height // 2 - text.get_height() // 2,
                ),
            )

    def _get_step_left_button_rect(self):
        """Get the rectangle for the step left button."""
        slider_x, slider_y, _, slider_height = self._get_slider_dimensions()
        return pygame.Rect(slider_x - 60, slider_y - 40, 40, slider_height)

    def _get_step_right_button_rect(self):
        """Get the rectangle for the step right button."""
        slider_x, slider_y, slider_width, slider_height = self._get_slider_dimensions()
        return pygame.Rect(slider_x + slider_width + 20, slider_y - 40, 40, slider_height)

    def _get_play_pause_button_rect(self):
        """Get the rectangle for the play/pause button."""
        slider_x, slider_y, _, _ = self._get_slider_dimensions()
        return pygame.Rect(slider_x - 120, slider_y, 100, 40)

    def _draw_buttons(self):
        """Draw the step left, step right, and play/pause buttons."""
        # Step left button
        step_left_button_rect = self._get_step_left_button_rect()
        pygame.draw.rect(self.screen, LIGHT_GRAY, step_left_button_rect)
        left_text = self.demand_font.render("<", True, BLACK)
        self.screen.blit(
            left_text,
            (
                step_left_button_rect.centerx - left_text.get_width() // 2,
                step_left_button_rect.centery - left_text.get_height() // 2,
            ),
        )

        # Step right button
        step_right_button_rect = self._get_step_right_button_rect()
        pygame.draw.rect(self.screen, LIGHT_GRAY, step_right_button_rect)
        right_text = self.demand_font.render(">", True, BLACK)
        self.screen.blit(
            right_text,
            (
                step_right_button_rect.centerx - right_text.get_width() // 2,
                step_right_button_rect.centery - right_text.get_height() // 2,
            ),
        )

        # Play/pause button
        play_pause_button_rect = self._get_play_pause_button_rect()
        pygame.draw.rect(self.screen, LIGHT_GRAY, play_pause_button_rect)
        if self.playing:
            # Draw pause icon (two vertical bars)
            bar_width = 10
            bar_height = 20
            bar_spacing = 5
            left_bar = pygame.Rect(
                play_pause_button_rect.centerx - bar_width - bar_spacing,
                play_pause_button_rect.centery - bar_height // 2,
                bar_width,
                bar_height,
            )
            right_bar = pygame.Rect(
                play_pause_button_rect.centerx + bar_spacing,
                play_pause_button_rect.centery - bar_height // 2,
                bar_width,
                bar_height,
            )
            pygame.draw.rect(self.screen, BLACK, left_bar)
            pygame.draw.rect(self.screen, BLACK, right_bar)
        else:
            # Draw play icon (triangle)
            triangle_points = [
                (
                    play_pause_button_rect.centerx - 10,
                    play_pause_button_rect.centery - 10,
                ),
                (
                    play_pause_button_rect.centerx - 10,
                    play_pause_button_rect.centery + 10,
                ),
                (play_pause_button_rect.centerx + 10, play_pause_button_rect.centery),
            ]
            pygame.draw.polygon(self.screen, BLACK, triangle_points)

    def _draw_slider(self):
        """Draw the slider."""
        slider_x, slider_y, slider_width, slider_height = self._get_slider_dimensions()
        slider_pos = int((self.current_frame / self.total_frames) * slider_width)
        pygame.draw.rect(self.screen, LIGHT_GRAY, (slider_x, slider_y, slider_width, slider_height))
        pygame.draw.rect(self.screen, GREEN, (slider_x + slider_pos - 5, slider_y, 10, slider_height))

    def _draw_stats(self):
        self.screen.blit(
            self.demand_font.render(f"Revenue: {self.history['revenue'][self.current_frame]:.2f}", True, BLACK),
            (self.screen_width // 2, self.padding_y + self.screen_height // 2 + 20),
        )
        self.screen.blit(
            self.demand_font.render(f"Bookings: {self.history['bookings'][self.current_frame]:.2f}", True, BLACK),
            (self.screen_width // 2, self.padding_y + self.screen_height // 2 + 50),
        )

        remaining_capacity = self.env.initial_capacity - np.sum(self.history["availabilities"][self.current_frame])
        self.screen.blit(
            self.demand_font.render(f"Remaining Capacity: {remaining_capacity}", True, BLACK),
            (self.screen_width // 2, self.padding_y + self.screen_height // 2 + 80),
        )

        reward = self.history["rewards"][self.current_frame]
        cumulated_reward = np.sum(
            np.dot(
                self.history["rewards"][: self.current_frame + 1],
                np.array([self.model.gamma**i for i in range(self.current_frame + 1)]),
            )
        )
        self.screen.blit(
            self.demand_font.render(f"Cumulated reward: {cumulated_reward:.2f}", True, BLACK),
            (self.screen_width // 2, self.padding_y + self.screen_height // 2 + 100),
        )
        self.screen.blit(
            self.demand_font.render(f"Last reward: {reward:.2f}", True, BLACK),
            (self.screen_width // 2, self.padding_y + self.screen_height // 2 + 150),
        )

    def _draw_frame(self):
        """Draw the current frame."""
        self._draw_demand_matrix()
        self._draw_availabilities()
        self._draw_stats()
        self._draw_buttons()
        self._draw_slider()
