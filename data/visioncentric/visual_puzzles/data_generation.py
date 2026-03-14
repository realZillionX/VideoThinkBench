import base64
import copy
import io
import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.video_encoding import encode_rgb_frames_to_mp4
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from fire import Fire
except Exception:
    Fire = None

try:
    from pydantic import BaseModel
except Exception:
    class BaseModel:
        def __init__(self, **kwargs):
            annotations = getattr(self.__class__, "__annotations__", {})
            for name in annotations:
                if name in kwargs:
                    value = kwargs.pop(name)
                elif hasattr(self.__class__, name):
                    value = copy.deepcopy(getattr(self.__class__, name))
                else:
                    raise TypeError(f"Missing required argument: {name}")
                setattr(self, name, value)
            for name, value in kwargs.items():
                setattr(self, name, value)

Point = Tuple[float, float]

_MODULE_DIR = Path(__file__).resolve().parent


def _load_font(font_spec: str, *, size: int) -> ImageFont.ImageFont:
    candidates = []
    if font_spec:
        raw_path = Path(font_spec)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(_MODULE_DIR / raw_path)
            candidates.append(_MODULE_DIR.parent.parent.parent / raw_path)
    candidates.extend(
        [
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    for fallback_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(fallback_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


class ColorGridPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )

    def sample_colors(self) -> Tuple[List[str], List[str]]:
        while True:
            names = random.sample(list(self.colors), k=3)
            if "orange" in names and "yellow" in names:
                continue  # Hard to distinguish
            return names

    def draw_circle(self, draw: ImageDraw, point: Point, radius: int, color: str):
        x, y = point
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, fill=color, outline="black", width=line_width)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)
        a, b, c = size // 4, size // 2, size * 3 // 4
        positions = [(x, y) for x in [a, b, c] for y in [a, b, c]]

        names = self.sample_colors()
        values = [[0, 2, 6, 8], [1, 3, 5, 7], [4]]
        mapping = {k: v for k, v in zip(names, values)}
        i_answer = random.choice([0, 2, 6, 8, 1, 3, 5, 7])
        answer = ""
        font = _load_font(self.path_font, size=size // 10)

        for k, lst in mapping.items():
            for i in lst:
                if i == i_answer:
                    answer = k
                    solution_color = self.colors[k]
                    self.draw_circle(solution_draw, positions[i], radius=size // 10, color=solution_color)
                    puzzle_draw.text(
                        positions[i],
                        text="?",
                        font=font,
                        anchor="mm",
                        fill="black",
                    )
                else:
                    color = self.colors[k]
                    self.draw_circle(solution_draw, positions[i], radius=size // 10, color=color)
                    self.draw_circle(puzzle_draw, positions[i], radius=size // 10, color=color)

        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        grid = ["?"] * 9
        for k, lst in mapping.items():
            for i in lst:
                if i != i_answer:
                    grid[i] = k
        grid = grid[::-1]
        location = "at the corner" if answer == names[0] else "adjacent to the center"

        return (
            dict(
                question="What is the color of the missing part denoted with a question mark?",
                answer=answer,
                options=sample_options(answer, sorted(self.colors), k=4),
                caption=f"There are circles with different colors arranged with a grid formation in the image. The colors in the first row are {grid[:3]}, the colors in the second row are {grid[3:6]}, and the colors in the third row are {grid[6:9]}.",
                explanation=f"We observe that the circles at the corners are {names[0]}, while the circles directly adjacent to the center are {names[1]}. Only the center circle is {names[2]}. Hence, the pattern is that the circles alternate in color depending on if they are at the corner or adjacent to the center.",
                deduction=f"Based on the pattern that the circles alternate in color depending on if they are at the corner or adjacent to the center, the missing color of the part that is {location} should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


class ColorHexagonPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"

    @staticmethod
    def get_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return x, y

    def sample_colors(self) -> Tuple[List[str], List[str]]:
        while True:
            names = random.sample(list(self.colors), k=3)
            if "orange" in names and "yellow" in names:
                continue  # Hard to distinguish
            names = names + names
            colors = [self.colors[n] for n in names]
            return names, colors

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)
        center = size // 2

        # Hexagon properties
        side_length = size // 3  # Length of a side of the hexagon and triangles
        triangle_height = math.sqrt(3) / 2 * side_length

        # The vertices of the hexagon
        hexagon = [
            (center + side_length / 2, center - triangle_height),
            (center - side_length / 2, center - triangle_height),
            (center - side_length, center),
            (center - side_length / 2, center + triangle_height),
            (center + side_length / 2, center + triangle_height),
            (center + side_length, center),
        ]

        # Colors for the triangles
        names, colors = self.sample_colors()
        i_answer = random.randint(0, len(colors) - 1)
        answer = names[i_answer]
        puzzle_colors = colors.copy()
        puzzle_colors[i_answer] = "#eeeeee"  # Grey
        font = _load_font(self.path_font, size=size // 10)

        # Draw the hexagon made of six triangles
        for i in range(6):
            # Coordinates of the triangle vertices
            triangle = [hexagon[i], hexagon[(i + 1) % 6], (center, center)]
            # Draw the triangle
            solution_draw.polygon(triangle, fill=colors[i])
            puzzle_draw.polygon(triangle, fill=puzzle_colors[i])
            # Draw the outline with custom width
            points = [hexagon[i], hexagon[(i + 1) % 6], (center, center), hexagon[i]]
            solution_draw.line(points, fill="black", width=self.scale_factor * 4)
            puzzle_draw.line(points, fill="black", width=self.scale_factor * 4)
            # Draw "?" on the missing answer part
            if i == i_answer:
                puzzle_draw.text(
                    self.get_centroid(triangle),
                    text="?",
                    font=font,
                    anchor="mm",
                    fill="black",
                )

        names_display = names.copy()
        names_display[i_answer] = "?"
        instances = sorted(set(n for n in names_display if n not in [answer, "?"]))
        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=answer,
                options=sample_options(answer, options=list(self.colors), k=4),
                caption=f"There is a hexagon split into six parts with the colors {names_display} in an anti-clockwise order.",
                explanation=f"We observe that a {instances[0]} part is opposite another {instances[0]} part, and a {instances[1]} part is opposite another {instances[1]} part. Thus, the pattern is that the colors in opposite parts are the same.",
                deduction=f"Based on the pattern that spatially opposite parts have the same color, the missing color of the part which is opposite a {answer} part should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


class ColorOverlapSquaresPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    numbers: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    num_sides: int = 4
    rotate_range: Optional[Tuple[int, int]] = (-45, 0)

    def get_points(self, center: Point, radius: float, angle: int = 0) -> List[Point]:
        vertices = []
        for i in range(self.num_sides):
            theta = 2 * math.pi / self.num_sides * i
            theta -= math.pi / 4  # Adjust to flat rectangle by default
            theta -= math.radians(angle)
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_squares(self, draw: ImageDraw, colors: List[str]):
        size = self.image_size * self.scale_factor
        line_width = size // 150

        # Center big square
        a = self.get_points((size / 2, size / 2), radius=size / 4)
        draw.polygon(a, outline="black", fill=colors[1], width=line_width)

        # Top right rotated square
        b = self.get_points(a[0], radius=size / 4, angle=45)
        draw.polygon(b, outline="black", fill=colors[2], width=line_width)

        # Bottom left rotated square
        c = self.get_points(a[2], radius=size / 4, angle=45)
        draw.polygon(c, outline="black", fill=colors[0], width=line_width)

        # Top right overlap triangle
        ab = [a[0], b[2], b[3]]
        draw.polygon(ab, outline="black", fill=colors[4], width=line_width)

        # Bottom left overlap triangle
        ac = [a[2], c[0], c[1]]
        draw.polygon(ac, outline="black", fill=colors[3], width=line_width)

    def sample_color_names(self) -> Tuple[str, str, str, str, str]:
        a, b, c = random.sample(["red", "yellow", "blue"], k=3)
        mapping = dict(redyellow="orange", blueyellow="green", bluered="purple")
        d = mapping["".join(sorted([a, b]))]
        e = mapping["".join(sorted([b, c]))]
        assert [x in self.colors for x in [a, b, c, d, e]]
        return a, b, c, d, e

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)

        color_names = list(self.sample_color_names())
        solution_colors = [self.colors[n] for n in color_names]
        puzzle_colors = solution_colors[:]
        font = _load_font(self.path_font, size=size // 10)

        if random.random() > 0.5:
            missing_index = 0
            position = (size // 4, size * 3 // 4)
        else:
            missing_index = 2
            position = (size * 3 // 4, size // 4)

        answer = color_names[missing_index]
        puzzle_colors[missing_index] = "#eeeeee"  # Grey

        self.draw_squares(solution_draw, solution_colors)
        self.draw_squares(puzzle_draw, puzzle_colors)

        puzzle_draw.text(
            position,
            text="?",
            font=font,
            anchor="mm",
            fill="black",
        )

        display_names = [("?" if idx == missing_index else n) for idx, n in enumerate(color_names)]
        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)

        if self.rotate_range is not None:
            rotate_angle = random.uniform(self.rotate_range[0], self.rotate_range[1])

            puzzle_image = puzzle_image.rotate(
                rotate_angle,  resample=Image.BICUBIC, expand=False, fillcolor='white'
            )
            solution_image = solution_image.rotate(
                rotate_angle, resample=Image.BICUBIC, expand=False, fillcolor='white'
            )

        instances = [display_names[0], display_names[1], display_names[3]]
        overlap = display_names[4]
        if "?" in instances:
            instances = [display_names[1], display_names[2], display_names[4]]
            overlap = display_names[3]

        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=answer,
                options=sample_options(answer, sorted(self.colors), k=4),
                caption=f"There are 3 squares which overlap each other in the image. The color of the squares are {display_names[:3]}. The part where the first and second squares overlap is {display_names[3]}. The part where the second and third squares overlap is {display_names[4]}.",
                explanation=f"We observe that the {instances[0]} and {instances[1]} squares overlap to form {instances[2]}. Hence, the pattern is that the color of the part where two squares overlap is determined by mixing the two colors.",
                deduction=f"Based on the pattern that the color of the part where two squares overlap is determined by mixing the two colors, the missing color of the part which overlaps with {display_names[1]} to form {overlap} should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )

class ColorSizePattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    colors: Dict[str, str] = dict(
        blue=["#b5d5f2", "#8dbcea", "#5f9ed6", "#3d85c5"],
        green=["#c4e3b8", "#9dd293", "#6fb76a", "#489241"],
        yellow=["#ffe191", "#ffd05d", "#fcbc1f", "#d79a00"],
        red=["#f2a8a8", "#e67b7b", "#d64e4e", "#af1919"],
        purple=["#c9bde4", "#a995d6", "#845dcc", "#5b32a0"],
        orange=["#f7ca9d", "#f3a96b", "#e98330", "#c9600e"],
    )
    shape_sides: Dict[str, Optional[int]] = dict(circle=None, square=4, pentagon=5, hexagon=6)

    def draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int, **kwargs):
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 150
        draw.ellipse(position, width=line_width, **kwargs)

    @staticmethod
    def get_regular_polygon_points(center: Point, radius: float, sides: int) -> List[Point]:
        angle_offset = -math.pi / 2 if sides % 2 != 0 else -math.pi / 2 + math.pi / sides
        return [
            (
                center[0] + radius * math.cos(angle_offset + 2 * math.pi * i / sides),
                center[1] + radius * math.sin(angle_offset + 2 * math.pi * i / sides),
            )
            for i in range(sides)
        ]

    def draw_regular_polygon(self, draw: ImageDraw, center: Point, radius: int, sides: int, **kwargs):
        points = self.get_regular_polygon_points(center, radius, sides)
        draw.polygon(points, **kwargs)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)
        font = _load_font(self.path_font, size=size // 10)

        key = random.choice(sorted(self.colors))
        colors = self.colors[key]
        assert len(set(colors)) == len(colors)
        names = [f"light {key}", f"medium {key}", f"dark {key}"]
        if random.random() > 0.5:
            colors, names = colors[::-1], names[::-1]

        shape_type = random.choice(list(self.shape_sides))
        shape_singular, shape_plural = dict(
            circle=("circle", "circles"),
            square=("square", "squares"),
            pentagon=("pentagon", "pentagons"),
            hexagon=("hexagon", "hexagons"),
        )[shape_type]
        radii = [size * 0.4, size * 0.3, size * 0.2, size * 0.1]
        for i, r in enumerate(radii):
            x = y = size // 2
            solution_fill = colors[i]
            line_width = self.image_size * self.scale_factor // 150
            if shape_type == "circle":
                self.draw_circle(
                    solution_draw, x, y, round(r), fill=solution_fill, outline="black"
                )
            else:
                sides = self.shape_sides[shape_type]
                self.draw_regular_polygon(
                    solution_draw,
                    center=(x, y),
                    radius=round(r),
                    sides=sides,
                    fill=solution_fill,
                    outline="black",
                    width=line_width,
                )
            puzzle_fill = solution_fill if i != len(radii) - 1 else "#eeeeee"
            if shape_type == "circle":
                self.draw_circle(
                    puzzle_draw, x, y, round(r), fill=puzzle_fill, outline="black"
                )
            else:
                self.draw_regular_polygon(
                    puzzle_draw,
                    center=(x, y),
                    radius=round(r),
                    sides=sides,
                    fill=puzzle_fill,
                    outline="black",
                    width=line_width,
                )
            if i == len(radii) - 1:
                puzzle_draw.text(
                    xy=(x, y),
                    text="?",
                    font=font,
                    anchor="mm",
                    fill="black",
                )

        answer = names[-1]
        lst = ["light " + k for k in self.colors] + ["dark " + k for k in self.colors]
        lst.remove(names[0])
        lst.remove(names[-1])
        options = [names[0], names[-1]] + random.sample(lst, k=2)
        assert answer in options
        assert len(set(options)) == len(options)

        if "dark" in answer:
            pairs = [
                ("extra large", f"very light {key}"),
                ("large", f"light {key}"),
                ("medium", f"medium {key}"),
                ("small", "?"),
            ]
            trend = "darker"
        else:
            pairs = [
                ("extra large", f"very dark {key}"),
                ("large", f"dark {key}"),
                ("medium", f"medium {key}"),
                ("small", "?"),
            ]
            trend = "lighter"

        shuffled = random.sample(pairs, k=len(pairs))
        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=answer,
                options=random.sample(options, k=len(options)),
                caption=f"There are {shape_plural} of various sizes and colors in the image. The {shape_plural} are {[p[0] for p in shuffled]} size, and their colors are {[p[1] for p in shuffled]}.",
                explanation=f"We observe that the largest {shape_singular} is {pairs[0][1]} color, and the smaller {shape_plural} change color from {pairs[1][1]} to {pairs[2][1]}. Hence, the pattern is that the {shape_plural} become {trend} as they become smaller.",
                deduction=f"Based on the pattern that the {shape_plural} become {trend} as they become smaller, the missing color of the smallest {shape_singular} denoted with a question mark should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


class PolygonSidesColorPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"

    @staticmethod
    def draw_polygon(draw, sides, center, size, color):
        angle = 360 / sides
        points = []

        for i in range(sides):
            x = center[0] + size * math.cos(math.radians(i * angle))
            y = center[1] + size * math.sin(math.radians(i * angle))
            points.append((x, y))

        draw.polygon(points, outline="black", fill=color, width=4)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)
        font = _load_font(self.path_font, size=50 * self.scale_factor)

        sides = random.sample(range(3, 10), 3)
        side2col = dict(zip(sides, random.sample(list(self.colors.keys()), 3)))
        sides *= 2
        random.shuffle(sides)
        answer_location = random.choice(range(len(sides)))

        answer = side2col[sides[answer_location]]

        options = set(list(self.colors.keys())) - {answer}
        options = random.sample(list(options), 3)
        options.append(answer)
        random.shuffle(options)

        center = size // 2
        distance = 175 * self.scale_factor

        for i, side in enumerate(sides):
            polygon_distance = distance - 0.5 * (i % 2) * distance
            angle = (i / len(sides)) * 2 * math.pi
            center_y = center - int(polygon_distance * math.cos(angle))
            center_x = center - int(polygon_distance * math.sin(angle))
            polygon_size = 60 * self.scale_factor

            actual_color = self.colors[side2col[side]]
            self.draw_polygon(
                solution_draw, side, (center_x, center_y), polygon_size, actual_color
            )

            if i == answer_location:
                self.draw_polygon(
                    puzzle_draw, side, (center_x, center_y), polygon_size, "#eeeeee"
                )
                puzzle_draw.text(
                    (center_x, center_y),
                    "?",
                    font=font,
                    anchor="mm",
                    fill="black",
                )
            else:
                self.draw_polygon(
                    puzzle_draw, side, (center_x, center_y), polygon_size, actual_color
                )

        colors = [side2col[side] for side in sides]
        colors[answer_location] = "?"
        top_row = [colors[0]]
        middle_row = [colors[1], colors[5]]
        bottom_row = [colors[2], colors[3], colors[4]]

        explanation_side = list(
            set(
                [
                    side
                    for side in sides
                    if (side != "?" and side != sides[answer_location])
                ]
            )
        )

        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=answer,
                options=options,
                caption=f"There are 6 colored polygons arranged in a triangle with color {top_row} in the top row, {middle_row} in the middle row, and {bottom_row} in the bottom row.",
                explanation=f"We observe that the polygon with {explanation_side[0]} sides is {side2col[explanation_side[0]]} in color and the polygon with {explanation_side[1]} sides is {side2col[explanation_side[1]]} in color. Thus, the pattern is that the polygons with the same number of sides have the same color.",
                deduction=f"Based on the pattern that the polygons with the same number of sides have the same color, the missing color of the part with {sides[answer_location]} sides should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


class RectangleHeightColorPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"

    def draw_box(
        self,
        draw: ImageDraw,
        point: Point,
        width: float,
        height: float,
        color: str,
    ):
        size = self.image_size * self.scale_factor
        draw.rounded_rectangle(
            [point[0] - width, point[1] - height, point[0] + width, point[1] + height],
            outline="black",
            width=size // 200,
            radius=size // 20,
            fill=color,
        )

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=_load_font(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    @staticmethod
    def assign_numbers(colors: List[str]) -> List[int]:
        unique = sorted(set(colors))
        numbers = [i + 1 for i in range(len(unique))]
        random.shuffle(numbers)
        mapping = {u: i for u, i in zip(unique, numbers)}
        return [mapping[c] for c in colors]

    def make_sample(self):
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", (size, size), "white")
        solution_image = Image.new("RGB", (size, size), "white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)

        colors = random.sample(sorted(self.colors), k=3) * 2
        answer = colors[0]
        random.shuffle(colors)
        colors.append(answer)
        numbers = self.assign_numbers(colors)

        for i, num in enumerate(numbers):
            factor = size / (len(numbers) + 1)
            point = (factor * (i + 1), size // 2)
            is_answer = i == len(numbers) - 1
            current_color = self.colors[colors[i]]
            self.draw_box(
                solution_draw,
                point=point,
                width=factor / 2,
                height=factor * num,
                color=current_color,
            )
            self.draw_box(
                puzzle_draw,
                point=point,
                width=factor / 2,
                height=factor * num,
                color="#eeeeee" if is_answer else self.colors[colors[i]],
            )
            if is_answer:
                self.draw_text(puzzle_draw, point, text="?")

        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)

        lengths = [["short", "medium", "long"][num - 1] for num in numbers]
        instances = list(set((a, b) for (a, b) in zip(lengths, colors) if b != answer))
        colors_display = colors[:]
        colors_display[-1] = "?"

        return (
            dict(
                question="What is the missing color of the part denoted with a question mark?",
                answer=str(answer),
                options=sample_options(answer, sorted(self.colors), k=4),
                caption=f"There are {len(numbers)} rectangles in the image with varying colors and lengths. The lengths from left to right are {lengths}. The colors from left to right are {colors_display}.",
                explanation=f"We observe that the {instances[0][1]} rectangles are of {instances[0][0]} length and the {instances[1][1]} rectangles are of {instances[1][0]} length. Hence, the pattern is that the color of each rectangle corresponds to its length.",
                deduction=f"Based on the pattern that the color of each rectangle corresponds to its length, the missing color of the part denoted with a question mark should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


class ShapeReflectPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9ead3"  # Light green for all shapes
    shapes: Dict[str, int] = dict(triangle=3, square=4, pentagon=5, hexagon=6)

    def draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int, **kwargs):
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, width=line_width, **kwargs)

    def draw_dotted_circle(
        self, draw: ImageDraw, center: Tuple[float, float], radius: int, num_dots: int
    ):
        self.draw_circle(draw, *center, radius, outline="black")
        angle_between_dots = 2 * math.pi / num_dots
        for i in range(0, num_dots, 2):
            theta = angle_between_dots * i
            x = round(center[0] + radius * math.cos(theta))
            y = round(center[1] + radius * math.sin(theta))
            self.draw_circle(draw, x, y, radius=radius // 10, fill="white")

    def draw_shape(
        self,
        draw: ImageDraw,
        center: Tuple[float, float],
        num_sides: int,
        do_flip: bool,
    ):
        size = self.image_size * self.scale_factor
        if num_sides == 0:
            draw.text(
                center,
                text="?",
                font=_load_font(self.path_font, size=size // 10),
                anchor="mm",
                fill="black",
            )
            self.draw_dotted_circle(draw, center, radius=size // 10, num_dots=32)
            return

        # Adjust start angle based on even or odd number of sides
        angle = math.pi * 2 / num_sides
        if num_sides % 2 == 0:
            start = math.pi / 2 - angle / 2
        else:
            start = 0
        if do_flip:
            start += math.pi

        radius = size // 10
        points = [
            (
                center[0] + math.sin(start + angle * i) * radius,
                center[1] - math.cos(start + angle * i) * radius,
            )
            for i in range(num_sides)
        ]

        width = size // 200
        draw.polygon(points, fill=self.color, outline="black", width=width)
        draw.line([(size // 8, size // 2), (size * 7 // 8, size // 2)], "black", width)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)

        a, b, c = size // 4, size // 2, size * 3 // 4
        positions = [(x, y) for y in [size // 3, size * 2 // 3] for x in [a, b, c]]
        names = random.sample(sorted(self.shapes), k=3) * 2
        i_answer = random.randint(0, len(names) - 1)
        answer = names[i_answer]

        for i, n in enumerate(names):
            actual_sides = self.shapes[n]
            # Solution always shows the actual shape
            self.draw_shape(solution_draw, positions[i], actual_sides, do_flip=i >= 3)
            if i == i_answer:
                self.draw_shape(puzzle_draw, positions[i], 0, do_flip=i >= 3)
            else:
                self.draw_shape(puzzle_draw, positions[i], actual_sides, do_flip=i >= 3)

        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        names[i_answer] = "?"
        instances = sorted(set(n for n in names if n != "?"))
        return (
            dict(
                question="What is the missing shape denoted by a question mark?",
                answer=answer,
                options=sample_options(answer, sorted(self.shapes), k=4),
                caption=f"There are six shapes in the image separated by a line. In the top part there are {names[:3]}. In the bottom part there are {names[3:]}.",
                explanation=f"We observe that the {instances[0]} is reflected across the line as a {instances[0]}. Similarly, the {instances[1]} is reflected as a {instances[1]}. Hence, the pattern is that each shape in the top part is reflected in the bottom part.",
                deduction=f"Based on the pattern that each shape in the top part is reflected in the bottom part, the missing shape which is reflected from a {answer} part should be a {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


class ShapeSizeGridPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9ead3"  # Light green for all shapes
    shapes: Dict[str, int] = dict(triangle=3, square=4, pentagon=5, hexagon=6)

    @staticmethod
    def get_points(num_sides: int, center: Point, radius: int) -> List[Point]:
        vertices = []
        for i in range(num_sides):
            theta = 2 * math.pi / num_sides * i
            if num_sides % 2 != 0:
                theta -= math.pi / 2
            elif num_sides == 4:
                theta -= math.pi / 4

            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=_load_font(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    @staticmethod
    def random_rotate_matrix(matrix: List[list]) -> List[list]:
        angle = random.choice([90, 180, 270, 360])
        if angle == 90:
            # Rotate by 90 degrees
            new = [list(row) for row in zip(*matrix[::-1])]
        elif angle == 180:
            # Rotate by 180 degrees
            new = [row[::-1] for row in matrix[::-1]]
        elif angle == 270:
            # Rotate by 270 degrees (or 90 degrees counter-clockwise)
            new = [list(row) for row in zip(*matrix)][::-1]
        else:
            new = matrix

        return [
            [
                (new[i][j][0], new[i][j][1], matrix[i][j][2], matrix[i][j][3])
                for j in range(len(matrix[i]))
            ]
            for i in range(len(matrix))
        ]

    def make_sample(self):
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)

        a, b, c = random.sample(sorted(self.shapes), k=3)
        mapping = dict(small=(size * 0.05), medium=(size * 0.09), large=(size * 0.13))
        d, e, f = size * 0.25, size * 0.50, size * 0.75
        data = [
            [(a, "small", d, d), (b, "small", e, d), (c, "small", f, d)],
            [(a, "medium", d, e), (b, "medium", e, e), (c, "medium", f, e)],
            [(a, "large", d, f), (b, "large", e, f), (c, "large", f, f)],
        ]
        data = self.random_rotate_matrix(data)
        answer = random.choice([item for lst in data for item in lst])

        for lst in data:
            for item in lst:
                name, radius, x, y = item
                shape = self.get_points(
                    num_sides=self.shapes[name],
                    center=(x, y),
                    radius=mapping[radius],
                )
                solution_draw.polygon(
                    shape, fill=self.color, outline="black", width=size // 100
                )
                if item == answer:
                    self.draw_text(puzzle_draw, point=(x, y), text="?")
                else:
                    puzzle_draw.polygon(
                        shape, fill=self.color, outline="black", width=size // 100
                    )

        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        m = [[("?" if x == answer else f"{x[1]} {x[0]}") for x in row] for row in data]
        if len(set(x[1] for x in data[0])) == 1:
            trend_size = f"rows contain {data[0][0][1]} shapes, {data[1][0][1]} shapes, and {data[2][0][1]} shapes respectively"
            trend_shapes = f"columns contain {data[0][0][0]}s, {data[0][1][0]}s, and {data[0][2][0]}s respectively"
            pattern = "the shapes within each column are the same, while each row progresses the size of the shapes"
        else:
            trend_size = f"columns contain {data[0][0][1]} shapes, {data[0][1][1]} shapes, and {data[0][2][1]} shapes respectively"
            trend_shapes = f"rows contain {data[0][0][0]}s, {data[1][0][0]}s, and {data[2][0][0]}s respectively"
            pattern = "the shapes within each row are the same, while each column progresses the size of the shapes"

        return (
            dict(
                question=f"What is the size of the missing part denoted by a question mark?",
                answer=answer[1],
                options=sample_options(answer[1], sorted(mapping), k=3),
                caption=f"There are 9 shapes arranged in a grid with different sizes in the image, of which there is 1 missing shape. The first row is {m[0]}, the second row is {m[1]}, and the third row is {m[2]}.",
                explanation=f"We observe that the {trend_size}. On the other hand, the {trend_shapes}. Hence, the pattern is that {pattern}.",
                deduction=f"Based on the pattern that {pattern}, the size of the missing {answer[0]} should be {answer[1]}.",
            ),
            puzzle_image,
            solution_image,
        )


class SizeCyclePattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#fff2cc"  # Light yellow for all circles

    def draw_circle(self, draw: ImageDraw, point: Point, radius: int):
        x, y = point
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 150
        draw.ellipse(position, fill=self.color, outline="black", width=line_width)

    @staticmethod
    def get_points(n_sides: int, center: Point, radius: int, angle: int) -> List[Point]:
        def regular_polygon_vertices(num_sides):
            vertices = []
            for i in range(num_sides):
                theta = 2 * math.pi / num_sides * i
                x = center[0] + radius * math.cos(theta)
                y = center[1] + radius * math.sin(theta)
                vertices.append((x, y))
            return vertices

        def rotate_point(origin, point):
            ox, oy = origin
            px, py = point
            theta = math.radians(angle)  # Convert to radians
            qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
            qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)
            return qx, qy

        polygon_vertices = regular_polygon_vertices(n_sides)
        # assert self.get_centroid(polygon_vertices) == center
        rotated_vertices = [rotate_point(center, v) for v in polygon_vertices]
        # assert self.get_centroid(rotated_vertices) == center
        return rotated_vertices

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        font = _load_font(self.path_font, size=size // 10)
        text_bbox = font.getbbox(text)
        half_width = (text_bbox[2] - text_bbox[0]) / 2.0
        half_height = (text_bbox[3] - text_bbox[1]) / 2.0
        safe_margin = max(8.0, size * 0.015)
        safe_x = min(max(point[0], half_width + safe_margin), size - half_width - safe_margin)
        safe_y = min(max(point[1], half_height + safe_margin), size - half_height - safe_margin)
        draw.text(
            (safe_x, safe_y),
            text=text,
            font=font,
            anchor="mm",
            fill="black",
        )

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)

        center = size // 2, size // 2
        offset = random.randint(0, 360)
        mapping = dict(
            small=(size * 0.048, round(size * 0.11), 0 + offset),
            medium=(size * 0.072, round(size * 0.23), 20 + offset),
            large=(size * 0.092, round(size * 0.32), 45 + offset),
        )

        names = []
        num_sides = 3
        answer = ""
        i_answer = random.randint(0, num_sides * len(mapping) - 1)
        for n, (radius, distance, angle) in mapping.items():
            for p in self.get_points(num_sides, center, distance, angle):
                names.append(n)
                if len(names) - 1 == i_answer:
                    answer = n
                    solution_draw_radius = round(radius)
                    self.draw_circle(solution_draw, p, solution_draw_radius)
                    self.draw_text(puzzle_draw, p, "?")
                else:
                    draw_radius = round(radius)
                    self.draw_circle(solution_draw, p, draw_radius)
                    self.draw_circle(puzzle_draw, p, draw_radius)

        names[i_answer] = "?"
        arms = [
            [names[0], names[3], names[6]],
            [names[1], names[4], names[7]],
            [names[2], names[5], names[8]],
        ]
        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        answer_location = dict(
            small="closest to center",
            medium="neither closest nor farthest from center",
            large="farthest from center",
        )[answer]

        return (
            dict(
                question="What is the size of the missing circle denoted with a question mark?",
                answer=answer,
                options=sample_options(answer, sorted(mapping), k=3),
                caption=f"There are circles arranged in a spiral with three arms. The first arm has circles of sizes {arms[0]}, the second arm has circles of sizes {arms[1]}, and the third arm has circles of sizes {arms[2]}.",
                explanation=f"We observe that the circles in each arm progress in size from small to medium to large. Thus, the pattern is that the circles in each arm get bigger as they progress away from the center of the spiral.",
                deduction=f"Based on the pattern that the circles in each arm get bigger as they progress away from the center of the spiral, the size of the missing part that is {answer_location} should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


class SizeGridPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#fff2cc"  # Light yellow for all circles

    def draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int):
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, fill=self.color, outline="black", width=line_width)

    def make_sample(self):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)
        font = _load_font(self.path_font, size=size // 10)
        a, b, c = size // 4, size // 2, size * 3 // 4
        positions = [(x, y) for x in [a, b, c] for y in [a, b, c]]

        radii = dict(small=size // 24, medium=size // 14, large=size // 10)
        keys = random.sample(list(radii.keys()), k=len(radii))
        values = [[0, 2, 6, 8], [1, 3, 5, 7], [4]]
        mapping = {k: v for k, v in zip(keys, values)}
        i_answer = random.choice([0, 2, 6, 8, 1, 3, 5, 7])
        answer = ""

        for k, lst in mapping.items():
            radius = radii[k]
            for i in lst:
                if i == i_answer:
                    answer = k
                    solution_draw_circle_radius = radius
                    self.draw_circle(solution_draw, *positions[i], radius=solution_draw_circle_radius)
                    puzzle_draw.text(
                        positions[i],
                        text="?",
                        font=font,
                        anchor="mm",
                        fill="black",
                    )
                else:
                    self.draw_circle(solution_draw, *positions[i], radius=radius)
                    self.draw_circle(puzzle_draw, *positions[i], radius=radius)

        options = sample_options(answer, list(radii.keys()), k=3)
        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        grid = ["?"] * 9
        for k, lst in mapping.items():
            for i in lst:
                if i != i_answer:
                    grid[i] = k
        grid = grid[::-1]
        answer_location = (
            "at the corner" if i_answer in [0, 2, 6, 8] else "adjacent to the center"
        )

        return (
            dict(
                question="What is the size of the missing part denoted with a question mark?",
                answer=answer,
                options=options,
                caption=f"There are circles arranged in a grid formation with varying sizes in the image. The sizes in the first row are {grid[:3]}, the sizes in the second row are {grid[3:6]}, and the sizes in the third row are {grid[6:9]}.",
                explanation=f"We observe that the circles at the corners are {keys[0]} size, while the circles directly adjacent to the center are {keys[1]} size. Only the center circle is {keys[2]} size. Hence, the pattern is that the circles alternate in size depending on if they are at the corner or adjacent to the center.",
                deduction=f"Based on the pattern that the circles alternate in size depending on if they are at the corner or adjacent to the center, the size of the missing part that is {answer_location} should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


class NumbersTrianglePattern(BaseModel):
    path_font: str = "fonts/OpenSans-Medium.ttf"
    image_size: int = 512
    scale_factor: int = 4
    num_sides: int = 3
    color: str = "#cfe2f3"  # Light blue

    def get_points(self, center: Point, radius: float) -> List[Point]:
        vertices = []
        for i in range(self.num_sides):
            theta = 2 * math.pi / self.num_sides * i
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_circle(self, draw: ImageDraw, point: Point, radius: float, **kwargs):
        x, y = point
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, width=line_width, fill=self.color, **kwargs)

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=_load_font(self.path_font, size=size // 14),
            anchor="mm",
            fill="black",
        )

    def make_sample(self):
        size = self.image_size * self.scale_factor
        puzzle_image = Image.new("RGB", size=(size, size), color="white")
        solution_image = Image.new("RGB", size=(size, size), color="white")
        puzzle_draw = ImageDraw.Draw(puzzle_image)
        solution_draw = ImageDraw.Draw(solution_image)
        center = size / 2, size / 2

        a, b, c, d, e, f = random.sample(list(range(1, 10)), k=6)
        numbers = [a * b, a, b, c * d, c, d, e * f, e, f]
        i_answer = random.randint(0, len(numbers) - 1)
        answer = numbers[i_answer]
        numbers_puzzle = numbers[:]
        numbers_puzzle[i_answer] = "?"

        for i, point in enumerate(self.get_points(center, radius=size / 4)):
            # noinspection PyTypeChecker
            subpoints = self.get_points(point, radius=size / 10)
            puzzle_draw.polygon(subpoints, outline="black", width=size // 200)
            solution_draw.polygon(subpoints, outline="black", width=size // 200)
            for j, sub in enumerate(subpoints):
                # noinspection PyTypeChecker
                self.draw_circle(puzzle_draw, sub, radius=size / 16, outline="black")
                self.draw_circle(
                    solution_draw, sub, radius=size / 16, outline="black"
                )
                # noinspection PyTypeChecker
                self.draw_text(solution_draw, sub, str(numbers[i * 3 + j]))
                self.draw_text(puzzle_draw, sub, str(numbers_puzzle[i * 3 + j]))

        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        groups = [numbers_puzzle[:3][::-1], numbers_puzzle[3:6][::-1], numbers_puzzle[6:][::-1]][::-1]
        instances = [lst for lst in groups if "?" not in lst]
        instances.extend([lst for lst in groups if "?" in lst])
        assert len(instances) == 3

        return (
            dict(
                question="What is the missing number of the part denoted with a question mark?",
                options=[str(o) for o in generate_number_options(int(answer), k=4)],
                answer=answer,
                caption=f"There are three groups of numbers with a triangle arrangement in the image. The first group is {groups[0]}, the second group is {groups[1]}, and the third group is {groups[2]}.",
                explanation=f"We observe that the number {instances[0][2]} is the product of {instances[0][1]} and {instances[0][0]}. Similarly, the number {instances[1][2]} is the product of {instances[1][1]} and {instances[1][0]}. Hence, the pattern is that the rightmost number in each group is the product of the other two numbers.",
                deduction=f"Based on the pattern that the rightmost number in each group is the product of the other two numbers, the missing number of the group {instances[2]} should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


def get_pixels(image: Image, fraction_x: float, fraction_y: float) -> Tuple[int, int]:
    x = round(image.width * fraction_x)
    y = round(image.height * fraction_y)
    return x, y


class VennPattern(BaseModel):
    path_template: str = "templates/puzzle-venn.png"
    path_font: str = "fonts/OpenSans-Medium.ttf"
    rule: str = "{} + {}"
    image_size: int = 512

    def draw_text(self, image: Image, text: str, position: Tuple[int, int]):
        draw = ImageDraw.Draw(image)
        draw.text(
            position,
            text,
            font=_load_font(self.path_font, size=image.width // 16),
            anchor="mm",
            fill="black",
        )

    def make_sample(self):
        base_image = Image.open(self.path_template).convert("RGB")
        solution_image = base_image.copy()
        puzzle_image = base_image.copy()
        a, b, c = random.sample(list(range(1, 10)), k=3)
        ab = eval(self.rule.format(a, b))
        bc = eval(self.rule.format(b, c))

        if random.random() > 0.5:
            answer = a
            a_puzzle = "?"
            c_puzzle = c
        else:
            answer = c
            c_puzzle = "?"
            a_puzzle = a

        # Draw on solution image (true values)
        self.draw_text(solution_image, str(a), get_pixels(solution_image, 0.25, 0.5))
        self.draw_text(solution_image, str(b), get_pixels(solution_image, 0.50, 0.5))
        self.draw_text(solution_image, str(c), get_pixels(solution_image, 0.75, 0.5))
        self.draw_text(solution_image, str(ab), get_pixels(solution_image, 0.38, 0.5))
        self.draw_text(solution_image, str(bc), get_pixels(solution_image, 0.62, 0.5))

        # Draw on puzzle image (with placeholder)
        self.draw_text(puzzle_image, str(a_puzzle), get_pixels(puzzle_image, 0.25, 0.5))
        self.draw_text(puzzle_image, str(b), get_pixels(puzzle_image, 0.50, 0.5))
        self.draw_text(puzzle_image, str(c_puzzle), get_pixels(puzzle_image, 0.75, 0.5))
        self.draw_text(puzzle_image, str(ab), get_pixels(puzzle_image, 0.38, 0.5))
        self.draw_text(puzzle_image, str(bc), get_pixels(puzzle_image, 0.62, 0.5))

        puzzle_image = puzzle_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        solution_image = solution_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        lst = [a_puzzle, b, ab] if c_puzzle == "?" else [b, c_puzzle, bc]
        lst_b = [a_puzzle, b, ab] if c_puzzle != "?" else [b, c_puzzle, bc]

        return (
            dict(
                question="What is the missing number of the part denoted with a question mark?",
                answer=answer,
                options=[str(o) for o in generate_number_options(answer, k=4)],
                caption=f"There are 3 overlapping circles containing the numbers {[a_puzzle, b, c_puzzle]}. The overlapping part between the first and second circle contains the number {ab}. The overlapping part between the second and third circle contains the number {bc}.",
                explanation=f"We observe that the circles with {lst[0]} and {lst[1]} overlap to form the part {lst[2]}, where {lst[0]} + {lst[1]} = {lst[2]}. Hence, the pattern is most likely that the numbers in the overlapping parts are the sum of the numbers in the corresponding circles.",
                deduction=f"Based on the pattern that the numbers in the overlapping parts are the sum of the numbers in the corresponding circles, the missing number of the circle where the overlapping part is {lst_b[-1]} should be {answer}.",
            ),
            puzzle_image,
            solution_image,
        )


def sample_options(answer: str, options: List[str], k: int):
    # Ensure random order and no duplicates
    options = [o for o in options if o != answer]
    assert len(options) + 1 >= k
    options = random.sample(options, k=k - 1)
    options.append(answer)
    assert len(set(options)) == k
    return random.sample(options, k=k)


def generate_number_options(num: int, k: int) -> List[int]:
    # Automatically detect the range and random.sample
    assert num >= 0, "Negative numbers not supported yet"
    options = [10, 100, 1000, 10000, 100000]
    for max_value in options:
        if num <= max_value:
            values = [i for i in range(max_value) if i != num]
            lst = random.sample(values, k=k - 1)
            lst.append(num)
            assert len(set(lst)) == len(lst)
            return random.sample(lst, k=len(lst))
    raise ValueError(f"Range exceeded: {num}, options: {options}")


def convert_image_to_text(image: Image) -> str:
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


def pad_image(img, target_size=(512, 512)) -> Image:
    # Create a new image with a white background
    new_img = Image.new("RGB", target_size, "white")
    
    # Calculate position for centering
    paste_x = (target_size[0] - img.width) // 2
    paste_y = (target_size[1] - img.height) // 2
    
    # Paste the original image onto the new background
    new_img.paste(img, (paste_x, paste_y))
    
    return new_img


def save_visual_puzzle_video(
    puzzle_img,
    solution_img,
    video_path: str,
    fps: int = 16,
) -> int:
    """Save a crossfade video from puzzle to solution image.

    Three phases:
    1. Hold puzzle image
    2. Crossfade puzzle -> solution
    3. Hold solution image

    Returns the total number of frames written.
    """
    puzzle_arr = np.array(puzzle_img.convert("RGB"), dtype=np.uint8)
    solution_arr = np.array(solution_img.convert("RGB"), dtype=np.uint8)

    hold_frames = fps
    fade_frames = fps + fps // 2

    frames: List[np.ndarray] = []
    frames.extend(puzzle_arr.copy() for _ in range(hold_frames))
    for i in range(fade_frames):
        alpha = (i + 1) / fade_frames
        blended = np.clip(
            np.round(puzzle_arr * (1.0 - alpha) + solution_arr * alpha),
            0,
            255,
        ).astype(np.uint8)
        frames.append(blended)
    frames.extend(solution_arr.copy() for _ in range(hold_frames))

    if not encode_rgb_frames_to_mp4(frames, Path(video_path), fps=fps):
        return 0
    return len(frames)


def select_pattern(name: str, **kwargs):

    if name == "color_grid":
        return ColorGridPattern(**kwargs)
    if name == "color_hexagon":
        return ColorHexagonPattern(**kwargs)
    if name == "color_overlap_squares":
        return ColorOverlapSquaresPattern(**kwargs)
    if name == "color_size":
        return ColorSizePattern(**kwargs)
    if name == "polygon_sides_color":
        return PolygonSidesColorPattern(**kwargs)
    if name == "rectangle_height_color":
        return RectangleHeightColorPattern(**kwargs)
    if name == "shape_reflect":
        return ShapeReflectPattern(**kwargs)
    if name == "shape_size_grid":
        return ShapeSizeGridPattern(**kwargs)
    if name == "size_cycle":
        return SizeCyclePattern(**kwargs)
    if name == "size_grid":
        return SizeGridPattern(**kwargs)

    raise KeyError(name)


VIDEOGEN_INSTRUCTION_COMMON = (
    "The video holds on the puzzle frame, then smoothly crossfades to the solution so that only the question-mark region "
    "changes into the correct answer while every other shape stays fixed, and finally remains still on the solved image "
    "with a static camera and no zoom or pan."
)

pattern_instructions = {
    "color_grid": (
        "A centered visual puzzle sits on a clean white square canvas. It shows nine evenly spaced pastel circles with "
        "thin black outlines arranged in a 3x3 grid, where the four corner circles share one color, the four edge-middle "
        "circles share a second color, and the center circle is a third color. One non-center position contains only a "
        "black question mark, which represents the missing circle color that should appear at that exact grid cell."
    ),
    "color_hexagon": (
        "A centered regular hexagon appears on a clean white square canvas. The hexagon is divided from the center into "
        "six triangular wedges with thin black edges, and opposite wedges share the same pastel color chosen from blue, "
        "green, yellow, red, purple, or orange. One wedge is filled light gray and marked with a black question mark, "
        "indicating the missing wedge color."
    ),
    "color_overlap_squares": (
        "A centered composition on a clean white square canvas shows three overlapping rotated squares with thin black "
        "outlines, and the whole cluster is slightly rotated. The three main squares use primary colors while the overlap "
        "regions show the corresponding mixed secondary colors, forming a compact layered arrangement of square and "
        "triangular regions. One side square is replaced by a light gray shape with a black question mark, indicating the "
        "missing square color."
    ),
    "color_size": (
        "A large centered stack of four nested shapes sits on a clean white square canvas. All four shapes share the same "
        "geometry, either circles or regular polygons with black outlines, and they use four shades of one hue so the color "
        "changes steadily lighter or darker as the shapes shrink inward. The smallest inner shape is replaced by a light "
        "gray shape carrying a black question mark, indicating the missing final shade."
    ),
    "polygon_sides_color": (
        "A centered triangular arrangement on a clean white square canvas contains six filled regular polygons with thin "
        "black outlines, laid out as one on the top row, two on the middle row, and three on the bottom row. The polygons "
        "use a pastel palette, and every polygon with the same number of sides shares the same color even though triangles, "
        "quadrilaterals, pentagons, hexagons, heptagons, octagons, or nonagons may appear. One polygon is replaced by a "
        "light gray version with a black question mark, meaning its color is missing."
    ),
    "rectangle_height_color": (
        "A centered row of seven tall rounded rectangles appears on a clean white square canvas. The bars are solid pastel "
        "colors with black outlines, equally spaced left to right, vertically centered, and they repeat three distinct "
        "heights so rectangles of the same height share the same color. The rightmost bar is shown as a light gray rounded "
        "rectangle with a black question mark, indicating the missing color for that height."
    ),
    "shape_reflect": (
        "A clean white square canvas shows a horizontal black line across the middle dividing a top row and a bottom row "
        "of three pale green polygons. Matching shapes above and below the line are vertically mirrored versions of the same "
        "outlined triangle, square, pentagon, or hexagon. One position is replaced by a dotted outline circle with a black "
        "question mark, indicating the missing reflected shape that should appear there."
    ),
    "shape_size_grid": (
        "A centered 3x3 arrangement on a clean white square canvas shows pale green outlined polygons with black borders. "
        "Across one axis the shape family stays constant and across the other axis the size steps from small to medium to "
        "large, using three of triangle, square, pentagon, and hexagon. One cell is empty except for a black question mark "
        "at its center, meaning the missing polygon of the correct shape and size must be inserted there."
    ),
    "size_cycle": (
        "A clean white square canvas shows nine pale yellow circles with thin black outlines arranged in three spiral-like "
        "arms around the center. Along each arm the circles grow from small near the center to medium and large farther "
        "out, and the three arms are evenly spaced around the middle with a slight rotation. One circle position is "
        "replaced by a black question mark, indicating the missing circle size that belongs at that location."
    ),
    "size_grid": (
        "A centered 3x3 grid of pale yellow circles with thin black outlines appears on a clean white square canvas. The "
        "four corner circles share one size, the four edge-middle circles share another size, and the center circle is a "
        "third size. One non-center position contains only a black question mark instead of a circle, indicating the missing "
        "circle size that must be filled in."
    ),
}

def build_visual_puzzle_ti2v_prompt(sample: dict, pattern_name: str) -> str:
    question = str(sample.get("question") or "").strip()
    instruction = pattern_instructions[pattern_name].strip()
    return " ".join(
        part
        for part in (
            "Use the provided visual puzzle image as the starting frame.",
            question,
            instruction,
            VIDEOGEN_INSTRUCTION_COMMON,
        )
        if part
    ).strip()


def build_visual_puzzle_ti2t_prompt(sample: dict) -> str:
    question = str(sample.get("question") or "").strip()
    options = sample.get("options") or []
    option_text = ", ".join(str(option).strip() for option in options if str(option).strip())
    if option_text:
        return f"Use the provided visual puzzle image to solve the task. {question} Choose from: {option_text}. Answer with the final option text only."
    return f"Use the provided visual puzzle image to solve the task. {question} Answer with the final answer only."


def create_data(
    pattern_name: str,
    path: str,
    limit: int = 1,
    seed: int = 42,
    target_size: Tuple[int, int] = (512, 512),
    unique: bool = True
):
    random.seed(seed)
    np.random.seed(seed)

    pattern_dir = Path(path) / pattern_name
    images_dir = pattern_dir / "puzzles"
    solutions_dir = pattern_dir / "solutions"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(solutions_dir, exist_ok=True)

    progress = tqdm(range(limit))

    pattern = select_pattern(pattern_name)
    samples = []
    seen = set()
    question_idx = 0

    count = 0
    while len(samples) < limit:
        count += 1
        if count > limit * 100:
            print(f"Stuck in a loop, exiting. Found {len(samples)} samples.")
            break

        sample, puzzle_image, solution_image = pattern.make_sample()
        image_string = convert_image_to_text(puzzle_image)

        if unique and image_string in seen:
            continue
        seen.add(image_string)

        sample["id"] = f"{pattern_name}-{question_idx:02d}"
        ti2v_prompt = build_visual_puzzle_ti2v_prompt(sample, pattern_name)
        ti2t_prompt = build_visual_puzzle_ti2t_prompt(sample)
        sample["ti2v_prompt"] = ti2v_prompt
        sample["prompt"] = ti2v_prompt
        sample["ti2i_prompt"] = None
        sample["ti2t_prompt"] = ti2t_prompt
        sample["vlm_prompt"] = ti2t_prompt
        sample["ti2ti_prompt"] = None

        puzzle_image = pad_image(puzzle_image, target_size=target_size)
        solution_image = pad_image(solution_image, target_size=target_size)

        image_path = images_dir / f"{question_idx:02d}.png"
        solution_path = solutions_dir / f"{question_idx:02d}.png"
        puzzle_image.save(image_path)
        solution_image.save(solution_path)
        sample["image"] = str(image_path.relative_to(pattern_dir))
        sample["reasoning_image"] = sample["image"]
        sample["solution_image_path"] = str(solution_path.relative_to(pattern_dir))

        samples.append(sample)
        progress.update()
        question_idx += 1

    with open(pattern_dir / "data.json", "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to {pattern_dir / 'data.json'}")


if __name__ == "__main__":
    if Fire is None:
        raise RuntimeError("python-fire is required to use the visual_puzzles CLI entrypoint.")
    Fire()
