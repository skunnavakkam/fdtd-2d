import numpy as np
from PIL import Image, ImageDraw
import math

class RegionDrawer:
    def __init__(self, width: int, height: int):
        """Initialize a blank canvas of given dimensions."""
        self.width = width
        self.height = height
        self.image = Image.new('L', (width, height), 255)  # White background
        self.draw = ImageDraw.Draw(self.image)

    def draw_waveguide(self, start: tuple[int,int], end: tuple[int,int], width: int):
        """Draw a straight waveguide between two points with given width."""
        self.draw.line([start, end], fill=0, width=width)

    def draw_ring_resonator(self, center: tuple[int,int], radius: int, ring_width: int):
        """Draw a ring resonator centered at given point."""
        # Draw outer circle
        outer_bbox = [
            center[0] - radius - ring_width//2,
            center[1] - radius - ring_width//2,
            center[0] + radius + ring_width//2,
            center[1] + radius + ring_width//2
        ]
        self.draw.ellipse(outer_bbox, outline=0, width=ring_width)

    def draw_sphere(self, center: tuple[int,int], radius: int, sphere_width: int):
        """Draw a sphere centered at given point."""
        # Draw outer circle
        outer_bbox = [
            center[0] - radius - sphere_width//2,
            center[1] - radius - sphere_width//2,
            center[0] + radius + sphere_width//2, 
            center[1] + radius + sphere_width//2
        ]
        self.draw.ellipse(outer_bbox, fill="black")

    def draw_curved_waveguide(self, start: tuple[int,int], end: tuple[int,int], 
                            control_point: tuple[int,int], width: int):
        """Draw a curved waveguide using a quadratic Bezier curve."""
        # Generate points along the Bezier curve
        points = []
        for t in np.linspace(0, 1, 100):
            x = (1-t)**2 * start[0] + 2*(1-t)*t * control_point[0] + t**2 * end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t * control_point[1] + t**2 * end[1]
            points.append((x, y))
        
        # Draw the curve
        self.draw.line(points, fill=0, width=width)

    def draw_directional_coupler(self, start: tuple[int,int], length: int, 
                               gap: int, waveguide_width: int):
        """Draw a directional coupler (two parallel waveguides)."""
        y_offset = gap//2 + waveguide_width//2
        # Top waveguide
        self.draw_waveguide(
            (start[0], start[1] - y_offset),
            (start[0] + length, start[1] - y_offset),
            waveguide_width
        )
        # Bottom waveguide
        self.draw_waveguide(
            (start[0], start[1] + y_offset),
            (start[0] + length, start[1] + y_offset),
            waveguide_width
        )

    def save(self, filename: str):
        """Save the drawn structure to a file."""
        self.image.save(filename)

# Example usage:
if __name__ == "__main__":
    # Create a 500x500 canvas
    drawer = RegionDrawer(5000, 5000)
    
    # Draw the ring
    # drawer.draw_ring_resonator(center, radius, ring_width)
    drawer.draw_sphere((5000// 3, 5000//2), 5000//5, 20)
    
    # Draw a bus waveguide
    
    # Save the result
    drawer.save("high_res_sphere.png")
