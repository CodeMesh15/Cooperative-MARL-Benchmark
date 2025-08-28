import pygame

class Visualizer:
    def __init__(self, grid_size, cell_size=50):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = self.height = grid_size * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Cooperative MARL")
        
        # Colors
        self.colors = {
            "pusher": (0, 0, 255),    # Blue
            "scout": (0, 255, 0),     # Green
            "block": (255, 0, 0),     # Red
            "target": (255, 255, 0),  # Yellow
            "grid": (200, 200, 200),
            "bg": (255, 255, 255)
        }

    def draw(self, env_state):
        """Draws the current state of the environment."""
        self.screen.fill(self.colors["bg"])
        
        # Draw grid
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.colors["grid"], (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.colors["grid"], (0, y), (self.width, y))
            
        # Draw entities
        entities = {"pusher": env_state[0], "scout": env_state[1], 
                    "block": env_state[2], "target": env_state[3]}
        
        for name, pos in entities.items():
            rect = pygame.Rect(pos[1] * self.cell_size, pos[0] * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors[name], rect)
            
        pygame.display.flip()
