import math
import random

from utils.config import *


# --- Classes (Star, Civilization - slightly modified for Env context) ---
class Star:
    """Represents a single star system in the galaxy."""
    def __init__(self, x, y, id):
        self.id = id
        self.x = float(x); self.y = float(y)
        self.claimed_by_id = -1 # Store ID instead of direct reference
        self.resource_bonus = 0
        self.is_destroyed = False
        self.vx = random.uniform(-STAR_MAX_VELOCITY, STAR_MAX_VELOCITY)
        self.vy = random.uniform(-STAR_MAX_VELOCITY, STAR_MAX_VELOCITY)
        if random.random() < STAR_RESOURCE_PROBABILITY:
            self.resource_bonus = random.randint(MIN_STAR_RESOURCE, MAX_STAR_RESOURCE)

    def update_position(self):
        if self.is_destroyed: return
        self.x += self.vx; self.y += self.vy
        if self.x < GALAXY_PADDING: self.x = GALAXY_PADDING; self.vx *= -1
        elif self.x > SCREEN_WIDTH - GALAXY_PADDING: self.x = SCREEN_WIDTH - GALAXY_PADDING; self.vx *= -1
        if self.y < GALAXY_PADDING: self.y = GALAXY_PADDING; self.vy *= -1
        elif self.y > SCREEN_HEIGHT - GALAXY_PADDING: self.y = SCREEN_HEIGHT - GALAXY_PADDING; self.vy *= -1

    def distance_to(self, other_star):
        return math.sqrt((self.x - other_star.x)**2 + (self.y - other_star.y)**2)

    def get_state(self): # Helper to get star state for observation
        return {
            'x': self.x, 'y': self.y, 'id': self.id,
            'claimed_by_id': self.claimed_by_id,
            'resource_bonus': self.resource_bonus,
            'is_destroyed': self.is_destroyed
        }
