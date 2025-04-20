import math
import random

# --- Constants ---
# Screen dimensions (optional for rendering)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
RENDER_FPS = 15 # Lower FPS for rendering during evaluation

# Galaxy properties
NUM_STARS = 150 # Reduced for potentially faster simulation steps
GALAXY_PADDING = 50

# Star properties
STAR_RADIUS = 3
UNCLAIMED_COLOR = (100, 100, 100)
DESTROYED_STAR_COLOR = (255, 0, 0)
RESOURCE_INDICATOR_COLOR = (255, 255, 0)
RESOURCE_INDICATOR_RADIUS = 1
STAR_RESOURCE_PROBABILITY = 0.15
MIN_STAR_RESOURCE = 50
MAX_STAR_RESOURCE = 200
STAR_MAX_VELOCITY = 0.03

# Civilization properties
NUM_CIVILIZATIONS = 5 # Total number of civs, including the agent
CIV_COLORS = [ # Agent (Civ 0) will use the first color
    (0, 255, 255), (255, 100, 0), (0, 200, 100),
    (255, 200, 0), (150, 50, 200), (255, 50, 100),
]

INITIAL_RESOURCES = 500
RESOURCE_INCOME_PER_STAR = 0.1
EXPANSION_COST_FACTOR = 0.5
INITIAL_EXPANSION_RADIUS = 5
MAX_EXPANSION_RADIUS_FACTOR = 0.8

# --- FTL Technology ---
INITIAL_SLOW_EXPANSION_RATE = 0.01
FTL_EXPANSION_RATE = 0.12
FTL_DISCOVERY_BASE_PROB_PER_FRAME = 0.01
FTL_DISCOVERY_SCALING_FACTOR_PER_AGE = 0.0000005
MAX_FTL_DISCOVERY_PROB = 0.005

# --- Civil War ---
CIVIL_WAR_BASE_PROB_PER_FRAME = 0.005
CIVIL_WAR_SCALING_FACTOR_PER_STAR = 0.000001
MAX_CIVIL_WAR_PROB = 0.5
MIN_STARS_FOR_CIVIL_WAR = 10 # Reduced min size
MIN_SPLIT_CIVS = 2
MAX_SPLIT_CIVS = 3
SPLIT_RESOURCE_FRACTION = 0.3

# --- Dark Forest Strike ---
DARK_FOREST_STRIKE_COST = 1500
DARK_FOREST_STRIKE_PROBABILITY_PER_FRAME = 0.0001 # For non-agent civs
MIN_STARS_FOR_STRIKE = 8 # Reduced min size
MIN_RESOURCES_FOR_STRIKE_BUFFER = 500

# --- RL Env Configuration ---
AGENT_ID = 0 # The ID of the civilization controlled by the RL agent
MAX_STEPS_PER_EPISODE = 2000 # Set a max episode length
N_NEAREST_STARS_OBS = 10 # Number of nearest stars to include in observation

# --- Visualization ---
DASH_LENGTH = 4
GAP_LENGTH = 3

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
