import pygame
import random
import math
import sys
import time
import numpy as np

# RL Environment Imports
import gymnasium as gym
from gymnasium import spaces

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
def generate_random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
if NUM_CIVILIZATIONS > len(CIV_COLORS):
    for _ in range(NUM_CIVILIZATIONS - len(CIV_COLORS)):
        CIV_COLORS.append(generate_random_color())

INITIAL_RESOURCES = 500
RESOURCE_INCOME_PER_STAR = 0.1
EXPANSION_COST_FACTOR = 0.5
INITIAL_EXPANSION_RADIUS = 5
MAX_EXPANSION_RADIUS_FACTOR = 0.8

# --- FTL Technology ---
INITIAL_SLOW_EXPANSION_RATE = 0.01
FTL_EXPANSION_RATE = 0.12
FTL_DISCOVERY_BASE_PROB_PER_FRAME = 0.00001
FTL_DISCOVERY_SCALING_FACTOR_PER_AGE = 0.0000005
MAX_FTL_DISCOVERY_PROB = 0.005

# --- Civil War ---
CIVIL_WAR_BASE_PROB_PER_FRAME = 0.000005
CIVIL_WAR_SCALING_FACTOR_PER_STAR = 0.000001
MAX_CIVIL_WAR_PROB = 0.002
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

class Civilization:
    """Represents a civilization within the RL environment."""
    def __init__(self, id, home_star_id, color, stars_dict, initial_resources=INITIAL_RESOURCES, initial_ftl=False, initial_age=0):
        self.id = id
        self.home_star_id = home_star_id
        self.color = color
        self.controlled_star_ids = {home_star_id}
        self.expansion_radius = INITIAL_EXPANSION_RADIUS
        self.max_expansion_radius = math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2) * MAX_EXPANSION_RADIUS_FACTOR
        self.is_eliminated = False
        self.resources = float(initial_resources)
        self.age = initial_age
        self.has_ftl = initial_ftl
        self.last_action = 0 # Track agent's last action if needed

        home_star = stars_dict[home_star_id]
        if home_star.claimed_by_id != self.id:
             home_star.claimed_by_id = self.id
             if home_star.resource_bonus > 0:
                 self.resources += home_star.resource_bonus
                 home_star.resource_bonus = 0

    def get_strength(self, stars_dict):
        """Calculates valid strength based on current star dictionary."""
        valid_stars = {sid for sid in self.controlled_star_ids if sid in stars_dict and not stars_dict[sid].is_destroyed}
        self.controlled_star_ids = valid_stars # Clean up invalid IDs
        return len(self.controlled_star_ids)

    def update_resources(self, stars_dict):
        if not self.is_eliminated:
            valid_star_count = len([sid for sid in self.controlled_star_ids if sid in stars_dict and not stars_dict[sid].is_destroyed])
            income = valid_star_count * RESOURCE_INCOME_PER_STAR
            self.resources += income

    def check_for_events(self, stars_dict):
        """Checks for FTL and Civil War. Returns True if civil war triggered."""
        if self.is_eliminated: return False
        self.age += 1
        # FTL Check
        if not self.has_ftl:
            ftl_prob = min(FTL_DISCOVERY_BASE_PROB_PER_FRAME + self.age * FTL_DISCOVERY_SCALING_FACTOR_PER_AGE, MAX_FTL_DISCOVERY_PROB)
            if random.random() < ftl_prob: self.has_ftl = True; return False, True # No civil war, FTL discovered
        # Civil War Check
        num_stars = self.get_strength(stars_dict)
        if num_stars >= MIN_STARS_FOR_CIVIL_WAR:
            civil_war_prob = min(CIVIL_WAR_BASE_PROB_PER_FRAME + num_stars * CIVIL_WAR_SCALING_FACTOR_PER_STAR, MAX_CIVIL_WAR_PROB)
            if random.random() < civil_war_prob: return True, self.has_ftl # Signal civil war, return current FTL status
        return False, self.has_ftl # No civil war, return current FTL status

    def expand(self, stars_dict):
        """ Expands influence and handles conflicts. Returns claimed_ids, conquered_ids, lost_ids sets."""
        if self.is_eliminated: return set(), set(), set()
        self.update_resources(stars_dict) # Update resources first

        claimed_this_step = set()
        conquered_this_step = set()
        # Lost stars are handled by defender's perspective or strikes

        current_expansion_rate = FTL_EXPANSION_RATE if self.has_ftl else INITIAL_SLOW_EXPANSION_RATE
        if self.expansion_radius < self.max_expansion_radius:
             self.expansion_radius += current_expansion_rate

        potential_claims_with_cost = {}
        current_controlled_stars = [stars_dict[sid] for sid in self.controlled_star_ids if sid in stars_dict and not stars_dict[sid].is_destroyed]
        if not current_controlled_stars:
             self.is_eliminated = True
             return set(), set(), set()

        for controlled_star in current_controlled_stars:
            for star_id, potential_star in stars_dict.items():
                if potential_star.claimed_by_id == self.id or potential_star.is_destroyed: continue
                dist = controlled_star.distance_to(potential_star)
                if dist <= self.expansion_radius:
                    cost = dist * EXPANSION_COST_FACTOR
                    target_star_id = potential_star.id
                    if target_star_id not in potential_claims_with_cost or cost < potential_claims_with_cost[target_star_id][0]:
                         potential_claims_with_cost[target_star_id] = (cost, controlled_star)

        sorted_potential_claims = sorted(potential_claims_with_cost.items(), key=lambda item: item[1][0])

        for target_star_id, (cost, _) in sorted_potential_claims:
            if self.resources < cost: continue
            target_star = stars_dict.get(target_star_id)
            if not target_star or target_star.is_destroyed: continue

            claimed_successfully = False
            attacked_successfully = False

            if target_star.claimed_by_id == -1: # Claim unclaimed (-1 indicates unclaimed)
                target_star.claimed_by_id = self.id
                self.controlled_star_ids.add(target_star.id)
                claimed_successfully = True
                claimed_this_step.add(target_star.id)
            else: # Attack claimed star
                # Need access to the defender civ object - this requires the Env to manage civs
                # For now, simplify: assume strength comparison happens elsewhere or is simplified
                # Let's simulate a simple strength check here for non-agent civs
                # Note: This part needs careful handling in the Env step function
                # For simplicity in this class structure, we'll just assume a random chance based on cost
                # A proper implementation needs the Env to resolve conflicts globally.
                # Placeholder: Random chance to conquer if affordable
                if random.random() < 0.3: # Simplified conflict chance
                     # We don't know the defender here, assume conquest happens
                     target_star.claimed_by_id = self.id
                     self.controlled_star_ids.add(target_star.id)
                     attacked_successfully = True
                     conquered_this_step.add(target_star.id)


            if claimed_successfully or attacked_successfully:
                self.resources -= cost
                if target_star.resource_bonus > 0:
                    self.resources += target_star.resource_bonus
                    target_star.resource_bonus = 0

        # Lost stars need to be determined globally in the Env step
        return claimed_this_step, conquered_this_step, set() # Return empty set for lost stars here

    def consider_dark_forest_strike(self, stars_dict):
        """Checks eligibility and probability for non-agent strike."""
        if self.is_eliminated: return False, None

        can_afford = self.resources >= DARK_FOREST_STRIKE_COST + MIN_RESOURCES_FOR_STRIKE_BUFFER
        is_large_enough = self.get_strength(stars_dict) >= MIN_STARS_FOR_STRIKE
        meets_probability = random.random() < DARK_FOREST_STRIKE_PROBABILITY_PER_FRAME

        if can_afford and is_large_enough and meets_probability:
            potential_targets = [star for star_id, star in stars_dict.items()
                                 if star.claimed_by_id != self.id and not star.is_destroyed]
            if potential_targets:
                target_star = random.choice(potential_targets)
                return True, target_star # Signal to execute strike
        return False, None


# --- RL Environment ---

class GalaxySimEnv(gym.Env):
    """Custom Environment for Galaxy Simulation that follows gym interface."""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': RENDER_FPS}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Define action space: 0 = Passive Expand, 1 = Attempt Dark Forest Strike
        self.action_space = spaces.Discrete(2)

        # Define observation space (example structure, adjust sizes and bounds)
        # Internal State (5 floats) + N Nearest Stars (N * 6 floats/bools) + Global Context (3 floats)
        # N_NEAREST_STARS_OBS = 10
        # Star Obs: rel_x, rel_y, claimed_unclaimed, claimed_other, claimed_destroyed, resource_bonus, distance
        star_obs_size = 7
        obs_size = 5 + N_NEAREST_STARS_OBS * star_obs_size + 3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Simulation state variables (initialized in reset)
        self.stars_list = []
        self.stars_dict = {}
        self.civilizations = {} # Dict: id -> Civilization object
        self.current_step = 0
        self.agent_id = AGENT_ID
        self.next_civ_id = 0
        self.last_agent_resources = INITIAL_RESOURCES

        # Pygame font (optional, for rendering)
        if self.render_mode == 'human':
             pygame.init()
             pygame.font.init()
             self.font_small = pygame.font.Font(None, 18)
             self.font_large = pygame.font.Font(None, 20)


    def _get_obs(self):
        """Extracts the current observation for the agent."""
        agent_civ = self.civilizations.get(self.agent_id)
        if not agent_civ or agent_civ.is_eliminated:
            # Return a zero observation if agent is eliminated
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # 1. Agent's Internal State (Normalize these values)
        obs_internal = np.array([
            np.clip(agent_civ.resources / (INITIAL_RESOURCES * 10), -1, 1), # Example normalization
            np.clip(agent_civ.age / MAX_STEPS_PER_EPISODE, -1, 1),
            1.0 if agent_civ.has_ftl else 0.0,
            np.clip(agent_civ.expansion_radius / agent_civ.max_expansion_radius, -1, 1),
            np.clip(agent_civ.get_strength(self.stars_dict) / NUM_STARS, -1, 1)
        ], dtype=np.float32)

        # 2. Local Periphery (N nearest non-self, non-destroyed stars)
        agent_stars = [self.stars_dict[sid] for sid in agent_civ.controlled_star_ids if sid in self.stars_dict]
        if not agent_stars: # Handle case where agent has no stars (e.g., just eliminated)
            center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 # Default center
        else:
            center_x = sum(s.x for s in agent_stars) / len(agent_stars)
            center_y = sum(s.y for s in agent_stars) / len(agent_stars)

        nearby_stars = []
        for star_id, star in self.stars_dict.items():
            if star.claimed_by_id != self.agent_id and not star.is_destroyed:
                dist = math.sqrt((star.x - center_x)**2 + (star.y - center_y)**2)
                nearby_stars.append((dist, star))

        nearby_stars.sort(key=lambda x: x[0]) # Sort by distance
        obs_periphery = np.zeros(N_NEAREST_STARS_OBS * 7, dtype=np.float32) # 7 features per star

        max_dist_norm = math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2) # For normalization

        for i in range(min(N_NEAREST_STARS_OBS, len(nearby_stars))):
            dist, star = nearby_stars[i]
            rel_x = (star.x - center_x) / (SCREEN_WIDTH / 2) # Normalize relative pos
            rel_y = (star.y - center_y) / (SCREEN_HEIGHT / 2)
            is_unclaimed = 1.0 if star.claimed_by_id == -1 else 0.0
            is_other_civ = 1.0 if star.claimed_by_id != -1 and star.claimed_by_id != self.agent_id else 0.0
            # is_destroyed is always 0 here because we filter them out
            resource_bonus = star.resource_bonus / MAX_STAR_RESOURCE # Normalize
            norm_dist = dist / max_dist_norm

            start_idx = i * 7
            obs_periphery[start_idx:start_idx+7] = [
                np.clip(rel_x, -1, 1), np.clip(rel_y, -1, 1),
                is_unclaimed, is_other_civ, 0.0, # 0.0 for is_destroyed placeholder
                np.clip(resource_bonus, 0, 1),
                np.clip(norm_dist, 0, 1)
            ]

        # 3. Global Context (Normalize these values)
        active_rivals = sum(1 for cid, civ in self.civilizations.items() if cid != self.agent_id and not civ.is_eliminated)
        total_claimed = sum(len(civ.controlled_star_ids) for cid, civ in self.civilizations.items() if not civ.is_eliminated)
        total_destroyed = sum(1 for star in self.stars_list if star.is_destroyed)

        obs_global = np.array([
            np.clip(active_rivals / (NUM_CIVILIZATIONS -1 + 1e-6), 0, 1), # Avoid div by zero if NUM_CIVS=1
            np.clip(total_claimed / NUM_STARS, 0, 1),
            np.clip(total_destroyed / NUM_STARS, 0, 1)
        ], dtype=np.float32)

        # Concatenate all parts
        observation = np.concatenate([obs_internal, obs_periphery, obs_global]).astype(np.float32)
        # Ensure the observation fits the defined space
        if observation.shape != self.observation_space.shape:
             # This indicates an error in constructing the observation vector
             print(f"Error: Observation shape mismatch. Expected {self.observation_space.shape}, got {observation.shape}")
             # Fallback to zeros, but this needs debugging
             observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        return observation


    def _get_info(self):
        """Returns auxiliary information (optional)."""
        agent_civ = self.civilizations.get(self.agent_id)
        if not agent_civ: return {}
        return {
            "agent_resources": agent_civ.resources,
            "agent_strength": agent_civ.get_strength(self.stars_dict),
            "agent_age": agent_civ.age,
            "agent_ftl": agent_civ.has_ftl,
            "active_civilizations": sum(1 for civ in self.civilizations.values() if not civ.is_eliminated),
            "step": self.current_step,
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed) # Important for reproducibility

        # Reset simulation state
        self.stars_list = []
        self.stars_dict = {}
        self.civilizations = {}
        self.current_step = 0
        self.next_civ_id = 0

        # Initialize stars
        for i in range(NUM_STARS):
            x = self.np_random.integers(GALAXY_PADDING, SCREEN_WIDTH - GALAXY_PADDING)
            y = self.np_random.integers(GALAXY_PADDING, SCREEN_HEIGHT - GALAXY_PADDING)
            star = Star(x, y, i)
            self.stars_list.append(star)
            self.stars_dict[i] = star

        # Initialize civilizations
        available_start_indices = list(range(len(self.stars_list)))
        self.np_random.shuffle(available_start_indices)

        if len(available_start_indices) < NUM_CIVILIZATIONS:
             raise ValueError("Not enough stars for the required number of civilizations.")

        for i in range(NUM_CIVILIZATIONS):
            start_index = available_start_indices.pop()
            home_star = self.stars_list[start_index]
            while home_star.is_destroyed: # Ensure start star isn't destroyed
                 if not available_start_indices: raise ValueError("Not enough valid stars.")
                 start_index = available_start_indices.pop()
                 home_star = self.stars_list[start_index]

            civ_color = CIV_COLORS[i % len(CIV_COLORS)]
            # Agent uses ID 0
            civ_id = self.next_civ_id
            civilization = Civilization(id=civ_id, home_star_id=home_star.id, color=civ_color, stars_dict=self.stars_dict)
            self.civilizations[civ_id] = civilization
            self.next_civ_id += 1

        # Get initial observation and info
        agent_civ = self.civilizations.get(self.agent_id)
        self.last_agent_resources = agent_civ.resources if agent_civ else 0
        observation = self._get_obs()
        info = self._get_info()

        # Reset rendering if needed
        if self.render_mode == "human":
            self._render_init()

        return observation, info

    def step(self, action):
        """Runs one timestep of the environment's dynamics."""
        if self.agent_id not in self.civilizations or self.civilizations[self.agent_id].is_eliminated:
             # If agent is already eliminated, return immediately
             obs = self._get_obs() # Should be zeros
             return obs, 0, True, False, self._get_info() # obs, reward, terminated, truncated, info

        terminated = False
        truncated = False
        reward = 0.0
        agent_civ = self.civilizations[self.agent_id]
        agent_civ.last_action = action # Store agent action

        # --- Simulation Step ---
        civs_to_split = []
        civs_triggering_strike = {} # civ_id -> target_star
        events_this_step = {'ftl_discovered': False, 'agent_strike_executed': False, 'stars_claimed': 0, 'stars_lost': 0}
        initial_agent_stars = agent_civ.controlled_star_ids.copy()


        # 1. Update Star Positions
        for star in self.stars_list:
            star.update_position()

        # 2. Process Civilization Logic (Events, Expansion, Strikes)
        # Need to iterate safely as list might change due to civil war
        current_civ_ids = list(self.civilizations.keys())
        random.shuffle(current_civ_ids) # Randomize update order

        for civ_id in current_civ_ids:
            civ = self.civilizations.get(civ_id)
            if not civ or civ.is_eliminated: continue

            # Check Events (FTL, Civil War)
            civil_war_triggered, ftl_status_changed = civ.check_for_events(self.stars_dict)
            if civil_war_triggered:
                civs_to_split.append(civ)
            if civ_id == self.agent_id and ftl_status_changed and civ.has_ftl:
                 events_this_step['ftl_discovered'] = True

            # Expansion & Conflict (Needs careful implementation for global conflict resolution)
            # --- Global Conflict Resolution (Simplified) ---
            # Store intended claims/attacks first, then resolve
            # This part is complex. For now, use the simplified logic within civ.expand
            # A better way: each civ proposes expansions, Env resolves conflicts based on strength
            claimed, conquered, _ = civ.expand(self.stars_dict) # expand handles resources/radius update
            if civ_id == self.agent_id:
                 events_this_step['stars_claimed'] += len(claimed) + len(conquered)


            # Dark Forest Strike Consideration
            if civ_id == self.agent_id:
                # Agent action determines strike attempt
                if action == 1: # Action 1: Attempt Strike
                    can_afford = civ.resources >= DARK_FOREST_STRIKE_COST + MIN_RESOURCES_FOR_STRIKE_BUFFER
                    is_large_enough = civ.get_strength(self.stars_dict) >= MIN_STARS_FOR_STRIKE
                    if can_afford and is_large_enough:
                         potential_targets = [star for star_id, star in self.stars_dict.items()
                                             if star.claimed_by_id != civ.id and not star.is_destroyed]
                         if potential_targets:
                             target_star = random.choice(potential_targets)
                             civs_triggering_strike[civ_id] = target_star # Schedule strike
                             events_this_step['agent_strike_executed'] = True # Mark attempt leading to execution
            else:
                # Non-agent civs use their random check
                strike_triggered, target_star = civ.consider_dark_forest_strike(self.stars_dict)
                if strike_triggered:
                    civs_triggering_strike[civ_id] = target_star


        # 3. Execute Dark Forest Strikes (after all civs moved/expanded)
        for striker_id, target_star in civs_triggering_strike.items():
             striker_civ = self.civilizations.get(striker_id)
             if striker_civ and not striker_civ.is_eliminated and target_star and not target_star.is_destroyed:
                 # Ensure striker can still afford it (might have lost resources)
                 if striker_civ.resources >= DARK_FOREST_STRIKE_COST:
                     self._execute_dark_forest_strike(striker_civ, target_star)
                 elif striker_id == self.agent_id:
                      events_this_step['agent_strike_executed'] = False # Failed due to cost change


        # 4. Handle Civil Wars
        if civs_to_split:
            new_civs_added = []
            for civ_to_split in civs_to_split:
                 if not civ_to_split.is_eliminated:
                     self.next_civ_id = self._handle_civil_war(civ_to_split, new_civs_added, self.next_civ_id)
            # Add newly created civs to the main dictionary
            for new_civ in new_civs_added:
                 self.civilizations[new_civ.id] = new_civ


        # 5. Final State Checks & Reward Calculation for Agent
        agent_civ = self.civilizations.get(self.agent_id) # Re-get agent civ

        if not agent_civ or agent_civ.is_eliminated:
            terminated = True
            reward += -100.0 # r_elimination
        else:
            # Check for stars lost by the agent this step
            current_agent_stars = agent_civ.controlled_star_ids
            lost_stars_count = len(initial_agent_stars - current_agent_stars)
            events_this_step['stars_lost'] = lost_stars_count

            # Calculate reward components
            reward += 0.01 # r_survival
            resource_change = agent_civ.resources - self.last_agent_resources
            reward += resource_change * 0.001 # r_resource_change
            reward += events_this_step['stars_claimed'] * 0.5 # r_claim
            reward += events_this_step['stars_lost'] * -0.5 # r_loss
            if events_this_step['ftl_discovered']: reward += 10.0 # r_ftl
            if events_this_step['agent_strike_executed']: reward += -0.1 # r_strike_cost

            self.last_agent_resources = agent_civ.resources # Update for next step


        # 6. Check Truncation (max steps)
        self.current_step += 1
        if self.current_step >= MAX_STEPS_PER_EPISODE:
            truncated = True

        # 7. Get Next Observation & Info
        observation = self._get_obs()
        info = self._get_info()

        # 8. Render (optional)
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info


    # --- Simulation Logic Helpers (moved inside Env) ---

    def _execute_dark_forest_strike(self, striking_civ, target_star):
        """ Executes the effects of a Dark Forest strike. """
        if striking_civ.resources < DARK_FOREST_STRIKE_COST: return # Double check cost
        striking_civ.resources -= DARK_FOREST_STRIKE_COST
        target_star.is_destroyed = True
        target_star.resource_bonus = 0

        victim_civ_id = target_star.claimed_by_id
        target_star.claimed_by_id = -1 # Star is no longer claimed

        if victim_civ_id != -1 and victim_civ_id != striking_civ.id:
            victim_civ = self.civilizations.get(victim_civ_id)
            if victim_civ:
                victim_civ.controlled_star_ids.discard(target_star.id)
                # Check elimination
                if victim_civ.get_strength(self.stars_dict) == 0:
                    victim_civ.is_eliminated = True


    def _handle_civil_war(self, original_civ, new_civs_list, next_civ_id):
        """ Handles the splitting of a civilization. Adds new civs to new_civs_list. """
        if original_civ.is_eliminated: return next_civ_id

        valid_star_ids = [sid for sid in original_civ.controlled_star_ids if sid in self.stars_dict and not self.stars_dict[sid].is_destroyed]

        if len(valid_star_ids) < MIN_SPLIT_CIVS : return next_civ_id

        original_civ.is_eliminated = True
        num_splits = random.randint(MIN_SPLIT_CIVS, min(MAX_SPLIT_CIVS, len(valid_star_ids)))
        random.shuffle(valid_star_ids)
        resources_per_split = original_civ.resources * SPLIT_RESOURCE_FRACTION
        stars_per_split = len(valid_star_ids) // num_splits
        extra_stars = len(valid_star_ids) % num_splits
        star_index = 0

        for i in range(num_splits):
            num_stars_for_this_split = stars_per_split + (1 if i < extra_stars else 0)
            if num_stars_for_this_split == 0: continue
            split_star_ids = set(valid_star_ids[star_index : star_index + num_stars_for_this_split])
            star_index += num_stars_for_this_split
            if not split_star_ids: continue

            new_home_star = self.stars_dict[next(iter(split_star_ids))]

            new_civ = Civilization(
                id=next_civ_id, home_star_id=new_home_star.id, color=generate_random_color(),
                stars_dict=self.stars_dict, # Pass stars_dict
                initial_resources=resources_per_split, initial_ftl=False, initial_age=0
            )
            new_civ.controlled_star_ids = split_star_ids # Assign correct stars

            # Update star ownership globally
            for star_id in split_star_ids:
                 if star_id in self.stars_dict: self.stars_dict[star_id].claimed_by_id = new_civ.id

            new_civs_list.append(new_civ) # Add to temporary list
            next_civ_id += 1
        return next_civ_id


    # --- Rendering Methods ---
    def _render_init(self):
         if self.screen is None:
             pygame.init()
             pygame.display.init()
             self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
             pygame.display.set_caption("Galaxy Sim RL Environment")
         if self.clock is None:
             self.clock = pygame.time.Clock()
         if not hasattr(self, 'font_small'): # Initialize fonts if not present
              pygame.font.init()
              self.font_small = pygame.font.Font(None, 18)
              self.font_large = pygame.font.Font(None, 20)


    def render(self):
        """Renders the current environment state using Pygame."""
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return
        if self.render_mode != 'human':
             # Other modes like 'rgb_array' could be implemented here
             return

        self._render_init() # Ensure Pygame is initialized

        # --- Drawing ---
        self.screen.fill((10, 10, 20)) # Dark background

        # Draw all stars
        for star in self.stars_list:
            self._draw_star(star) # Use helper method

        # Draw simulation info
        y_offset = 10
        total_claimed = 0
        active_civ_count = 0
        destroyed_star_count = sum(1 for star in self.stars_list if star.is_destroyed)

        # Sort civs for consistent display order
        sorted_civ_ids = sorted(self.civilizations.keys())

        for civ_id in sorted_civ_ids:
            civ = self.civilizations[civ_id]
            if not civ.is_eliminated:
                is_agent = "(A)" if civ_id == self.agent_id else ""
                strength = civ.get_strength(self.stars_dict)
                res_text = f"{civ.resources:.0f}" if civ.resources < 10000 else f"{civ.resources/1000:.1f}k"
                ftl_status = " (FTL)" if civ.has_ftl else ""
                text = f"Civ {civ.id}{is_agent}{ftl_status}: {strength} stars | Res: {res_text}"
                text_surface = self.font_small.render(text, True, civ.color)
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 16
                total_claimed += strength
                active_civ_count += 1

        # Draw summary info
        unclaimed_count = NUM_STARS - total_claimed - destroyed_star_count
        total_civ_count = len(self.civilizations)
        summary_text = f"Active: {active_civ_count}/{total_civ_count} | Unclaimed: {max(0, unclaimed_count)} | Destroyed: {destroyed_star_count} | Step: {self.current_step}"
        summary_surface = self.font_large.render(summary_text, True, (255, 255, 255))
        self.screen.blit(summary_surface, (SCREEN_WIDTH - summary_surface.get_width() - 10, 10))

        pygame.event.pump() # Process internal Pygame events
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])


    def _draw_star(self, star):
         """ Helper to draw a single star. """
         draw_x = int(star.x); draw_y = int(star.y)
         if star.is_destroyed:
             pygame.draw.line(self.screen, DESTROYED_STAR_COLOR, (draw_x - STAR_RADIUS, draw_y - STAR_RADIUS), (draw_x + STAR_RADIUS, draw_y + STAR_RADIUS), 1)
             pygame.draw.line(self.screen, DESTROYED_STAR_COLOR, (draw_x - STAR_RADIUS, draw_y + STAR_RADIUS), (draw_x + STAR_RADIUS, draw_y - STAR_RADIUS), 1)
         else:
             color = UNCLAIMED_COLOR
             if star.claimed_by_id != -1 and star.claimed_by_id in self.civilizations:
                 color = self.civilizations[star.claimed_by_id].color
             pygame.draw.circle(self.screen, color, (draw_x, draw_y), STAR_RADIUS)
             if star.resource_bonus > 0:
                 pygame.draw.circle(self.screen, RESOURCE_INDICATOR_COLOR, (draw_x, draw_y), STAR_RADIUS + 1, 1)


    def close(self):
        """Closes the environment and cleans up resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


# --- Example Usage (Requires stable-baselines3) ---
if __name__ == '__main__':
    # Example of how to use the environment

    print("Creating and checking the environment...")
    # Create the environment with human rendering enabled
    env = GalaxySimEnv(render_mode='human')

    # It's recommended to wrap the environment with checks during development
    # from stable_baselines3.common.env_checker import check_env
    # check_env(env) # This will run checks on your custom environment

    print("Environment created. Running random actions for a few steps...")

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    # Run for a limited number of steps with random actions
    max_random_steps = 100
    while step_count < max_random_steps and not terminated and not truncated:
        action = env.action_space.sample() # Choose a random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        # Render the environment state (if render_mode='human')
        env.render()
        # Add a small delay to make rendering watchable
        # time.sleep(0.05)
        if terminated or truncated:
            print(f"Episode finished after {step_count} steps. Terminated: {terminated}, Truncated: {truncated}")
            print(f"Total Reward: {total_reward}")
            # Optionally reset for another short run
            # obs, info = env.reset()
            # terminated = False
            # truncated = False
            # total_reward = 0
            # step_count = 0

    print("Finished random action example.")
    env.close()


    # --- Placeholder for actual training ---
    print("\n--- Training Placeholder ---")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        # Instantiate the env
        # For training, typically don't render: render_mode=None
        # Use make_vec_env for parallel environments (optional but faster)
        # vec_env = make_vec_env(lambda: GalaxySimEnv(render_mode=None), n_envs=4)
        train_env = GalaxySimEnv(render_mode=None)

        # Define the model
        model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./galaxy_ppo_tensorboard/")

        # Train the model
        print("Starting PPO training (this will take a long time)...")
        # Increase total_timesteps significantly for actual training (e.g., 1e6 or more)
        model.learn(total_timesteps=10000, log_interval=10) # Train for a small number of steps as demo
        model.save("ppo_galaxy_sim")
        print("Training finished and model saved.")

        # --- Evaluate the trained agent ---
        print("\n--- Evaluation ---")
        del model # remove to demonstrate saving and loading
        model = PPO.load("ppo_galaxy_sim")
        eval_env = GalaxySimEnv(render_mode='human') # Render during evaluation
        obs, info = eval_env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            eval_env.render()
            if terminated or truncated: print("Evaluation finished.")
        eval_env.close()

    except ImportError:
         print("\nStable Baselines3 not found. Skipping training example.")
         print("Install it using: pip install stable-baselines3[extra]")
    except Exception as e:
         print(f"\nAn error occurred during the training/evaluation placeholder: {e}")

