import pygame
import random
import math
import sys
import time # For potential seeding or timing, if needed

# --- Constants ---
# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Galaxy properties
NUM_STARS = 450
GALAXY_PADDING = 50 # Stars bounce off this boundary

# Star properties
STAR_RADIUS = 3
UNCLAIMED_COLOR = (100, 100, 100)
DESTROYED_STAR_COLOR = (255, 0, 0) # Red for destroyed stars
RESOURCE_INDICATOR_COLOR = (255, 255, 0)
RESOURCE_INDICATOR_RADIUS = 1
STAR_RESOURCE_PROBABILITY = 0.15
MIN_STAR_RESOURCE = 50
MAX_STAR_RESOURCE = 200
# --- Star Movement ---
STAR_MAX_VELOCITY = 0.03 # Max speed in pixels/frame (keep very low)

# Civilization properties
NUM_CIVILIZATIONS = 5
CIV_COLORS = [
    (0, 150, 255), (255, 100, 0), (0, 200, 100),
    (255, 200, 0), (150, 50, 200), (255, 50, 100),
]
# Function to generate random colors for split-off civs
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
CIVIL_WAR_BASE_PROB_PER_FRAME = 0.05
CIVIL_WAR_SCALING_FACTOR_PER_STAR = 0.000001
MAX_CIVIL_WAR_PROB = 0.002
MIN_STARS_FOR_CIVIL_WAR = 15
MIN_SPLIT_CIVS = 2
MAX_SPLIT_CIVS = 4
SPLIT_RESOURCE_FRACTION = 0.3

# --- Dark Forest Strike ---
DARK_FOREST_STRIKE_COST = 1500 # High resource cost
DARK_FOREST_STRIKE_PROBABILITY_PER_FRAME = 0.0001 # Small chance per frame for eligible civs
MIN_STARS_FOR_STRIKE = 10 # Minimum size to consider striking
MIN_RESOURCES_FOR_STRIKE_BUFFER = 500 # Minimum resources needed *after* paying cost

# Simulation properties
FPS = 30

# --- Classes ---

class Star:
    """Represents a single star system in the galaxy."""
    def __init__(self, x, y, id):
        self.id = id
        self.x = float(x) # Use float for position to handle small velocities
        self.y = float(y)
        self.claimed_by = None
        self.resource_bonus = 0
        self.is_destroyed = False # Flag for Dark Forest strikes

        # Initialize velocity for star movement
        self.vx = random.uniform(-STAR_MAX_VELOCITY, STAR_MAX_VELOCITY)
        self.vy = random.uniform(-STAR_MAX_VELOCITY, STAR_MAX_VELOCITY)

        if random.random() < STAR_RESOURCE_PROBABILITY:
            self.resource_bonus = random.randint(MIN_STAR_RESOURCE, MAX_STAR_RESOURCE)

    def update_position(self):
        """Updates star position based on velocity and handles bouncing."""
        if self.is_destroyed: # Destroyed stars don't move
            return

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Bounce off padded boundaries
        if self.x < GALAXY_PADDING:
            self.x = GALAXY_PADDING
            self.vx *= -1 # Reverse horizontal velocity
        elif self.x > SCREEN_WIDTH - GALAXY_PADDING:
            self.x = SCREEN_WIDTH - GALAXY_PADDING
            self.vx *= -1

        if self.y < GALAXY_PADDING:
            self.y = GALAXY_PADDING
            self.vy *= -1 # Reverse vertical velocity
        elif self.y > SCREEN_HEIGHT - GALAXY_PADDING:
            self.y = SCREEN_HEIGHT - GALAXY_PADDING
            self.vy *= -1

    def draw(self, screen):
        """Draws the star on the screen."""
        # Use integer coordinates for drawing
        draw_x = int(self.x)
        draw_y = int(self.y)

        if self.is_destroyed:
            # Draw a red 'X' for destroyed stars
            pygame.draw.line(screen, DESTROYED_STAR_COLOR, (draw_x - STAR_RADIUS, draw_y - STAR_RADIUS), (draw_x + STAR_RADIUS, draw_y + STAR_RADIUS), 1)
            pygame.draw.line(screen, DESTROYED_STAR_COLOR, (draw_x - STAR_RADIUS, draw_y + STAR_RADIUS), (draw_x + STAR_RADIUS, draw_y - STAR_RADIUS), 1)
        else:
            # Draw normal star
            color = UNCLAIMED_COLOR
            if self.claimed_by:
                color = self.claimed_by.color
            pygame.draw.circle(screen, color, (draw_x, draw_y), STAR_RADIUS)
            # Draw resource indicator if bonus exists
            if self.resource_bonus > 0:
                pygame.draw.circle(screen, RESOURCE_INDICATOR_COLOR, (draw_x, draw_y), STAR_RADIUS + 1, 1) # Outline

    def distance_to(self, other_star):
        """Calculates the Euclidean distance to another star."""
        # Use current float positions for accurate distance
        return math.sqrt((self.x - other_star.x)**2 + (self.y - other_star.y)**2)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Star) and self.id == other.id


class Civilization:
    """Represents an expanding civilization with resources, tech, and internal strife."""
    def __init__(self, id, home_star, color, initial_resources=INITIAL_RESOURCES, initial_ftl=False, initial_age=0):
        self.id = id
        self.home_star = home_star
        self.color = color
        self.controlled_star_ids = {home_star.id}
        self.expansion_radius = INITIAL_EXPANSION_RADIUS
        self.max_expansion_radius = math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2) * MAX_EXPANSION_RADIUS_FACTOR
        self.is_eliminated = False
        self.resources = initial_resources
        self.age = initial_age
        self.has_ftl = initial_ftl

        if home_star.claimed_by != self:
             home_star.claimed_by = self
             if home_star.resource_bonus > 0:
                 self.resources += home_star.resource_bonus
                 home_star.resource_bonus = 0

    def get_strength(self):
        """Calculates the civilization's strength (number of stars)."""
        # Ensure controlled stars actually exist and aren't destroyed before counting
        valid_stars = {sid for sid in self.controlled_star_ids if sid in stars_dict and not stars_dict[sid].is_destroyed}
        self.controlled_star_ids = valid_stars # Clean up potentially invalid IDs
        return len(self.controlled_star_ids) if not self.is_eliminated else 0


    def update_resources(self):
        """Calculates passive resource income based on valid controlled stars."""
        if not self.is_eliminated:
            # Count only valid stars for income
            valid_star_count = len([sid for sid in self.controlled_star_ids if sid in stars_dict and not stars_dict[sid].is_destroyed])
            income = valid_star_count * RESOURCE_INCOME_PER_STAR
            self.resources += income

    def check_for_events(self):
        """Checks for FTL discovery and civil war triggers. Returns True if civil war triggered."""
        if self.is_eliminated:
            return False
        self.age += 1
        # FTL Check
        if not self.has_ftl:
            ftl_prob = min(FTL_DISCOVERY_BASE_PROB_PER_FRAME + self.age * FTL_DISCOVERY_SCALING_FACTOR_PER_AGE, MAX_FTL_DISCOVERY_PROB)
            if random.random() < ftl_prob:
                self.has_ftl = True
        # Civil War Check
        num_stars = self.get_strength() # Use current valid star count
        if num_stars >= MIN_STARS_FOR_CIVIL_WAR:
            civil_war_prob = min(CIVIL_WAR_BASE_PROB_PER_FRAME + num_stars * CIVIL_WAR_SCALING_FACTOR_PER_STAR, MAX_CIVIL_WAR_PROB)
            if random.random() < civil_war_prob:
                return True # Signal civil war
        return False

    def expand(self, stars_dict):
        """ Expands influence and handles conflicts. """
        if self.is_eliminated: return
        self.update_resources() # Update resources first
        current_expansion_rate = FTL_EXPANSION_RATE if self.has_ftl else INITIAL_SLOW_EXPANSION_RATE
        if self.expansion_radius < self.max_expansion_radius:
             self.expansion_radius += current_expansion_rate

        potential_claims_with_cost = {}
        # Get current valid controlled stars for expansion source
        current_controlled_stars = [stars_dict[sid] for sid in self.controlled_star_ids if sid in stars_dict and not stars_dict[sid].is_destroyed]
        if not current_controlled_stars:
             # If no valid stars left, ensure elimination
             self.controlled_star_ids.clear()
             self.is_eliminated = True
             return

        for controlled_star in current_controlled_stars:
            for star_id, potential_star in stars_dict.items():
                # Ignore self-owned, destroyed stars
                if potential_star.claimed_by == self or potential_star.is_destroyed:
                    continue

                dist = controlled_star.distance_to(potential_star) # Distance uses current positions
                if dist <= self.expansion_radius:
                    cost = dist * EXPANSION_COST_FACTOR
                    target_star_id = potential_star.id
                    if target_star_id not in potential_claims_with_cost or cost < potential_claims_with_cost[target_star_id][0]:
                         potential_claims_with_cost[target_star_id] = (cost, controlled_star)

        sorted_potential_claims = sorted(potential_claims_with_cost.items(), key=lambda item: item[1][0])

        for target_star_id, (cost, _) in sorted_potential_claims:
            if self.resources < cost: continue
            target_star = stars_dict.get(target_star_id)
            # Re-check target validity before action
            if not target_star or target_star.is_destroyed: continue

            claimed_successfully = False
            attacked_successfully = False

            if target_star.claimed_by is None: # Claim unclaimed
                target_star.claimed_by = self
                self.controlled_star_ids.add(target_star.id)
                claimed_successfully = True
            else: # Attack claimed star
                defender = target_star.claimed_by
                if defender != self and not defender.is_eliminated:
                    # Use updated strength for combat resolution
                    attacker_strength = self.get_strength()
                    defender_strength = defender.get_strength()
                    if attacker_strength > defender_strength or (attacker_strength == defender_strength and random.random() < 0.5):
                        defender.controlled_star_ids.discard(target_star.id)
                        target_star.claimed_by = self
                        self.controlled_star_ids.add(target_star.id)
                        attacked_successfully = True
                        # Check defender elimination based on their updated valid star count
                        if defender.get_strength() == 0: defender.is_eliminated = True

            if claimed_successfully or attacked_successfully:
                self.resources -= cost
                if target_star.resource_bonus > 0:
                    self.resources += target_star.resource_bonus
                    target_star.resource_bonus = 0

    def consider_dark_forest_strike(self, stars_dict, civilizations_list):
        """Decides whether to perform a Dark Forest strike and executes it."""
        if self.is_eliminated: return

        # Check eligibility based on current valid strength and resources
        can_afford = self.resources >= DARK_FOREST_STRIKE_COST + MIN_RESOURCES_FOR_STRIKE_BUFFER
        is_large_enough = self.get_strength() >= MIN_STARS_FOR_STRIKE
        meets_probability = random.random() < DARK_FOREST_STRIKE_PROBABILITY_PER_FRAME

        if can_afford and is_large_enough and meets_probability:
            # Find potential targets (stars not owned by self and not destroyed)
            potential_targets = [star for star_id, star in stars_dict.items()
                                 if star.claimed_by != self and not star.is_destroyed]

            if potential_targets:
                target_star = random.choice(potential_targets)
                execute_dark_forest_strike(self, target_star, stars_dict, civilizations_list)


# --- Helper Functions ---

def execute_dark_forest_strike(striking_civ, target_star, stars_dict, civilizations_list):
    """Executes the effects of a Dark Forest strike."""
    striking_civ.resources -= DARK_FOREST_STRIKE_COST
    target_star.is_destroyed = True
    target_star.resource_bonus = 0

    victim_civ = target_star.claimed_by
    if victim_civ and victim_civ != striking_civ:
        victim_civ.controlled_star_ids.discard(target_star.id)
        # Check elimination based on updated valid star count
        if victim_civ.get_strength() == 0:
            victim_civ.is_eliminated = True

    target_star.claimed_by = None


def handle_civil_war(original_civ, civilizations_list, stars_dict, next_civ_id):
    """Handles the splitting of a civilization due to civil war."""
    if original_civ.is_eliminated: return next_civ_id

    # Use current valid stars for splitting
    valid_star_ids = list(original_civ.controlled_star_ids) # Already filtered in get_strength

    if len(valid_star_ids) < MIN_SPLIT_CIVS :
        return next_civ_id

    original_civ.is_eliminated = True
    num_splits = random.randint(MIN_SPLIT_CIVS, min(MAX_SPLIT_CIVS, len(valid_star_ids)))
    random.shuffle(valid_star_ids)
    resources_per_split = original_civ.resources * SPLIT_RESOURCE_FRACTION
    newly_created_civs = []
    stars_per_split = len(valid_star_ids) // num_splits
    extra_stars = len(valid_star_ids) % num_splits
    star_index = 0

    for i in range(num_splits):
        num_stars_for_this_split = stars_per_split + (1 if i < extra_stars else 0)
        if num_stars_for_this_split == 0: continue
        split_star_ids = set(valid_star_ids[star_index : star_index + num_stars_for_this_split])
        star_index += num_stars_for_this_split
        if not split_star_ids: continue

        # Choose home star from the valid set
        new_home_star = stars_dict[next(iter(split_star_ids))]

        new_civ = Civilization(
            id=next_civ_id, home_star=new_home_star, color=generate_random_color(),
            initial_resources=resources_per_split, initial_ftl=False, initial_age=0
        )
        # Constructor already handles claiming home star, need to assign the rest
        new_civ.controlled_star_ids = split_star_ids # Overwrite initial set with full split set

        # Update star ownership in the main stars_dict for all assigned stars
        for star_id in split_star_ids:
             if star_id in stars_dict: stars_dict[star_id].claimed_by = new_civ

        newly_created_civs.append(new_civ)
        next_civ_id += 1

    civilizations_list.extend(newly_created_civs)
    return next_civ_id


# --- Global stars_dict needed for get_strength ---
# This is slightly awkward design, but necessary for get_strength to access star state
stars_dict = {}

# --- Main Game Logic ---
def main():
    """Main function to run the simulation."""
    global stars_dict # Declare intent to modify the global dict

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Galaxy Sim (Stellar Motion Update)")
    clock = pygame.time.Clock()
    font_small = pygame.font.Font(None, 18)
    font_large = pygame.font.Font(None, 20)

    # --- Initialization ---
    stars_list = []
    stars_dict.clear() # Clear global dict before populating
    for i in range(NUM_STARS):
        x = random.randint(GALAXY_PADDING, SCREEN_WIDTH - GALAXY_PADDING)
        y = random.randint(GALAXY_PADDING, SCREEN_HEIGHT - GALAXY_PADDING)
        star = Star(x, y, i)
        stars_list.append(star)
        stars_dict[i] = star # Populate global dict

    if len(stars_list) < NUM_CIVILIZATIONS:
        print(f"Error: Not enough stars ({len(stars_list)}) for {NUM_CIVILIZATIONS} civilizations.")
        pygame.quit(); sys.exit()

    civilizations = []
    available_start_indices = list(range(len(stars_list)))
    random.shuffle(available_start_indices)
    next_civ_id = 0

    for i in range(NUM_CIVILIZATIONS):
        start_index = available_start_indices.pop()
        # Ensure starting star is valid
        while start_index >= len(stars_list) or stars_list[start_index].is_destroyed:
             if not available_start_indices: # No more valid stars left
                 print("Error: Could not find enough valid starting stars.")
                 pygame.quit(); sys.exit()
             start_index = available_start_indices.pop()

        home_star = stars_list[start_index]
        civ_color = CIV_COLORS[i % len(CIV_COLORS)]
        civilization = Civilization(id=next_civ_id, home_star=home_star, color=civ_color)
        civilizations.append(civilization)
        next_civ_id += 1

    # --- Game Loop ---
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False

        # --- Update ---

        # 1. Update all star positions FIRST
        for star in stars_list:
            star.update_position()

        # 2. Process civilization logic
        civs_to_split = []
        current_civilizations = [civ for civ in civilizations if not civ.is_eliminated]
        random.shuffle(current_civilizations)

        for civ in current_civilizations:
            if civ.is_eliminated: continue

            # Check for FTL/Civil War events
            if civ.check_for_events():
                civs_to_split.append(civ)

            # Perform expansion/conflict (uses updated star positions for distance)
            civ.expand(stars_dict)

            # Consider Dark Forest Strike
            civ.consider_dark_forest_strike(stars_dict, civilizations)


        # 3. Handle Civil Wars (after main update loop)
        if civs_to_split:
            for civ_to_split in civs_to_split:
                 if not civ_to_split.is_eliminated:
                     next_civ_id = handle_civil_war(civ_to_split, civilizations, stars_dict, next_civ_id)


        # --- Drawing ---
        screen.fill((10, 10, 20)) # Dark background

        # Draw all stars (position is now float, cast to int for drawing)
        for star in stars_list:
            star.draw(screen)

        # Draw simulation info (Civ stats)
        y_offset = 10
        total_claimed = 0
        active_civ_count = 0
        destroyed_star_count = 0

        for star in stars_list:
             if star.is_destroyed: destroyed_star_count += 1

        for civ in civilizations:
            if not civ.is_eliminated:
                # Use get_strength() to ensure count is based on valid stars
                strength = civ.get_strength()
                # If strength becomes 0 after check but civ not marked eliminated yet, mark it now
                if strength == 0 and len(civ.controlled_star_ids) == 0:
                    civ.is_eliminated = True
                    continue # Skip drawing this frame as it's just been eliminated

                res_text = f"{civ.resources:.0f}" if civ.resources < 10000 else f"{civ.resources/1000:.1f}k"
                ftl_status = " (FTL)" if civ.has_ftl else ""
                text = f"Civ {civ.id}{ftl_status}: {strength} stars | Res: {res_text}"
                text_surface = font_small.render(text, True, civ.color)
                screen.blit(text_surface, (10, y_offset))
                y_offset += 16
                total_claimed += strength # Strength already counts only valid stars
                active_civ_count += 1

        # Draw summary info
        # Unclaimed = Total - Claimed (valid) - Destroyed
        unclaimed_count = NUM_STARS - total_claimed - destroyed_star_count
        total_civ_count = len(civilizations)
        summary_text = f"Active: {active_civ_count}/{total_civ_count} | Unclaimed: {max(0, unclaimed_count)} | Destroyed: {destroyed_star_count}" # Ensure unclaimed isn't negative
        summary_surface = font_large.render(summary_text, True, (255, 255, 255))
        screen.blit(summary_surface, (SCREEN_WIDTH - summary_surface.get_width() - 10, 10))

        # Update the display
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

# --- Run the simulation ---
if __name__ == '__main__':
    main()
