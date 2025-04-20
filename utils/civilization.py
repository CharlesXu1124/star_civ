import math
import random

from utils.config import *


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
        self.current_expansion_targets = set() # Store IDs of stars being targeted this step
        self.civil_war_base_prob = CIVIL_WAR_BASE_PROB_PER_FRAME

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
        """Checks for FTL and Civil War. Returns civil_war_triggered, ftl_discovered_this_step."""
        if self.is_eliminated: return False, False
        self.age += 1
        ftl_discovered_this_step = False
        # FTL Check
        if not self.has_ftl:
            ftl_prob = min(FTL_DISCOVERY_BASE_PROB_PER_FRAME + self.age * FTL_DISCOVERY_SCALING_FACTOR_PER_AGE, MAX_FTL_DISCOVERY_PROB)
            if random.random() < ftl_prob:
                self.has_ftl = True
                ftl_discovered_this_step = True # Mark FTL discovery
        # Civil War Check
        num_stars = self.get_strength(stars_dict)
        if num_stars >= MIN_STARS_FOR_CIVIL_WAR:
            civil_war_prob = min(self.civil_war_base_prob + num_stars * CIVIL_WAR_SCALING_FACTOR_PER_STAR, MAX_CIVIL_WAR_PROB)
            if random.random() < civil_war_prob:
                return True, ftl_discovered_this_step # Signal civil war
        return False, ftl_discovered_this_step # No civil war

    def expand(self, stars_dict):
        """ Expands influence and handles conflicts. Returns claimed_ids, conquered_ids, lost_ids sets."""
        if self.is_eliminated: return set(), set(), set()

        self.current_expansion_targets.clear() # Clear targets from previous step
        self.update_resources(stars_dict) # Update resources first

        claimed_this_step = set()
        conquered_this_step = set()

        current_expansion_rate = FTL_EXPANSION_RATE if self.has_ftl else INITIAL_SLOW_EXPANSION_RATE
        if self.expansion_radius < self.max_expansion_radius:
             self.expansion_radius += current_expansion_rate

        potential_claims_with_cost = {}
        current_controlled_stars = [stars_dict[sid] for sid in self.controlled_star_ids if sid in stars_dict and not stars_dict[sid].is_destroyed]
        if not current_controlled_stars:
             self.is_eliminated = True
             return set(), set(), set()

        # Identify potential targets
        for controlled_star in current_controlled_stars:
            for star_id, potential_star in stars_dict.items():
                if potential_star.claimed_by_id == self.id or potential_star.is_destroyed: continue
                dist = controlled_star.distance_to(potential_star)
                if dist <= self.expansion_radius:
                    cost = dist * EXPANSION_COST_FACTOR
                    target_star_id = potential_star.id
                    if target_star_id not in potential_claims_with_cost or cost < potential_claims_with_cost[target_star_id][0]:
                         potential_claims_with_cost[target_star_id] = (cost, controlled_star)

        # Store potential targets for visualization BEFORE checking affordability
        self.current_expansion_targets = set(potential_claims_with_cost.keys())

        # Attempt claims/attacks based on affordability and simplified conflict
        sorted_potential_claims = sorted(potential_claims_with_cost.items(), key=lambda item: item[1][0])

        for target_star_id, (cost, _) in sorted_potential_claims:
            if self.resources < cost: continue
            target_star = stars_dict.get(target_star_id)
            if not target_star or target_star.is_destroyed: continue # Recheck validity

            claimed_successfully = False
            attacked_successfully = False

            if target_star.claimed_by_id == -1: # Claim unclaimed
                target_star.claimed_by_id = self.id
                self.controlled_star_ids.add(target_star.id)
                claimed_successfully = True
                claimed_this_step.add(target_star.id)
            else: # Attack claimed star (simplified conflict placeholder)
                if random.random() < 0.3:
                     # We don't know the defender here, assume conquest happens
                     # A proper implementation needs global conflict resolution in Env.step
                     original_owner_id = target_star.claimed_by_id # Store original owner
                     target_star.claimed_by_id = self.id
                     self.controlled_star_ids.add(target_star.id)
                     attacked_successfully = True
                     conquered_this_step.add(target_star.id)
                     # The Env step function should handle removing the star from the original owner


            if claimed_successfully or attacked_successfully:
                self.resources -= cost
                if target_star.resource_bonus > 0:
                    self.resources += target_star.resource_bonus
                    target_star.resource_bonus = 0

        # Lost stars need to be determined globally in the Env step
        return claimed_this_step, conquered_this_step, set()

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
                return True, target_star
        return False, None