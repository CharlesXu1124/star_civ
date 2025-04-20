import pygame
import math
import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from utils import Civilization, Star
from utils.config import *


def generate_random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
if NUM_CIVILIZATIONS > len(CIV_COLORS):
    for _ in range(NUM_CIVILIZATIONS - len(CIV_COLORS)):
        CIV_COLORS.append(generate_random_color())


# --- Helper Function for Dashed Lines ---
def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=DASH_LENGTH, gap_length=GAP_LENGTH):
    """Draws a dashed line on a surface."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = dash_length
    gl = gap_length

    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0: return # Avoid division by zero

    dashes = int(dist / (dl + gl))
    if dashes == 0: # If too short for a full dash+gap, draw solid
         pygame.draw.line(surf, color, start_pos, end_pos, width)
         return

    # Calculate unit vector
    ux = dx / dist
    uy = dy / dist

    # Draw dashes
    for i in range(dashes):
        start = (x1 + ux * (dl + gl) * i, y1 + uy * (dl + gl) * i)
        end = (x1 + ux * ((dl + gl) * i + dl), y1 + uy * ((dl + gl) * i + dl))
        pygame.draw.line(surf, color, start, end, width)

    # Draw the last partial dash if needed
    remaining_dist = dist - dashes * (dl + gl)
    if remaining_dist > 0:
        start = (x1 + ux * (dl + gl) * dashes, y1 + uy * (dl + gl) * dashes)
        # Draw only up to the end point or the dash length, whichever is smaller
        final_dash_len = min(dl, remaining_dist)
        end = (start[0] + ux * final_dash_len, start[1] + uy * final_dash_len)
        # Ensure the final segment doesn't overshoot the end_pos due to float precision
        final_end_x = min(end[0], x2) if dx > 0 else max(end[0], x2)
        final_end_y = min(end[1], y2) if dy > 0 else max(end[1], y2)
        # Crude check to prevent overshooting on near-vertical/horizontal lines
        if abs(final_end_x - x1) > abs(dx) or abs(final_end_y - y1) > abs(dy):
             final_end_x, final_end_y = x2, y2

        pygame.draw.line(surf, color, start, (final_end_x, final_end_y), width)


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
        self.action_space = spaces.Discrete(3)

        # Define observation space
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
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # 1. Agent's Internal State
        obs_internal = np.array([
            np.clip(agent_civ.resources / (INITIAL_RESOURCES * 10), -1, 1),
            np.clip(agent_civ.age / MAX_STEPS_PER_EPISODE, -1, 1),
            1.0 if agent_civ.has_ftl else 0.0,
            np.clip(agent_civ.expansion_radius / agent_civ.max_expansion_radius, -1, 1),
            np.clip(agent_civ.get_strength(self.stars_dict) / NUM_STARS, -1, 1)
        ], dtype=np.float32)

        # 2. Local Periphery
        agent_stars = [self.stars_dict[sid] for sid in agent_civ.controlled_star_ids if sid in self.stars_dict]
        if not agent_stars:
            center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        else:
            center_x = sum(s.x for s in agent_stars) / len(agent_stars)
            center_y = sum(s.y for s in agent_stars) / len(agent_stars)

        nearby_stars = []
        for star_id, star in self.stars_dict.items():
            if star.claimed_by_id != self.agent_id and not star.is_destroyed:
                dist = math.sqrt((star.x - center_x)**2 + (star.y - center_y)**2)
                nearby_stars.append((dist, star))

        nearby_stars.sort(key=lambda x: x[0])
        obs_periphery = np.zeros(N_NEAREST_STARS_OBS * 7, dtype=np.float32)
        max_dist_norm = math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)

        for i in range(min(N_NEAREST_STARS_OBS, len(nearby_stars))):
            dist, star = nearby_stars[i]
            rel_x = (star.x - center_x) / (SCREEN_WIDTH / 2); rel_y = (star.y - center_y) / (SCREEN_HEIGHT / 2)
            is_unclaimed = 1.0 if star.claimed_by_id == -1 else 0.0
            is_other_civ = 1.0 if star.claimed_by_id != -1 else 0.0 # Simplified: assumes not self
            resource_bonus = star.resource_bonus / MAX_STAR_RESOURCE
            norm_dist = dist / max_dist_norm
            start_idx = i * 7
            obs_periphery[start_idx:start_idx+7] = [
                np.clip(rel_x, -1, 1), np.clip(rel_y, -1, 1),
                is_unclaimed, is_other_civ, 0.0, # is_destroyed placeholder
                np.clip(resource_bonus, 0, 1), np.clip(norm_dist, 0, 1)
            ]

        # 3. Global Context
        active_rivals = sum(1 for cid, civ in self.civilizations.items() if cid != self.agent_id and not civ.is_eliminated)
        total_claimed = sum(civ.get_strength(self.stars_dict) for cid, civ in self.civilizations.items() if not civ.is_eliminated) # Use get_strength
        total_destroyed = sum(1 for star in self.stars_list if star.is_destroyed)
        obs_global = np.array([
            np.clip(active_rivals / (NUM_CIVILIZATIONS -1 + 1e-6), 0, 1),
            np.clip(total_claimed / NUM_STARS, 0, 1),
            np.clip(total_destroyed / NUM_STARS, 0, 1)
        ], dtype=np.float32)

        # Concatenate
        observation = np.concatenate([obs_internal, obs_periphery, obs_global]).astype(np.float32)
        if observation.shape != self.observation_space.shape:
             print(f"Error: Obs shape mismatch. Expected {self.observation_space.shape}, got {observation.shape}")
             observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation


    def _get_info(self):
        """Returns auxiliary information (optional)."""
        agent_civ = self.civilizations.get(self.agent_id)
        if not agent_civ or agent_civ.is_eliminated: return {"step": self.current_step}
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
        super().reset(seed=seed)
        self.stars_list = []
        self.stars_dict = {}
        self.civilizations = {}
        self.current_step = 0
        self.next_civ_id = 0

        for i in range(NUM_STARS):
            x = self.np_random.integers(GALAXY_PADDING, SCREEN_WIDTH - GALAXY_PADDING)
            y = self.np_random.integers(GALAXY_PADDING, SCREEN_HEIGHT - GALAXY_PADDING)
            star = Star(x, y, i)
            self.stars_list.append(star)
            self.stars_dict[i] = star

        available_start_indices = list(range(len(self.stars_list)))
        self.np_random.shuffle(available_start_indices)
        if len(available_start_indices) < NUM_CIVILIZATIONS: raise ValueError("Not enough stars.")

        for i in range(NUM_CIVILIZATIONS):
            start_index = available_start_indices.pop()
            home_star = self.stars_list[start_index]
            while home_star.is_destroyed:
                 if not available_start_indices: raise ValueError("Not enough valid stars.")
                 start_index = available_start_indices.pop()
                 home_star = self.stars_list[start_index]
            civ_color = CIV_COLORS[i % len(CIV_COLORS)]
            civ_id = self.next_civ_id
            civilization = Civilization(id=civ_id, home_star_id=home_star.id, color=civ_color, stars_dict=self.stars_dict)
            self.civilizations[civ_id] = civilization
            self.next_civ_id += 1

        agent_civ = self.civilizations.get(self.agent_id)
        self.last_agent_resources = agent_civ.resources if agent_civ else 0
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human": self._render_init()
        return observation, info

    def step(self, action):
        """Runs one timestep of the environment's dynamics."""
        agent_civ = self.civilizations.get(self.agent_id)
        if not agent_civ or agent_civ.is_eliminated:
             obs = self._get_obs(); return obs, 0, True, False, self._get_info()

        terminated = False; truncated = False; reward = 0.0
        agent_civ.last_action = action
        events_this_step = {'ftl_discovered': False, 'agent_strike_executed': False, 'stars_claimed': 0, 'stars_lost': 0}
        initial_agent_stars = agent_civ.controlled_star_ids.copy()

        # --- Simulation Step ---
        civs_to_split = []
        civs_triggering_strike = {}
        all_conflicts = {} # Store potential conflicts: target_star_id -> list of attacker_ids

        # 1. Update Star Positions
        for star in self.stars_list: star.update_position()

        # 2. Process Civilization Logic (Events, Expansion Intent, Strike Intent)
        current_civ_ids = list(self.civilizations.keys())
        random.shuffle(current_civ_ids)

        for civ_id in current_civ_ids:
            civ = self.civilizations.get(civ_id)
            if not civ or civ.is_eliminated: continue

            # Store stars before expansion for loss calculation later
            stars_before_expand = civ.controlled_star_ids.copy()

            # Check Events
            civil_war_triggered, ftl_discovered = civ.check_for_events(self.stars_dict)
            if civil_war_triggered: civs_to_split.append(civ)
            if civ_id == self.agent_id and ftl_discovered: events_this_step['ftl_discovered'] = True

            # Expansion Intent (Populate civ.current_expansion_targets)
            # The expand method now primarily identifies targets
            civ.expand(self.stars_dict) # This updates radius, resources, and populates current_expansion_targets

            # --- Conflict Identification ---
            # Identify which stars multiple civs are targeting
            for target_id in civ.current_expansion_targets:
                 target_star = self.stars_dict.get(target_id)
                 if target_star and not target_star.is_destroyed: # Check if target is valid
                     # Check affordability for this specific target
                     # Simplification: Use average distance or home star distance for cost check?
                     # Let's use a simplified cost check for intent: assume affordable if resources > 0
                     if civ.resources > EXPANSION_COST_FACTOR * 10: # Heuristic affordability check
                         if target_id not in all_conflicts: all_conflicts[target_id] = []
                         all_conflicts[target_id].append(civ_id)


            # Dark Forest Strike Intent
            if civ_id == self.agent_id:
                if action == 1: # Action 1: Attempt Strike
                    can_afford = civ.resources >= DARK_FOREST_STRIKE_COST + MIN_RESOURCES_FOR_STRIKE_BUFFER
                    is_large_enough = civ.get_strength(self.stars_dict) >= MIN_STARS_FOR_STRIKE
                    if can_afford and is_large_enough:
                        potential_targets = [star for star_id, star in self.stars_dict.items()
                                            if star.claimed_by_id != civ.id and not star.is_destroyed]
                        if potential_targets:
                            target_star = random.choice(potential_targets)
                            civs_triggering_strike[civ_id] = target_star
                            events_this_step['agent_strike_executed'] = True
                if action == 2: # Action 2: induce civil war
                    can_afford = civ.resources >= 100
                    if can_afford:
                        potential_targets = [star for star_id, star in self.stars_dict.items()
                                            if star.claimed_by_id != civ.id and not star.is_destroyed]
                        if potential_targets and star.claimed_by_id is not None:
                            target_star = random.choice(potential_targets)
                            target_civ = self.civilizations.get(star.claimed_by_id)
                            if target_civ:
                                target_civ.civil_war_base_prob = max(target_civ.civil_war_base_prob * 10, 0.5)

            else: # Non-agent civs
                strike_triggered, target_star = civ.consider_dark_forest_strike(self.stars_dict)
                if strike_triggered: civs_triggering_strike[civ_id] = target_star


        # 3. Resolve Conflicts & Claims (Global Resolution)
        stars_claimed_by_agent = 0
        stars_lost_by_agent = 0 # Track agent losses specifically
        processed_conflicts = set() # Avoid double-processing

        # Shuffle conflict order for fairness
        conflict_items = list(all_conflicts.items())
        random.shuffle(conflict_items)

        for target_id, attacker_ids in conflict_items:
            if target_id in processed_conflicts: continue

            target_star = self.stars_dict.get(target_id)
            if not target_star or target_star.is_destroyed: continue

            current_owner_id = target_star.claimed_by_id
            valid_attackers = []
            for attacker_id in attacker_ids:
                 attacker_civ = self.civilizations.get(attacker_id)
                 # Check if attacker still exists and can afford (simplified cost check)
                 if self.civilizations.get(current_owner_id):
                    if attacker_civ and not attacker_civ.is_eliminated and attacker_civ.resources \
                        > EXPANSION_COST_FACTOR * 10 + self.civilizations.get(current_owner_id).resources:
                        valid_attackers.append(attacker_civ)

            if not valid_attackers: continue # No one can actually afford to attack/claim

            winner = None
            if current_owner_id == -1: # Unclaimed star
                 # Strongest attacker claims
                 winner = max(valid_attackers, key=lambda c: c.get_strength(self.stars_dict))
            else: # Star is owned
                 owner_civ = self.civilizations.get(current_owner_id)
                 if owner_civ and not owner_civ.is_eliminated: # Owner still exists
                     owner_strength = owner_civ.get_strength(self.stars_dict)
                     # Find strongest attacker
                     strongest_attacker = max(valid_attackers, key=lambda c: c.get_strength(self.stars_dict))
                     attacker_strength = strongest_attacker.get_strength(self.stars_dict)

                     # Compare strength (owner keeps on tie)
                     if attacker_strength > owner_strength:
                          winner = strongest_attacker
                     else:
                          winner = owner_civ # Owner defends successfully
                 else: # Owner was eliminated or doesn't exist, strongest attacker claims
                     winner = max(valid_attackers, key=lambda c: c.get_strength(self.stars_dict))


            # Process winner
            if winner:
                 # Simplified cost deduction (average distance maybe?)
                 cost = winner.expansion_radius * EXPANSION_COST_FACTOR # Approximate cost
                 if winner.resources >= cost:
                     winner.resources -= cost
                     if target_star.resource_bonus > 0:
                         winner.resources += target_star.resource_bonus
                         target_star.resource_bonus = 0

                     original_owner_id = target_star.claimed_by_id
                     if original_owner_id != winner.id: # If ownership changes
                         target_star.claimed_by_id = winner.id
                         winner.controlled_star_ids.add(target_id)
                         if winner.id == self.agent_id: stars_claimed_by_agent += 1

                         # Remove from loser (if exists)
                         loser_civ = self.civilizations.get(original_owner_id)
                         if loser_civ:
                              loser_civ.controlled_star_ids.discard(target_id)
                              if loser_civ.id == self.agent_id: stars_lost_by_agent += 1
                              # Check elimination for loser
                              if loser_civ.get_strength(self.stars_dict) == 0: loser_civ.is_eliminated = True

            processed_conflicts.add(target_id)


        # 4. Execute Dark Forest Strikes
        for striker_id, target_star in civs_triggering_strike.items():
             striker_civ = self.civilizations.get(striker_id)
             if striker_civ and not striker_civ.is_eliminated and target_star and not target_star.is_destroyed:
                 if striker_civ.resources >= DARK_FOREST_STRIKE_COST:
                     original_owner_id = target_star.claimed_by_id
                     self._execute_dark_forest_strike(striker_civ, target_star)
                     # Check if agent lost a star due to this strike
                     if original_owner_id == self.agent_id:
                          stars_lost_by_agent += 1
                          # Re-check agent elimination possibility here
                          agent_civ_after_strike = self.civilizations.get(self.agent_id)
                          if agent_civ_after_strike and agent_civ_after_strike.get_strength(self.stars_dict) == 0:
                               agent_civ_after_strike.is_eliminated = True

                 elif striker_id == self.agent_id:
                      events_this_step['agent_strike_executed'] = False


        # 5. Handle Civil Wars
        if civs_to_split:
            new_civs_added = []
            civ_ids_before_split = set(self.civilizations.keys())
            for civ_to_split in civs_to_split:
                 if not civ_to_split.is_eliminated:
                     self.next_civ_id = self._handle_civil_war(civ_to_split, new_civs_added, self.next_civ_id)
            for new_civ in new_civs_added: self.civilizations[new_civ.id] = new_civ
            # Check if agent was the one split
            if self.agent_id not in self.civilizations:
                 agent_civ = None # Agent eliminated by civil war


        # 6. Final State Checks & Reward Calculation for Agent
        agent_civ = self.civilizations.get(self.agent_id) # Re-get agent civ instance

        if not agent_civ or agent_civ.is_eliminated:
            terminated = True
            reward += -100.0 # r_elimination
        else:
            # Use specific counts calculated during resolution
            events_this_step['stars_claimed'] = stars_claimed_by_agent
            events_this_step['stars_lost'] = stars_lost_by_agent

            reward += 0.01 # r_survival
            resource_change = agent_civ.resources - self.last_agent_resources
            reward += resource_change * 0.001 # r_resource_change
            reward += events_this_step['stars_claimed'] * 0.5 # r_claim
            reward += events_this_step['stars_lost'] * -0.5 # r_loss
            if events_this_step['ftl_discovered']: reward += 10.0 # r_ftl
            if events_this_step['agent_strike_executed']: reward += -0.1 # r_strike_cost

            self.last_agent_resources = agent_civ.resources


        # 7. Check Truncation
        self.current_step += 1
        if self.current_step >= MAX_STEPS_PER_EPISODE: truncated = True

        # 8. Get Next Observation & Info
        observation = self._get_obs()
        info = self._get_info()

        # 9. Render
        if self.render_mode == "human": self.render()

        return observation, reward, terminated, truncated, info


    # --- Simulation Logic Helpers ---
    def _execute_dark_forest_strike(self, striking_civ, target_star):
        """ Executes the effects of a Dark Forest strike. """
        if striking_civ.resources < DARK_FOREST_STRIKE_COST: return
        striking_civ.resources -= DARK_FOREST_STRIKE_COST
        target_star.is_destroyed = True
        target_star.resource_bonus = 0
        victim_civ_id = target_star.claimed_by_id
        target_star.claimed_by_id = -1
        if victim_civ_id != -1 and victim_civ_id != striking_civ.id:
            victim_civ = self.civilizations.get(victim_civ_id)
            if victim_civ:
                victim_civ.controlled_star_ids.discard(target_star.id)
                if victim_civ.get_strength(self.stars_dict) == 0: victim_civ.is_eliminated = True

    def _handle_civil_war(self, original_civ, new_civs_list, next_civ_id):
        """ Handles the splitting of a civilization. """
        if original_civ.is_eliminated: return next_civ_id
        valid_star_ids = [sid for sid in original_civ.controlled_star_ids if sid in self.stars_dict and not self.stars_dict[sid].is_destroyed]
        if len(valid_star_ids) < MIN_SPLIT_CIVS : return next_civ_id

        original_civ.is_eliminated = True
        # If the agent is the one splitting, mark it eliminated immediately
        if original_civ.id == self.agent_id:
             self.civilizations.pop(self.agent_id, None) # Remove agent from dict

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
                stars_dict=self.stars_dict,
                initial_resources=resources_per_split,
                initial_ftl=original_civ.has_ftl,
                initial_age=0
            )
            new_civ.controlled_star_ids = split_star_ids
            for star_id in split_star_ids:
                 if star_id in self.stars_dict: self.stars_dict[star_id].claimed_by_id = new_civ.id
            new_civs_list.append(new_civ)
            next_civ_id += 1
        return next_civ_id


    # --- Rendering Methods ---
    def _render_init(self):
         if self.screen is None and self.render_mode == 'human':
             pygame.init()
             pygame.display.init()
             self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
             pygame.display.set_caption("Galaxy Sim RL Environment")
         if self.clock is None and self.render_mode == 'human':
             self.clock = pygame.time.Clock()
         if self.render_mode == 'human' and not pygame.font.get_init():
              pygame.font.init()
              self.font_small = pygame.font.Font(None, 18)
              self.font_large = pygame.font.Font(None, 20)


    def render(self):
        """Renders the current environment state using Pygame."""
        if self.render_mode != 'human': return
        self._render_init()

        self.screen.fill((10, 10, 20))
        # Draw Stars
        for star in self.stars_list: self._draw_star(star)

        # Draw Expansion Target Lines
        for civ_id, civ in self.civilizations.items():
             if not civ.is_eliminated and civ.current_expansion_targets:
                 # Calculate center (approximate)
                 civ_stars = [self.stars_dict[sid] for sid in civ.controlled_star_ids if sid in self.stars_dict and not self.stars_dict[sid].is_destroyed]
                 if not civ_stars: continue # Skip if no valid stars
                 center_x = int(sum(s.x for s in civ_stars) / len(civ_stars))
                 center_y = int(sum(s.y for s in civ_stars) / len(civ_stars))
                 start_pos = (center_x, center_y)

                 # Draw line to each target
                 for target_id in civ.current_expansion_targets:
                     target_star = self.stars_dict.get(target_id)
                     if target_star and not target_star.is_destroyed:
                         end_pos = (int(target_star.x), int(target_star.y))
                         draw_dashed_line(self.screen, civ.color, start_pos, end_pos, width=1)


        # Draw Info Text
        y_offset = 10; total_claimed = 0; active_civ_count = 0
        destroyed_star_count = sum(1 for star in self.stars_list if star.is_destroyed)
        sorted_civ_ids = sorted(self.civilizations.keys())

        for civ_id in sorted_civ_ids:
            civ = self.civilizations[civ_id]
            if not civ.is_eliminated:
                is_agent = "(A)" if civ_id == self.agent_id else ""
                strength = civ.get_strength(self.stars_dict) # Recalculate for display
                if strength == 0 and len(civ.controlled_star_ids) == 0: # Handle just eliminated case
                     civ.is_eliminated = True; continue
                res_text = f"{civ.resources:.0f}" if civ.resources < 10000 else f"{civ.resources/1000:.1f}k"
                ftl_status = " (FTL)" if civ.has_ftl else ""
                text = f"Civ {civ.id}{is_agent}{ftl_status}: {strength} stars | Res: {res_text}"
                text_surface = self.font_small.render(text, True, civ.color)
                self.screen.blit(text_surface, (10, y_offset)); y_offset += 16
                total_claimed += strength; active_civ_count += 1

        unclaimed_count = NUM_STARS - total_claimed - destroyed_star_count
        total_civ_count = len(self.civilizations) # Includes eliminated civs created by splits
        summary_text = f"Active: {active_civ_count}/{total_civ_count} | Unclaimed: {max(0, unclaimed_count)} | Destroyed: {destroyed_star_count} | Step: {self.current_step}"
        summary_surface = self.font_large.render(summary_text, True, (255, 255, 255))
        self.screen.blit(summary_surface, (SCREEN_WIDTH - summary_surface.get_width() - 10, 10))

        pygame.event.pump(); pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])


    def _draw_star(self, star):
         """ Helper to draw a single star. """
         draw_x = int(star.x); draw_y = int(star.y)
         if star.is_destroyed:
             pygame.draw.line(self.screen, DESTROYED_STAR_COLOR, (draw_x - STAR_RADIUS, draw_y - STAR_RADIUS), (draw_x + STAR_RADIUS, draw_y + STAR_RADIUS), 1)
             pygame.draw.line(self.screen, DESTROYED_STAR_COLOR, (draw_x - STAR_RADIUS, draw_y + STAR_RADIUS), (draw_x + STAR_RADIUS, draw_y - STAR_RADIUS), 1)
         else:
             color = UNCLAIMED_COLOR
             owner_civ = self.civilizations.get(star.claimed_by_id)
             if owner_civ: color = owner_civ.color
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
