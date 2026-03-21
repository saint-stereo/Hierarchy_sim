"""ForagerAgent and Patch classes for the hierarchy simulation."""

import mesa
import numpy as np


class ForagerAgent(mesa.Agent):
    """An agent that forages, pools resources, and may become insider/outsider.

    Attributes:
        energy: Current kcal reserves.
        hunger_streak: Consecutive days with caloric deficit.
        role: "commoner" or "gatekeeper".
        insider: Whether the agent belongs to the in-group.
        myth_counter: Reinforcement counter — increases each exclusion cycle
                      where an outsider starved while insiders did not.
        carry_capacity: Max kcal an agent can harvest per tick.
        daily_need: Maintenance kcal required per day to avoid hunger.
    """

    def __init__(self, model, energy=100.0, carry_capacity=12.0, daily_need=10.0):
        super().__init__(model)
        self.energy = energy
        self.hunger_streak = 0
        self.role = "commoner"
        self.insider = True  # everyone starts as insider
        self.myth_counter = 0
        self.carry_capacity = carry_capacity
        self.daily_need = daily_need
        # Rolling intake log (last 30 days)
        self._intake_log = []

    @property
    def hunger_status(self):
        """Fraction of recent days with deficit (0 = well-fed, 1 = starving)."""
        if not self._intake_log:
            return 0.0
        window = self._intake_log[-30:]
        deficit_days = sum(1 for x in window if x < self.daily_need)
        return deficit_days / len(window)

    def record_intake(self, amount):
        self._intake_log.append(amount)
        if len(self._intake_log) > 30:
            self._intake_log.pop(0)

    def forage(self):
        """Harvest food from the patch the agent is standing on."""
        patch = self.model.patch_at(self.pos)
        if patch is None:
            return 0.0
        harvested = min(patch.daily_yield, self.carry_capacity)
        patch.daily_yield -= harvested
        return harvested

    def metabolise(self, intake):
        """Consume daily_need from intake; surplus goes to energy reserves."""
        self.energy += intake - self.daily_need
        if intake < self.daily_need:
            self.hunger_streak += 1
        else:
            self.hunger_streak = 0
        self.record_intake(intake)

    def step(self):
        """Per-tick logic is orchestrated by the model, not the agent."""
        pass


class Patch:
    """A grid cell that produces food each day.

    Attributes:
        pos: (x, y) tuple on the grid.
        mu: Mean daily yield.
        sigma: Std-dev of daily yield.
        daily_yield: Realised yield this tick (regenerated each step).
        granary_stock: Stored food that persists across ticks.
        granary_capacity: Maximum storable food.
    """

    def __init__(self, pos, mu=10.0, sigma=3.0, granary_capacity=0.0, rng=None):
        self.pos = pos
        self.mu = mu
        self.sigma = sigma
        self.granary_capacity = granary_capacity
        self.granary_stock = 0.0
        self.daily_yield = 0.0
        self._rng = rng or np.random.default_rng()

    def regenerate(self):
        """Produce a new daily yield drawn from N(mu, sigma), floored at 0."""
        self.daily_yield = max(0.0, self._rng.normal(self.mu, self.sigma))

    def deposit(self, amount):
        """Store food in the granary up to capacity. Returns overflow."""
        space = self.granary_capacity - self.granary_stock
        stored = min(amount, space)
        self.granary_stock += stored
        return amount - stored  # overflow

    def withdraw(self, amount):
        """Take food from granary. Returns actual amount withdrawn."""
        taken = min(amount, self.granary_stock)
        self.granary_stock -= taken
        return taken
