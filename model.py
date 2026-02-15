"""HierarchyModel — Mesa Model for recursive-othering hierarchy simulation.

Hierarchy emerges through *generational exclusion*:

1.  The founding community is egalitarian (all insiders).
2.  Ecological stress triggers resource pooling and gatekeeper election.
3.  The gatekeeper marks a one-time insider/outsider split among existing
    agents (the "founding exclusion").
4.  After that, outsider status is **inherited at birth** — children of
    outsiders are outsiders.  The gatekeeper controls the "gate": when the
    insider fraction already exceeds the bias, new births are outsiders.
5.  Population self-regulates via a logistic birth-rate that peaks with
    resource security and declines under population pressure.
6.  Outsiders who can afford to migrate get a fresh start (insider = True
    in a new location); those who cannot are trapped in a permanent
    underclass whose myth_counter climbs with each exclusion cycle.
"""

import mesa
import numpy as np

from agents import ForagerAgent, Patch


def _gini(values):
    """Compute Gini coefficient from a list of non-negative values."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))


class HierarchyModel(mesa.Model):
    """Hierarchy formation by recursive othering.

    Parameters:
        width, height: Grid dimensions.
        n_agents: Initial population.
        mu: Mean patch yield (kcal/day).
        sigma: Std-dev of patch yield.
        escape_cost: Probability threshold for migration (0 = easy, 1 = trapped).
        granary_capacity: Max storable food per patch.
        gatekeeper_bias: Target insider fraction the gatekeeper protects.
        myth_decay_rate: Per-tick decay applied to myth_counter.
        stress_threshold: stress_ratio below which pooling is triggered.
        daily_need: Agent maintenance kcal per day.
        carry_capacity: Max kcal an agent can harvest per tick.
        seed: Random seed.
    """

    def __init__(
        self,
        width=40,
        height=40,
        n_agents=200,
        mu=11.0,
        sigma=3.0,
        escape_cost=0.5,
        granary_capacity=50.0,
        gatekeeper_bias=0.6,
        myth_decay_rate=0.02,
        stress_threshold=1.0,
        daily_need=10.0,
        carry_capacity=15.0,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.mu = mu
        self.sigma = sigma
        self.escape_cost = escape_cost
        self.granary_capacity = granary_capacity
        self.gatekeeper_bias = gatekeeper_bias
        self.myth_decay_rate = myth_decay_rate
        self.stress_threshold = stress_threshold
        self.daily_need = daily_need
        self.carry_capacity = carry_capacity

        self._rng = np.random.default_rng(seed)

        # --- Grid ---
        self.grid = mesa.space.MultiGrid(width, height, torus=True)

        # --- Patches (one per cell) ---
        self.patches = {}
        for x in range(width):
            for y in range(height):
                p = Patch(
                    pos=(x, y),
                    mu=mu,
                    sigma=sigma,
                    granary_capacity=granary_capacity,
                    rng=self._rng,
                )
                self.patches[(x, y)] = p

        # --- Agents (founding egalitarian community — all insiders) ---
        for _ in range(n_agents):
            a = ForagerAgent(
                self,
                energy=100.0,
                carry_capacity=carry_capacity,
                daily_need=daily_need,
            )
            a.insider = True
            x = self._rng.integers(0, width)
            y = self._rng.integers(0, height)
            self.grid.place_agent(a, (x, y))

        # --- Gatekeeper tracking ---
        self.current_gatekeeper = None
        self.gatekeeper_tenure = 0

        # --- Tick counter ---
        self._step_count = 0

        # --- DataCollector ---
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Gini_Intake": self._gini_intake,
                "Insider_Share": self._insider_share,
                "Mean_Myth": self._mean_myth,
                "Gatekeeper_Tenure": lambda m: m.gatekeeper_tenure,
                "Population": lambda m: len(m.agents),
                "Deaths": lambda m: getattr(m, "_deaths_this_step", 0),
                "Migrations": lambda m: getattr(m, "_migrations_this_step", 0),
                "Births": lambda m: getattr(m, "_births_this_step", 0),
            },
            agent_reporters={
                "Energy": "energy",
                "Hunger_Streak": "hunger_streak",
                "Role": "role",
                "Insider": "insider",
                "Myth_Counter": "myth_counter",
            },
        )

    # ------------------------------------------------------------------ helpers
    def patch_at(self, pos):
        return self.patches.get(pos)

    def _living_agents(self):
        return [a for a in self.agents if isinstance(a, ForagerAgent)]

    @staticmethod
    def _gini_intake(model):
        agents = model._living_agents()
        if not agents:
            return 0.0
        intakes = [a._intake_log[-1] if a._intake_log else 0.0 for a in agents]
        return _gini(intakes)

    @staticmethod
    def _insider_share(model):
        agents = model._living_agents()
        if not agents:
            return 0.0
        return sum(1 for a in agents if a.insider) / len(agents)

    @staticmethod
    def _mean_myth(model):
        agents = model._living_agents()
        if not agents:
            return 0.0
        return np.mean([a.myth_counter for a in agents])

    # ------------------------------------------------------------------ step
    def step(self):
        self._deaths_this_step = 0
        self._migrations_this_step = 0
        self._births_this_step = 0

        agents = self._living_agents()
        if not agents:
            return

        # 0. Regenerate patches
        for p in self.patches.values():
            p.regenerate()

        # 1. Forage — each agent harvests from their patch
        self._rng.shuffle(agents)
        harvests = {}
        for a in agents:
            harvests[a.unique_id] = a.forage()

        # 2. Pool decision — global stress_ratio vs threshold
        total_harvest = sum(harvests.values())
        total_need = sum(a.daily_need for a in agents)
        stress_ratio = total_harvest / total_need if total_need > 0 else 1.0

        if stress_ratio < self.stress_threshold:
            # --- POOLING MODE ---
            new_gk = self._elect_gatekeeper(agents)
            # Founding exclusion: gatekeeper splits existing agents ONCE
            if new_gk:
                self._assign_insiders(agents)

            # Pool all harvest + granary reserves
            central = self.patches[(0, 0)]
            pool = total_harvest
            pool += central.withdraw(central.granary_stock)

            # Distribute: insiders first with modest premium
            insiders = [a for a in agents if a.insider]
            outsiders = [a for a in agents if not a.insider]
            self._rng.shuffle(insiders)
            self._rng.shuffle(outsiders)

            remaining = pool
            for a in insiders:
                share = min(
                    a.daily_need * 1.1,
                    remaining / max(len(insiders), 1),
                )
                a.metabolise(share)
                remaining -= share

            for a in outsiders:
                share = min(
                    a.daily_need,
                    remaining / max(len(outsiders), 1),
                )
                a.metabolise(share)
                remaining -= share

            # Leftover to granary
            if remaining > 0:
                central.deposit(remaining)
        else:
            # --- INDIVIDUAL MODE ---
            for a in agents:
                a.metabolise(harvests[a.unique_id])

            # Outsiders whose myth support has fully decayed are rehabilitated
            for a in agents:
                if not a.insider and a.myth_counter <= 0:
                    a.insider = True

        # Energy cap: surplus above 300 goes to local granary (both modes)
        for a in agents:
            surplus = max(0.0, a.energy - 300.0)
            if surplus > 0:
                patch = self.patch_at(a.pos)
                if patch:
                    overflow = patch.deposit(surplus)
                    a.energy -= surplus - overflow
                else:
                    a.energy = 300.0

        # 3. Narrative reinforcement
        self._narrative_reinforcement(agents)

        # 4. Myth decay
        self._myth_decay(agents)

        # 5. Migration / death
        self._migration_and_death(agents)

        # 6. Birth — gatekeeper-controlled status, logistic regulation
        self._birth()

        # 7. Collect data
        self._step_count += 1
        self.datacollector.collect(self)

    # ------------------------------------------------ sub-routines

    def _elect_gatekeeper(self, agents):
        """Elect gatekeeper (highest energy). Returns True if NEW election."""
        if self.current_gatekeeper and self.current_gatekeeper in agents:
            self.gatekeeper_tenure += 1
            return False
        # New election
        best = max(agents, key=lambda a: a.energy)
        if self.current_gatekeeper and self.current_gatekeeper in agents:
            self.current_gatekeeper.role = "commoner"
        best.role = "gatekeeper"
        self.current_gatekeeper = best
        self.gatekeeper_tenure = 1
        return True

    def _assign_insiders(self, agents):
        """Founding exclusion: gatekeeper marks a one-time insider/outsider
        split among existing agents.  Uses proximity + jitter (kin bias)."""
        n_insiders = max(1, int(len(agents) * self.gatekeeper_bias))
        gk_pos = self.current_gatekeeper.pos
        by_distance = sorted(
            agents,
            key=lambda a: (
                abs(a.pos[0] - gk_pos[0]) + abs(a.pos[1] - gk_pos[1])
                + self._rng.random() * 5
            ),
        )
        for i, a in enumerate(by_distance):
            a.insider = i < n_insiders

    def _narrative_reinforcement(self, agents):
        """If any outsider went hungry while an insider did not, insiders
        increment myth_counter (diminishing returns)."""
        outsiders_hungry = any(
            not a.insider and a._intake_log and a._intake_log[-1] < a.daily_need
            for a in agents
        )
        insiders_fed = any(
            a.insider and a._intake_log and a._intake_log[-1] >= a.daily_need
            for a in agents
        )
        if outsiders_hungry and insiders_fed:
            for a in agents:
                if a.insider:
                    a.myth_counter += 1.0 / (1.0 + a.myth_counter * 0.05)

    def _myth_decay(self, agents):
        """Decay myth_counter gradually.  Outsiders lose it faster."""
        for a in agents:
            if a.myth_counter > 0 and not a.insider:
                a.myth_counter = max(0, a.myth_counter - self.myth_decay_rate * 5)
            elif a.myth_counter > 0:
                a.myth_counter = max(0, a.myth_counter - self.myth_decay_rate)

    def _migration_and_death(self, agents):
        """Energy-depleted agents die.  Hungry outsiders may migrate."""
        to_remove = []
        for a in agents:
            # Death by energy depletion
            if a.energy <= 0:
                to_remove.append(a)
                continue
            # Outsiders with sustained hunger try to escape
            if not a.insider and a.hunger_status > 0.3:
                if self._rng.random() > self.escape_cost:
                    new_x = self._rng.integers(0, self.width)
                    new_y = self._rng.integers(0, self.height)
                    self.grid.move_agent(a, (new_x, new_y))
                    a.hunger_streak = 0
                    a.insider = True   # fresh start
                    a.myth_counter = 0
                    self._migrations_this_step += 1

        for a in to_remove:
            if a == self.current_gatekeeper:
                self.current_gatekeeper = None
                self.gatekeeper_tenure = 0
            self.grid.remove_agent(a)
            a.remove()
            self._deaths_this_step += 1

    def _birth(self):
        """Birth with logistic population pressure and inherited status.

        - Any pair with sufficient energy can breed.
        - Birth probability decreases as population approaches carrying
          capacity (2x initial), creating a natural demographic peak.
        - Child's insider status is INHERITED:
            * Both parents insider → child is insider (if slots remain).
            * Any parent outsider → child is outsider.
          When the gatekeeper is active and the insider fraction already
          exceeds the bias, even insider parents may produce outsider
          children — the "gate" closes.
        """
        agents = self._living_agents()
        pop = len(agents)
        if pop == 0:
            return

        # Logistic birth-rate: probability falls as pop approaches carrying cap
        carrying_cap = self.n_agents * 2
        birth_prob = max(0.0, 1.0 - (pop / carrying_cap) ** 2)
        if birth_prob < 0.01:
            return

        # Find fertile pairs (energy above threshold)
        fertile = [a for a in agents if a.energy > a.daily_need * 8]
        self._rng.shuffle(fertile)
        pairs = list(zip(fertile[::2], fertile[1::2]))

        insider_frac = sum(1 for a in agents if a.insider) / pop

        births_this_tick = 0
        for parent_a, parent_b in pairs:
            if births_this_tick >= 3:
                break
            if self._rng.random() > birth_prob:
                continue

            child = ForagerAgent(
                self,
                energy=40.0,
                carry_capacity=self.carry_capacity,
                daily_need=self.daily_need,
            )

            # --- Inherited status with gatekeeper gate control ---
            if parent_a.insider and parent_b.insider:
                if self.current_gatekeeper and insider_frac > self.gatekeeper_bias:
                    # Gate closes: even insider parents produce outsiders
                    # with probability proportional to overshoot
                    overshoot = insider_frac - self.gatekeeper_bias
                    child.insider = self._rng.random() > overshoot
                else:
                    child.insider = True
            else:
                # Any outsider parent → outsider child (inherited exclusion)
                child.insider = False

            # Inherit weakened myth from parents
            child.myth_counter = (
                parent_a.myth_counter + parent_b.myth_counter
            ) * 0.25

            self.grid.place_agent(child, parent_a.pos)
            parent_a.energy -= 30.0
            parent_b.energy -= 30.0
            births_this_tick += 1
            self._births_this_step += 1
