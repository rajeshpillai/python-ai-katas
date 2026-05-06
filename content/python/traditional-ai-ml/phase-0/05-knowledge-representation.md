# Knowledge Representation

> Phase 0 — What is AI? | Kata 0.5

---

## Concept & Intuition

### What problem are we solving?

For an AI system to reason about the world, it needs a way to **represent knowledge** — facts, relationships, hierarchies, and rules — in a structured format that algorithms can manipulate. Knowledge representation (KR) is the bridge between the messy real world and the precise computations that AI systems perform. Without good representation, even the smartest algorithms are useless.

Consider how differently you might represent the knowledge "a penguin is a bird that cannot fly." A flat database might store this as two separate, disconnected facts. A graph connects "penguin" to "bird" with an "is-a" relationship and to "fly" with a "cannot" relationship. An ontology adds formal semantics so that a reasoner can automatically infer that a penguin has feathers (because birds have feathers) but override the default assumption that it can fly. The choice of representation determines what inferences are possible, efficient, or even expressible.

The major knowledge representation schemes — **graphs**, **semantic networks**, **frames**, and **ontologies** — each offer different tradeoffs between expressiveness, computational tractability, and ease of knowledge acquisition. Understanding these schemes is essential because they underpin modern technologies like knowledge graphs (Google, Wikidata), recommendation systems, and even some aspects of how large language models organize information.

### Why naive approaches fail

The simplest approach — storing knowledge as a flat table of facts — fails because it cannot capture relationships, hierarchies, or context. Knowing that "Socrates is a man" and "all men are mortal" as separate facts does not let a flat table conclude that "Socrates is mortal." You need a representation that encodes the logical structure connecting these facts.

Another failure is **the frame problem** — the difficulty of representing what does NOT change when an action is taken. If you move a cup from a table to a shelf, you need to update the cup's location but NOT its color, weight, or material. In naive representations, you must explicitly state every non-change, which is combinatorially expensive.

### Mental models

- **The map is not the territory**: A knowledge representation is a model of reality — necessarily simplified. The art is choosing what to include and what to omit for a given task
- **Inheritance as efficiency**: Ontologies use "is-a" hierarchies so that shared properties are stated once at a high level and inherited by all subtypes, like how object-oriented programming uses class inheritance
- **Graphs as the universal connector**: Almost any knowledge structure can be represented as a graph. Nodes are entities; edges are relationships. This flexibility is why graphs are so pervasive in AI
- **Frames as stereotypes**: A frame represents a typical object of a class, with default values that can be overridden — like a template for "restaurant" with slots for cuisine, price range, and location

### Visual explanations

```
Knowledge Representation Approaches
======================================

1. SEMANTIC NETWORK (Graph-based)

   Animal
     |
    is-a
     |
   Bird ──has──> Feathers
     |              |
    is-a          has
     |              |
   Penguin ──has──> Tuxedo-like plumage
     |
   cannot
     |
    Fly


2. FRAME (Template-based)

   +------ Frame: Bird ------+
   | superclass: Animal      |
   | has_feathers: True      |
   | can_fly: True (default) |
   | has_wings: True         |
   | locomotion: flying      |
   +-------------------------+

   +------ Frame: Penguin ---+
   | superclass: Bird        |
   | can_fly: False (override)|
   | habitat: Antarctic      |
   | locomotion: swimming    |
   +-------------------------+


3. ONTOLOGY (Formal logic)

   Class(Animal)
   Class(Bird) SubClassOf(Animal)
   Bird SubClassOf(hasFeathers some Feather)
   Bird SubClassOf(canFly value true)

   Class(Penguin) SubClassOf(Bird)
   Penguin SubClassOf(canFly value false)  # override!

   Individual(Tux) Type(Penguin)
   => Inferred: Tux hasFeathers some Feather
   => Inferred: Tux canFly false
```

---

## Hands-on Exploration

1. **Build a knowledge graph**: Using the code below, explore the animal knowledge graph. Add new animals and relationships. Try querying for "all animals that can swim" or "all descendants of Mammal."

2. **Test inheritance**: Add a new entity "Ostrich" that, like Penguin, is a bird that cannot fly. Verify that the inheritance system correctly gives Ostrich feathers (from Bird) but overrides the flight ability.

3. **Explore semantic similarity**: The code computes a simple similarity measure between entities based on shared graph neighbors. Compare how "Dog" and "Wolf" are more similar than "Dog" and "Salmon." Think about why graph structure captures this intuition.

4. **Break the representation**: Try to represent the knowledge "Most birds can fly, but some cannot" in the graph. Notice how graphs struggle with uncertainty and defaults — this motivates probabilistic and frame-based approaches.

---

## Live Code

```python
"""
Knowledge Representation — Graphs, frames, and ontologies for AI reasoning.

This code builds a knowledge graph of animals, implements inheritance-based
reasoning, and visualizes the structure of knowledge.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ============================================================
# Part 1: Knowledge Graph
# ============================================================

class KnowledgeGraph:
    """A simple knowledge graph with entities, relationships, and inheritance."""

    def __init__(self):
        self.triples = []  # (subject, predicate, object)
        self.entities = set()
        self.adjacency = defaultdict(list)

    def add(self, subject, predicate, obj):
        """Add a triple (fact) to the knowledge graph."""
        self.triples.append((subject, predicate, obj))
        self.entities.add(subject)
        self.entities.add(obj)
        self.adjacency[subject].append((predicate, obj))

    def query(self, subject=None, predicate=None, obj=None):
        """Query the knowledge graph with optional wildcards (None = any)."""
        results = []
        for s, p, o in self.triples:
            if (subject is None or s == subject) and \
               (predicate is None or p == predicate) and \
               (obj is None or o == obj):
                results.append((s, p, o))
        return results

    def get_ancestors(self, entity):
        """Get all ancestors via 'is_a' relationships (inheritance chain)."""
        ancestors = []
        queue = deque([entity])
        visited = set()
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for pred, obj in self.adjacency.get(current, []):
                if pred == "is_a":
                    ancestors.append(obj)
                    queue.append(obj)
        return ancestors

    def get_properties(self, entity, inherited=True):
        """Get all properties of an entity, optionally including inherited ones."""
        properties = {}
        # Direct properties
        for pred, obj in self.adjacency.get(entity, []):
            if pred != "is_a":
                properties[pred] = obj
        # Inherited properties (from ancestors, overridden by closer definitions)
        if inherited:
            ancestors = self.get_ancestors(entity)
            for ancestor in reversed(ancestors):  # Start from most general
                for pred, obj in self.adjacency.get(ancestor, []):
                    if pred != "is_a" and pred not in properties:
                        properties[pred] = obj
        return properties

    def find_similar(self, entity1, entity2):
        """Compute simple similarity based on shared properties."""
        props1 = set(self.get_properties(entity1, inherited=True).items())
        props2 = set(self.get_properties(entity2, inherited=True).items())
        if not props1 and not props2:
            return 0.0
        shared = props1 & props2
        total = props1 | props2
        return len(shared) / len(total) if total else 0.0


# Build the knowledge graph
kg = KnowledgeGraph()

# Taxonomy (is-a relationships)
kg.add("Animal", "is_a", "LivingThing")
kg.add("Bird", "is_a", "Animal")
kg.add("Mammal", "is_a", "Animal")
kg.add("Fish", "is_a", "Animal")
kg.add("Penguin", "is_a", "Bird")
kg.add("Eagle", "is_a", "Bird")
kg.add("Sparrow", "is_a", "Bird")
kg.add("Dog", "is_a", "Mammal")
kg.add("Cat", "is_a", "Mammal")
kg.add("Whale", "is_a", "Mammal")
kg.add("Salmon", "is_a", "Fish")
kg.add("Shark", "is_a", "Fish")

# Properties at different levels (inheritance!)
kg.add("LivingThing", "needs", "energy")
kg.add("Animal", "can", "move")
kg.add("Animal", "has", "cells")
kg.add("Bird", "has", "feathers")
kg.add("Bird", "can", "fly")
kg.add("Bird", "has", "beak")
kg.add("Mammal", "has", "fur")
kg.add("Mammal", "feeds_young", "milk")
kg.add("Fish", "has", "scales")
kg.add("Fish", "lives_in", "water")
kg.add("Fish", "can", "swim")

# Overrides (exceptions to inherited properties)
kg.add("Penguin", "can", "swim")       # Override: can swim, not fly
kg.add("Penguin", "lives_in", "Antarctic")
kg.add("Whale", "lives_in", "ocean")
kg.add("Whale", "has", "blubber")      # Override: blubber, not fur

# Demonstrate the knowledge graph
print("=" * 60)
print("KNOWLEDGE GRAPH — ANIMAL ONTOLOGY")
print("=" * 60)

# Show inheritance for Penguin
print("\nPenguin's properties (with inheritance):")
props = kg.get_properties("Penguin", inherited=True)
for key, val in props.items():
    print(f"  {key}: {val}")

print("\nPenguin's ancestors:")
ancestors = kg.get_ancestors("Penguin")
print(f"  Penguin -> {' -> '.join(ancestors)}")

# Show inheritance for Eagle
print("\nEagle's properties (with inheritance):")
props = kg.get_properties("Eagle", inherited=True)
for key, val in props.items():
    print(f"  {key}: {val}")

# Query: all things that can swim
print("\n--- Query: 'What can swim?' ---")
swim_results = kg.query(predicate="can", obj="swim")
for s, p, o in swim_results:
    print(f"  {s} can swim")

# ============================================================
# Part 2: Similarity Computation
# ============================================================

print("\n" + "=" * 60)
print("ENTITY SIMILARITY (Jaccard on properties)")
print("=" * 60)

pairs = [
    ("Eagle", "Sparrow"),
    ("Dog", "Cat"),
    ("Penguin", "Eagle"),
    ("Dog", "Salmon"),
    ("Whale", "Shark"),
    ("Penguin", "Salmon"),
]

for e1, e2 in pairs:
    sim = kg.find_similar(e1, e2)
    bar = "#" * int(sim * 30)
    print(f"  {e1:>10} vs {e2:<10}  similarity: {sim:.2f}  {bar}")

# ============================================================
# Part 3: Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Knowledge graph structure
ax = axes[0]
# Manual layout for clarity
positions = {
    "LivingThing": (5, 6),
    "Animal": (5, 5),
    "Bird": (2, 4), "Mammal": (5, 4), "Fish": (8, 4),
    "Penguin": (0.5, 3), "Eagle": (2, 3), "Sparrow": (3.5, 3),
    "Dog": (4.5, 3), "Cat": (5.5, 3), "Whale": (6.5, 3),
    "Salmon": (7.5, 3), "Shark": (9, 3),
}

# Draw is_a edges
for s, p, o in kg.triples:
    if p == "is_a" and s in positions and o in positions:
        sx, sy = positions[s]
        ox, oy = positions[o]
        ax.annotate("", xy=(ox, oy - 0.15), xytext=(sx, sy + 0.15),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

# Draw entity nodes
colors_map = {
    "LivingThing": "#95a5a6", "Animal": "#3498db",
    "Bird": "#e74c3c", "Mammal": "#2ecc71", "Fish": "#f39c12",
}
for entity, (x, y) in positions.items():
    # Determine color by ancestor
    color = "#bdc3c7"
    if entity in colors_map:
        color = colors_map[entity]
    else:
        for ancestor_type, c in [("Bird", "#e74c3c"), ("Mammal", "#2ecc71"), ("Fish", "#f39c12")]:
            if ancestor_type in kg.get_ancestors(entity):
                color = c
                break

    ax.scatter(x, y, s=800, c=color, zorder=5, edgecolors="black", linewidth=1.5)
    ax.text(x, y - 0.35, entity, ha="center", va="top", fontsize=8, fontweight="bold")

ax.set_xlim(-1, 10)
ax.set_ylim(2.2, 6.8)
ax.set_title("Animal Ontology (is-a hierarchy)", fontsize=13, fontweight="bold")
ax.set_aspect("equal")
ax.axis("off")

# Right: Property inheritance visualization
ax = axes[1]

entities_to_show = ["Penguin", "Eagle", "Dog", "Whale", "Salmon"]
all_properties = set()
entity_props = {}
for e in entities_to_show:
    props = kg.get_properties(e, inherited=True)
    entity_props[e] = props
    all_properties.update(props.keys())

all_properties = sorted(all_properties)

# Create a matrix
matrix = np.zeros((len(entities_to_show), len(all_properties)))
for i, entity in enumerate(entities_to_show):
    for j, prop in enumerate(all_properties):
        if prop in entity_props[entity]:
            matrix[i, j] = 1.0

ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(all_properties)))
ax.set_xticklabels(all_properties, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(entities_to_show)))
ax.set_yticklabels(entities_to_show, fontsize=10)
ax.set_title("Inherited Properties Matrix", fontsize=13, fontweight="bold")

for i in range(len(entities_to_show)):
    for j in range(len(all_properties)):
        prop = all_properties[j]
        if prop in entity_props[entities_to_show[i]]:
            val = entity_props[entities_to_show[i]][prop]
            ax.text(j, i, val, ha="center", va="center", fontsize=7, color="black")

plt.tight_layout()
plt.show()

# ============================================================
# Part 4: Frame-Based Representation
# ============================================================

print("\n" + "=" * 60)
print("FRAME-BASED REPRESENTATION")
print("=" * 60)

class Frame:
    """A frame (template) for representing structured knowledge."""

    def __init__(self, name, superclass=None):
        self.name = name
        self.superclass = superclass
        self.slots = {}

    def set_slot(self, name, value, facet="value"):
        self.slots[name] = {"value": value, "facet": facet}

    def get_slot(self, name):
        """Get a slot value, using inheritance if not found locally."""
        if name in self.slots:
            return self.slots[name]["value"]
        elif self.superclass is not None:
            return self.superclass.get_slot(name)
        return None

    def describe(self):
        print(f"\n  Frame: {self.name}")
        if self.superclass:
            print(f"  Superclass: {self.superclass.name}")
        all_slots = set()
        frame = self
        while frame:
            all_slots.update(frame.slots.keys())
            frame = frame.superclass
        for slot_name in sorted(all_slots):
            value = self.get_slot(slot_name)
            source = "local" if slot_name in self.slots else "inherited"
            print(f"    {slot_name}: {value} ({source})")

# Build frame hierarchy
animal_frame = Frame("Animal")
animal_frame.set_slot("has_cells", True)
animal_frame.set_slot("can_move", True)

bird_frame = Frame("Bird", superclass=animal_frame)
bird_frame.set_slot("has_feathers", True)
bird_frame.set_slot("can_fly", True)
bird_frame.set_slot("has_beak", True)

penguin_frame = Frame("Penguin", superclass=bird_frame)
penguin_frame.set_slot("can_fly", False)  # Override!
penguin_frame.set_slot("can_swim", True)
penguin_frame.set_slot("habitat", "Antarctic")

eagle_frame = Frame("Eagle", superclass=bird_frame)
eagle_frame.set_slot("diet", "carnivore")
eagle_frame.set_slot("wingspan", "large")

for frame in [animal_frame, bird_frame, penguin_frame, eagle_frame]:
    frame.describe()

print("\n\nKey insight: Penguin inherits 'has_feathers' and 'has_beak'")
print("from Bird, but OVERRIDES 'can_fly' to False.")
print("This is the power of frame-based knowledge representation.")
```

---

## Key Takeaways

- **Knowledge representation is the foundation of AI reasoning** — the choice of how to represent facts, relationships, and hierarchies determines what inferences are possible
- **Graphs and semantic networks** are flexible and intuitive, representing entities as nodes and relationships as edges; they underpin modern knowledge graphs used by Google, Wikidata, and others
- **Frames provide template-based representations** with default values and inheritance, naturally modeling exceptions (like a penguin being a bird that cannot fly)
- **Ontologies add formal semantics** that enable automated reasoning and inference, but at the cost of complexity in knowledge engineering
- **No single representation is best for all tasks** — graphs excel at relationship queries, frames handle defaults and exceptions, and ontologies enable rigorous logical inference
