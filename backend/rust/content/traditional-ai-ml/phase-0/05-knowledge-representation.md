# Knowledge Representation

> Phase 0 — What is AI? | Kata 0.05

---

## Concept & Intuition

### What problem are we solving?

For an AI system to reason about the world, it needs a way to represent what it knows. Knowledge representation is the study of how to encode information — facts, relationships, hierarchies, constraints — in a form that a computer can manipulate. The choice of representation profoundly affects what kinds of reasoning are possible and how efficient that reasoning will be.

Consider a simple question: "Is a penguin a bird?" To answer this, the system needs to represent the concept of inheritance (penguin IS-A bird), exceptions (birds typically fly, penguins do not), and properties (birds have feathers, penguins have feathers). Different representations handle these challenges differently: semantic networks use graphs, frames use structured records, and ontologies use formal logic.

In this kata, we build a simple knowledge graph — a network of entities and relationships — and implement basic reasoning over it. Knowledge graphs are the backbone of modern AI systems like Google's Knowledge Graph, Wikidata, and enterprise knowledge management systems.

### Why naive approaches fail

Storing knowledge as flat key-value pairs loses all relational structure. You can look up "penguin -> has_feathers: true" but cannot answer "what other things have feathers?" without scanning everything. Storing it as nested if-else statements makes it impossible to add new knowledge without modifying code. A proper knowledge representation supports both efficient lookup and flexible reasoning, allowing the system to answer questions that were never explicitly programmed.

### Mental models

- **Knowledge as a graph**: Entities are nodes, relationships are labeled edges. Reasoning is graph traversal. "Is penguin a bird?" becomes "Is there an IS-A path from penguin to bird?"
- **Inheritance with exceptions**: Properties flow down the IS-A hierarchy unless explicitly overridden. Penguins inherit "has feathers" from bird but override "can fly" to false.
- **Open-world assumption**: The absence of a fact does not mean it is false — it means we do not know. This is different from databases which use a closed-world assumption.

### Visual explanations

```
  Knowledge Graph:

  [Animal] ----has----> [metabolism]
     ^
     |  IS-A
     |
  [Bird] -----has----> [feathers]
     |   \---can----> [fly]
     |
     |  IS-A
     |
  [Penguin] --can--> [swim]
     |   \---can't-> [fly]  (overrides Bird.can_fly)
     |
     |  IS-A
     |
  [Emperor Penguin] --lives_in--> [Antarctica]
```

---

## Hands-on Exploration

1. Build a knowledge graph with entities, relationships, and properties.
2. Implement IS-A inheritance so properties flow from parent to child.
3. Support property overriding — a child can override an inherited property.
4. Query the graph: "What can a penguin do?", "What lives in cold climates?"

---

## Live Code

```rust
fn main() {
    println!("=== Knowledge Representation: Knowledge Graph ===\n");

    let mut kg = KnowledgeGraph::new();

    // Build taxonomy (IS-A hierarchy)
    kg.add_relation("animal", "is_a", "living_thing");
    kg.add_relation("bird", "is_a", "animal");
    kg.add_relation("mammal", "is_a", "animal");
    kg.add_relation("penguin", "is_a", "bird");
    kg.add_relation("eagle", "is_a", "bird");
    kg.add_relation("emperor_penguin", "is_a", "penguin");
    kg.add_relation("dog", "is_a", "mammal");
    kg.add_relation("cat", "is_a", "mammal");

    // Add properties
    kg.add_property("living_thing", "has_metabolism", "true");
    kg.add_property("animal", "can_move", "true");
    kg.add_property("bird", "has_feathers", "true");
    kg.add_property("bird", "can_fly", "true");
    kg.add_property("bird", "has_wings", "true");
    kg.add_property("mammal", "has_fur", "true");
    kg.add_property("mammal", "warm_blooded", "true");
    kg.add_property("penguin", "can_fly", "false"); // Override!
    kg.add_property("penguin", "can_swim", "true");
    kg.add_property("emperor_penguin", "lives_in", "antarctica");
    kg.add_property("eagle", "can_hunt", "true");
    kg.add_property("dog", "is_domestic", "true");
    kg.add_property("cat", "is_domestic", "true");

    // Add other relationships
    kg.add_relation("penguin", "eats", "fish");
    kg.add_relation("eagle", "eats", "rodents");
    kg.add_relation("dog", "eats", "meat");
    kg.add_relation("cat", "eats", "meat");

    // Query 1: IS-A reasoning
    println!("--- IS-A Queries ---");
    let queries = vec![
        ("penguin", "bird"),
        ("penguin", "animal"),
        ("emperor_penguin", "bird"),
        ("dog", "bird"),
        ("cat", "animal"),
    ];
    for (entity, category) in &queries {
        let result = kg.is_a(entity, category);
        println!("Is {} a {}? {}", entity, category, result);
    }

    // Query 2: Property inheritance
    println!("\n--- Property Inheritance ---");
    let entities = vec!["emperor_penguin", "penguin", "eagle", "dog", "cat"];
    for entity in &entities {
        println!("\nProperties of {}:", entity);
        let props = kg.get_all_properties(entity);
        for (key, value) in &props {
            println!("  {} = {}", key, value);
        }
    }

    // Query 3: Reverse queries - "what can fly?"
    println!("\n--- Reverse Queries ---");
    let flyers = kg.find_entities_with_property("can_fly", "true");
    println!("Entities that can fly: {:?}", flyers);

    let swimmers = kg.find_entities_with_property("can_swim", "true");
    println!("Entities that can swim: {:?}", swimmers);

    let domestic = kg.find_entities_with_property("is_domestic", "true");
    println!("Domestic entities: {:?}", domestic);

    // Query 4: Relationship queries
    println!("\n--- Relationship Queries ---");
    let meat_eaters = kg.find_by_relation("eats", "meat");
    println!("Entities that eat meat: {:?}", meat_eaters);

    let fish_eaters = kg.find_by_relation("eats", "fish");
    println!("Entities that eat fish: {:?}", fish_eaters);

    // Metrics
    let total_entities = kg.entities().len();
    let total_relations = kg.relation_count();
    let total_properties = kg.property_count();

    kata_metric("total_entities", total_entities as f64);
    kata_metric("total_relations", total_relations as f64);
    kata_metric("total_properties", total_properties as f64);

    // Check penguin override
    let penguin_fly = kg.get_property("penguin", "can_fly");
    let eagle_fly = kg.get_property("eagle", "can_fly");
    println!("\n--- Override Verification ---");
    println!("Penguin can_fly: {:?} (overridden)", penguin_fly);
    println!("Eagle can_fly: {:?} (inherited)", eagle_fly);
}

struct KnowledgeGraph {
    // (subject, predicate, object)
    relations: Vec<(String, String, String)>,
    // (entity, property_name, property_value)
    properties: Vec<(String, String, String)>,
}

impl KnowledgeGraph {
    fn new() -> Self {
        KnowledgeGraph {
            relations: Vec::new(),
            properties: Vec::new(),
        }
    }

    fn add_relation(&mut self, subject: &str, predicate: &str, object: &str) {
        self.relations.push((
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        ));
    }

    fn add_property(&mut self, entity: &str, key: &str, value: &str) {
        self.properties.push((
            entity.to_string(),
            key.to_string(),
            value.to_string(),
        ));
    }

    fn is_a(&self, entity: &str, category: &str) -> bool {
        if entity == category {
            return true;
        }
        // Find direct parents
        for (subj, pred, obj) in &self.relations {
            if subj == entity && pred == "is_a" {
                if obj == category || self.is_a(obj, category) {
                    return true;
                }
            }
        }
        false
    }

    fn get_ancestors(&self, entity: &str) -> Vec<String> {
        let mut ancestors = Vec::new();
        let mut current = vec![entity.to_string()];
        while !current.is_empty() {
            let mut next = Vec::new();
            for c in &current {
                for (subj, pred, obj) in &self.relations {
                    if subj == c && pred == "is_a" {
                        if !ancestors.contains(obj) {
                            ancestors.push(obj.clone());
                            next.push(obj.clone());
                        }
                    }
                }
            }
            current = next;
        }
        ancestors
    }

    fn get_property(&self, entity: &str, key: &str) -> Option<String> {
        // Check entity's own properties first (allows overriding)
        for (e, k, v) in &self.properties {
            if e == entity && k == key {
                return Some(v.clone());
            }
        }
        // Then check ancestors in order
        let ancestors = self.get_ancestors(entity);
        for ancestor in &ancestors {
            for (e, k, v) in &self.properties {
                if e == ancestor && k == key {
                    return Some(v.clone());
                }
            }
        }
        None
    }

    fn get_all_properties(&self, entity: &str) -> Vec<(String, String)> {
        let mut result: Vec<(String, String)> = Vec::new();
        let mut seen_keys: Vec<String> = Vec::new();

        // Collect from entity first, then ancestors (for override semantics)
        let mut chain = vec![entity.to_string()];
        chain.extend(self.get_ancestors(entity));

        for e in &chain {
            for (ent, key, value) in &self.properties {
                if ent == e && !seen_keys.contains(key) {
                    result.push((key.clone(), value.clone()));
                    seen_keys.push(key.clone());
                }
            }
        }
        result
    }

    fn find_entities_with_property(&self, key: &str, value: &str) -> Vec<String> {
        let mut result = Vec::new();
        for entity in self.entities() {
            if let Some(v) = self.get_property(&entity, key) {
                if v == value {
                    result.push(entity);
                }
            }
        }
        result
    }

    fn find_by_relation(&self, predicate: &str, object: &str) -> Vec<String> {
        let mut result = Vec::new();
        for (subj, pred, obj) in &self.relations {
            if pred == predicate && obj == object {
                result.push(subj.clone());
            }
        }
        result
    }

    fn entities(&self) -> Vec<String> {
        let mut entities = Vec::new();
        for (subj, _, _) in &self.relations {
            if !entities.contains(subj) {
                entities.push(subj.clone());
            }
        }
        for (ent, _, _) in &self.properties {
            if !entities.contains(ent) {
                entities.push(ent.clone());
            }
        }
        entities
    }

    fn relation_count(&self) -> usize {
        self.relations.len()
    }

    fn property_count(&self) -> usize {
        self.properties.len()
    }
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- Knowledge representation determines what questions an AI system can answer and how efficiently it can reason.
- Knowledge graphs represent entities as nodes and relationships as labeled edges, enabling flexible querying and inference.
- IS-A hierarchies enable property inheritance, reducing redundancy while supporting exceptions through overrides.
- The choice of representation (graphs, frames, logic, embeddings) should match the reasoning tasks required by the application.
