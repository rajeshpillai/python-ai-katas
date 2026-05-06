# Text Features

> Phase 7 — Feature Engineering & Pipelines | Kata 7.3

---

## Concept & Intuition

### What problem are we solving?

Machine learning models operate on numbers, but much of the world's data is text. To use text as input to ML models, we need to convert it into numerical feature vectors. This process -- text featurization -- transforms documents into fixed-length vectors that capture their content and meaning.

The simplest approach is **Bag of Words (BoW)**: create a vocabulary of all unique words, then represent each document as a vector of word counts. This ignores word order but captures "what words are present." **TF-IDF** (Term Frequency - Inverse Document Frequency) improves on BoW by downweighting common words (like "the", "is") and upweighting rare, informative words. A word that appears in every document is not useful for distinguishing them; a word that appears in only a few documents is highly informative.

Beyond single words, **n-grams** capture short phrases (bigrams like "not good", trigrams like "very well done") that carry meaning lost in single-word BoW. The trade-off is always between expressiveness and dimensionality: more features capture more nuance but increase the risk of overfitting and computational cost.

### Why naive approaches fail

Using raw word counts treats all words equally -- "the" and "revolutionary" get the same treatment despite vastly different informativeness. Raw counts also favor longer documents (more words = higher counts), creating a bias. TF-IDF solves both problems: it normalizes by document length (TF) and penalizes ubiquitous words (IDF). Ignoring text preprocessing (lowercasing, removing punctuation, stemming) leads to treating "Running", "running", and "runs" as completely different features.

### Mental models

- **Document as a point in word-space**: each unique word is a dimension. A document's position in this space encodes its content.
- **TF-IDF as importance weighting**: TF says "how much does this document care about this word?" IDF says "how special is this word across all documents?"
- **The curse of vocabulary size**: with 10,000 unique words, each document is a point in 10,000-dimensional space. Most dimensions are zero (sparse vectors).

### Visual explanations

```
Bag of Words:
  Vocabulary: [cat, dog, fish, the, a, is, big, small]
  "the big cat"     -> [1, 0, 0, 1, 0, 0, 1, 0]
  "a small dog"     -> [0, 1, 0, 0, 1, 0, 0, 1]
  "the cat is big"  -> [1, 0, 0, 1, 0, 1, 1, 0]

TF-IDF:
  TF(word, doc) = count(word in doc) / total_words_in_doc
  IDF(word) = ln(total_docs / docs_containing_word)
  TF-IDF = TF * IDF

  "the" appears in 2/3 docs: IDF = ln(3/2) = 0.41  (common, low weight)
  "fish" appears in 0/3 docs: not in vocab
  "small" appears in 1/3: IDF = ln(3/1) = 1.10     (rare, high weight)
```

---

## Hands-on Exploration

1. Build a vocabulary from a small corpus and compute Bag of Words vectors. Observe the sparsity of the resulting matrix.
2. Implement TF-IDF from scratch. Compare the weights of common words vs rare words.
3. Add bigram features and show how they capture phrases like "not good" that single words miss.
4. Use the TF-IDF vectors to classify documents (e.g., positive vs negative sentiment) with a simple linear classifier.

---

## Live Code

```rust
fn main() {
    // --- Sample text corpus: movie reviews ---
    let documents = vec![
        ("This movie is great and the acting is wonderful", 1),
        ("Terrible film with bad acting and awful story", 0),
        ("The plot is excellent and characters are amazing", 1),
        ("Worst movie ever made with terrible direction", 0),
        ("Brilliant performances and a fantastic story", 1),
        ("Boring and dull film with no excitement", 0),
        ("A masterpiece of cinema with great direction", 1),
        ("Awful movie that is completely unwatchable", 0),
        ("Outstanding film with superb acting throughout", 1),
        ("Disappointing and badly written terrible movie", 0),
        ("Great story with wonderful character development", 1),
        ("Bad film with poor acting and weak plot", 0),
    ];

    let split = 8; // first 8 for training, last 4 for test

    // --- Text preprocessing ---
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    // Stopwords
    let stopwords: Vec<&str> = vec![
        "a", "an", "the", "is", "are", "was", "were", "and", "or", "but",
        "in", "on", "at", "to", "for", "of", "with", "that", "this", "it",
        "no", "not", "be", "has", "have", "had",
    ];

    fn remove_stopwords(tokens: &[String], stopwords: &[&str]) -> Vec<String> {
        tokens.iter()
            .filter(|t| !stopwords.contains(&t.as_str()))
            .cloned()
            .collect()
    }

    // --- Build vocabulary ---
    let mut vocab: Vec<String> = Vec::new();
    let mut doc_tokens: Vec<Vec<String>> = Vec::new();

    for (text, _) in &documents {
        let tokens = tokenize(text);
        let filtered = remove_stopwords(&tokens, &stopwords);
        for t in &filtered {
            if !vocab.contains(t) {
                vocab.push(t.clone());
            }
        }
        doc_tokens.push(filtered);
    }
    vocab.sort();

    println!("=== Text Feature Extraction ===\n");
    println!("Corpus: {} documents, {} unique terms", documents.len(), vocab.len());
    println!("Vocabulary (sorted): {:?}\n", &vocab[..vocab.len().min(15)]);

    // --- Bag of Words ---
    println!("=== Bag of Words ===\n");

    let mut bow_matrix: Vec<Vec<f64>> = Vec::new();
    for tokens in &doc_tokens {
        let mut row = vec![0.0; vocab.len()];
        for t in tokens {
            if let Some(idx) = vocab.iter().position(|v| v == t) {
                row[idx] += 1.0;
            }
        }
        bow_matrix.push(row);
    }

    // Show BoW for first 3 documents
    for i in 0..3 {
        let non_zero: Vec<String> = vocab.iter().enumerate()
            .filter(|(j, _)| bow_matrix[i][*j] > 0.0)
            .map(|(_, w)| w.clone())
            .collect();
        println!("  Doc {}: \"{}\"", i, documents[i].0);
        println!("    Active words: {:?}", non_zero);
        let nnz = bow_matrix[i].iter().filter(|&&v| v > 0.0).count();
        println!("    Sparsity: {}/{} = {:.1}% non-zero\n",
            nnz, vocab.len(), nnz as f64 / vocab.len() as f64 * 100.0);
    }

    // --- TF-IDF ---
    println!("=== TF-IDF ===\n");

    // Compute IDF
    let n_docs = documents.len() as f64;
    let mut idf: Vec<f64> = Vec::new();
    for (j, word) in vocab.iter().enumerate() {
        let doc_freq = doc_tokens.iter()
            .filter(|tokens| tokens.contains(word))
            .count() as f64;
        idf.push((n_docs / (doc_freq + 1.0)).ln() + 1.0); // smoothed IDF
    }

    // Compute TF-IDF matrix
    let mut tfidf_matrix: Vec<Vec<f64>> = Vec::new();
    for (i, tokens) in doc_tokens.iter().enumerate() {
        let total_words = tokens.len() as f64;
        let mut row = vec![0.0; vocab.len()];
        for (j, word) in vocab.iter().enumerate() {
            let tf = bow_matrix[i][j] / total_words.max(1.0);
            row[j] = tf * idf[j];
        }
        // L2 normalize
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in row.iter_mut() { *v /= norm; }
        }
        tfidf_matrix.push(row);
    }

    // Show top TF-IDF words per document
    println!("Top TF-IDF words per document:");
    for i in 0..4 {
        let mut indexed: Vec<(usize, f64)> = tfidf_matrix[i].iter().enumerate()
            .map(|(j, &v)| (j, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<String> = indexed.iter().take(4)
            .map(|&(j, v)| format!("{}({:.2})", vocab[j], v))
            .collect();
        let label = if documents[i].1 == 1 { "positive" } else { "negative" };
        println!("  Doc {} [{}]: {}", i, label, top.join(", "));
    }

    // Show IDF values: common vs rare words
    println!("\nIDF values (informativeness):");
    let mut idf_sorted: Vec<(usize, f64)> = idf.iter().enumerate()
        .map(|(j, &v)| (j, v)).collect();
    idf_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("  Most common (low IDF):");
    for &(j, v) in idf_sorted.iter().take(5) {
        println!("    {:>15}: IDF={:.3}", vocab[j], v);
    }
    println!("  Most rare (high IDF):");
    for &(j, v) in idf_sorted.iter().rev().take(5) {
        println!("    {:>15}: IDF={:.3}", vocab[j], v);
    }

    // --- Simple classifier using TF-IDF features ---
    println!("\n=== Sentiment Classification with TF-IDF ===\n");

    // Cosine similarity-based nearest centroid classifier
    // Compute centroid for positive and negative classes
    let mut pos_centroid = vec![0.0; vocab.len()];
    let mut neg_centroid = vec![0.0; vocab.len()];
    let mut pos_count = 0.0;
    let mut neg_count = 0.0;

    for i in 0..split {
        if documents[i].1 == 1 {
            for j in 0..vocab.len() { pos_centroid[j] += tfidf_matrix[i][j]; }
            pos_count += 1.0;
        } else {
            for j in 0..vocab.len() { neg_centroid[j] += tfidf_matrix[i][j]; }
            neg_count += 1.0;
        }
    }
    for j in 0..vocab.len() {
        pos_centroid[j] /= pos_count;
        neg_centroid[j] /= neg_count;
    }

    fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-10 || nb < 1e-10 { return 0.0; }
        dot / (na * nb)
    }

    // Classify test documents
    println!("Test set predictions:");
    let mut correct = 0;
    for i in split..documents.len() {
        let sim_pos = cosine_sim(&tfidf_matrix[i], &pos_centroid);
        let sim_neg = cosine_sim(&tfidf_matrix[i], &neg_centroid);
        let pred = if sim_pos > sim_neg { 1 } else { 0 };
        let actual = documents[i].1;
        if pred == actual { correct += 1; }
        let status = if pred == actual { "correct" } else { "WRONG" };
        println!("  \"{}\"", documents[i].0);
        println!("    sim_pos={:.3}, sim_neg={:.3}, pred={}, actual={} [{}]",
            sim_pos, sim_neg, pred, actual, status);
    }
    let accuracy = correct as f64 / (documents.len() - split) as f64;
    println!("\nTest accuracy: {}/{} = {:.2}%", correct, documents.len() - split, accuracy * 100.0);

    // --- Bigrams ---
    println!("\n=== Bigram Features ===");
    fn bigrams(tokens: &[String]) -> Vec<String> {
        tokens.windows(2).map(|w| format!("{}_{}", w[0], w[1])).collect()
    }

    println!("Example bigrams:");
    for i in 0..3 {
        let bgs = bigrams(&doc_tokens[i]);
        println!("  Doc {}: {:?}", i, &bgs[..bgs.len().min(5)]);
    }

    println!();
    println!("kata_metric(\"vocabulary_size\", {})", vocab.len());
    println!("kata_metric(\"test_accuracy\", {:.2})", accuracy);
    println!("kata_metric(\"n_documents\", {})", documents.len());
}
```

---

## Key Takeaways

- **Bag of Words converts text to numerical vectors by counting word occurrences.** Simple but effective as a baseline for text classification.
- **TF-IDF improves on BoW by downweighting common words and upweighting rare, informative ones.** This makes the representation more discriminative.
- **Text preprocessing (lowercasing, stopword removal, tokenization) is critical.** Without it, the vocabulary explodes and features become noisy.
- **N-grams capture multi-word phrases** that carry meaning lost in single-word representations, at the cost of increased dimensionality.
