use rusqlite::{params, Connection, Result as SqliteResult};
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use crate::messages::SystemMessage;
use fastembed::TextEmbedding;
use bytemuck;

/// Factor de decaimiento: media-vida de 7 días.
const DECAY_LAMBDA: f64 = 1.0 / (86400.0 * 7.0);

const AROUSAL_INCREMENT: f64 = 0.05;
const AROUSAL_DECAY: f64 = 0.92;
const AROUSAL_MAX: f64 = 1.0;

#[allow(dead_code)]
pub struct MemoryActor {
    rx: mpsc::Receiver<SystemMessage>,
    db_conn: Connection,
    embedding_model: TextEmbedding,
}

impl MemoryActor {
    pub fn new(rx: mpsc::Receiver<SystemMessage>) -> Self {
        let db_conn = Connection::open("cognitive_memory.db")
            .expect("Error al abrir o crear cognitive_memory.db");
        Self::init_schema(&db_conn).expect("Error al inicializar el esquema de la base de datos");
        
        let embedding_model = TextEmbedding::try_new(Default::default())
            .unwrap_or_else(|e| panic!("Error inicializando fastembed: {}", e));

        Self { rx, db_conn, embedding_model }
    }

    fn init_schema(conn: &Connection) -> SqliteResult<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                concept TEXT UNIQUE NOT NULL,
                first_seen INTEGER NOT NULL,
                last_accessed INTEGER NOT NULL,
                access_count INTEGER NOT NULL,
                emotional_valence REAL NOT NULL,
                embedding BLOB
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS edges (
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                strength REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, relation),
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS personality_state (
                id INTEGER PRIMARY KEY,
                pleasure REAL,
                arousal REAL,
                dominance REAL
            )",
            [],
        )?;

        let count: i64 = conn.query_row("SELECT COUNT(*) FROM personality_state", [], |row| row.get(0))?;
        if count == 0 {
            conn.execute(
                "INSERT INTO personality_state (id, pleasure, arousal, dominance) VALUES (1, 0.0, 0.0, 0.0)",
                [],
            )?;
        }
        Ok(())
    }

    fn current_timestamp() -> i64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
    }

    fn get_pad(&self) -> (f64, f64, f64) {
        self.db_conn.query_row(
            "SELECT pleasure, arousal, dominance FROM personality_state WHERE id = 1",
            [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).unwrap_or((0.0, 0.0, 0.0))
    }

    fn set_pad(&self, pleasure: f64, arousal: f64, dominance: f64) {
        let _ = self.db_conn.execute(
            "UPDATE personality_state SET pleasure = ?1, arousal = ?2, dominance = ?3 WHERE id = 1",
            params![pleasure, arousal, dominance],
        );
    }

    fn upsert_node(&mut self, concept: &str, valence: f64) -> SqliteResult<i64> {
        let timestamp = Self::current_timestamp();
        
        let node = {
            let mut stmt = self.db_conn.prepare(
                "SELECT id, access_count, emotional_valence FROM nodes WHERE concept = ?1"
            )?;
            stmt.query_row(params![concept], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, f64>(2)?))
            }).ok()
        };

        match node {
            Some((id, count, current_valence)) => {
                let new_valence = (current_valence * count as f64 + valence) / (count as f64 + 1.0);
                self.db_conn.execute(
                    "UPDATE nodes SET last_accessed=?1, access_count=?2, emotional_valence=?3 WHERE id=?4",
                    params![timestamp, count + 1, new_valence, id],
                )?;
                Ok(id)
            }
            None => {
                let embedding = self.compute_embedding(concept);
                let blob = Self::serialize_embedding(&embedding);
                self.db_conn.execute(
                    "INSERT INTO nodes (concept, first_seen, last_accessed, access_count, emotional_valence, embedding)
                     VALUES (?1, ?2, ?3, 1, ?4, ?5)",
                    params![concept, timestamp, timestamp, valence, blob],
                )?;
                Ok(self.db_conn.last_insert_rowid())
            }
        }
    }

    fn create_edge(&mut self, source_id: i64, target_id: i64, relation: &str) -> SqliteResult<()> {
        let mut stmt = self.db_conn.prepare(
            "SELECT strength FROM edges WHERE source_id = ?1 AND target_id = ?2 AND relation = ?3"
        )?;
        let edge = stmt.query_row(params![source_id, target_id, relation], |row| {
            row.get::<_, f64>(0)
        });
        match edge {
            Ok(strength) => {
                let new_strength = (strength + 1.0).min(10.0);
                self.db_conn.execute(
                    "UPDATE edges SET strength = ?1 WHERE source_id = ?2 AND target_id = ?3 AND relation = ?4",
                    params![new_strength, source_id, target_id, relation],
                )?;
            }
            Err(_) => {
                self.db_conn.execute(
                    "INSERT INTO edges (source_id, target_id, relation, strength) VALUES (?1, ?2, ?3, 1.0)",
                    params![source_id, target_id, relation],
                )?;
            }
        }
        Ok(())
    }

    fn compute_embedding(&mut self, text: &str) -> Vec<f32> {
        let embeddings = self.embedding_model.embed(vec![text], None).unwrap_or_default();
        embeddings.into_iter().next().unwrap_or_default()
    }

    fn serialize_embedding(embedding: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(embedding).to_vec()
    }

    fn deserialize_embedding(bytes: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice(bytes).to_vec()
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        if a.is_empty() || b.is_empty() || a.len() != b.len() {
            return 0.0;
        }
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { (dot_product / (norm_a * norm_b)) as f64 }
    }

    /// Similitud de Jaccard entre dos conjuntos de palabras (Fallback).
    fn jaccard_sim(a_words: &[String], b_concept: &str) -> f64 {
        let a_set: HashSet<&str> = a_words.iter().map(|w| w.as_str()).collect();
        let b_set: HashSet<&str> = b_concept.split_whitespace().collect();
        let intersection = a_set.intersection(&b_set).count();
        let union = a_set.union(&b_set).count();
        if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
    }

    /// Recupera hasta `limit` nodos relevantes usando búsqueda vectorial semántica
    fn find_relevant_node_ids(&mut self, concept: &str, limit: usize) -> Vec<i64> {
        let now = Self::current_timestamp();
        let query_embedding = self.compute_embedding(concept);

        let mut stmt = match self.db_conn.prepare("SELECT id, access_count, last_accessed, concept, embedding FROM nodes") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let mut candidates: Vec<(i64, f64)> = Vec::new();

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<Vec<u8>>>(4)?,
            ))
        }).unwrap();

        for row in rows.flatten() {
            let (id, count, last_accessed, concept_text, embedding_blob) = row;
            let age = (now - last_accessed).max(0) as f64;
            
            let mut semantic_sim = 0.0;
            if let Some(blob) = embedding_blob {
                let node_embedding = Self::deserialize_embedding(&blob);
                semantic_sim = Self::cosine_similarity(&query_embedding, &node_embedding);
            }
            
            // Fallback si no hay embedding
            if semantic_sim == 0.0 {
                let keywords: Vec<String> = concept.split_whitespace().map(|w| w.to_lowercase()).collect();
                semantic_sim = Self::jaccard_sim(&keywords, &concept_text.to_lowercase());
            }

            semantic_sim = semantic_sim.max(0.0);
            let score = (count as f64).sqrt() * (-DECAY_LAMBDA * age).exp() * semantic_sim;
            
            if score > 0.1 {
                candidates.push((id, score));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.into_iter().take(limit).map(|(id, _)| id).collect()
    }

    pub async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                SystemMessage::StoreMemory { subject, predicate, object, valence } => {
                    let res_subject = self.upsert_node(&subject, valence);
                    let res_object = self.upsert_node(&object, valence);
                    if let (Ok(s_id), Ok(o_id)) = (res_subject, res_object) {
                        if let Err(e) = self.create_edge(s_id, o_id, &predicate) {
                            eprintln!("[MEMORY] - Error al crear relación: {}", e);
                        } else {
                            println!("[MEMORY] - Grafo actualizado: [{} -> {} -> {}]", subject, predicate, object);
                        }
                    }
                }

                SystemMessage::ActivityPulse => {
                    let (p, a, d) = self.get_pad();
                    self.set_pad(p, (a + AROUSAL_INCREMENT).min(AROUSAL_MAX), d);
                }

                SystemMessage::Tick => {
                    // Con silencio, el arousal sube (curiosidad por aburrimiento).
                    // Cuando el usuario habla, ActivityPulse lo sube más.
                    // Así el sistema tiene un empuje interior real, no simulado.
                    let (p, a, d) = self.get_pad();
                    let boredom_increment = 0.008;
                    self.set_pad(p, (a + boredom_increment).min(AROUSAL_MAX), d);
                }

                SystemMessage::QueryContext { concept, reply_to } => {
                    // Graph RAG Multi-Salto: Recursive CTE en SQLite.
                    // Expande el grafo de conocimiento hasta 3 saltos de profundidad
                    // desde los nodos semánticamente más relevantes para la consulta.
                    // Esto permite deducir cadenas A->B->C que el single-hop nunca vería.
                    let seed_ids = self.find_relevant_node_ids(&concept, 3);
                    let mut context_str = String::new();

                    for seed_id in seed_ids {
                        let sql = "
                            WITH RECURSIVE graph_walk(node_id, depth) AS (
                                SELECT ?1, 0
                                UNION
                                SELECT
                                    CASE
                                        WHEN e.source_id = gw.node_id THEN e.target_id
                                        ELSE e.source_id
                                    END,
                                    gw.depth + 1
                                FROM edges e
                                JOIN graph_walk gw ON (e.source_id = gw.node_id OR e.target_id = gw.node_id)
                                WHERE gw.depth < 3
                            )
                            SELECT DISTINCT n1.concept, e.relation, n2.concept
                            FROM edges e
                            JOIN nodes n1 ON e.source_id = n1.id
                            JOIN nodes n2 ON e.target_id = n2.id
                            WHERE e.source_id IN (SELECT node_id FROM graph_walk)
                               OR e.target_id IN (SELECT node_id FROM graph_walk)
                            ORDER BY e.strength DESC
                            LIMIT 15
                        ";

                        let mut stmt = match self.db_conn.prepare(sql) {
                            Ok(s) => s,
                            Err(_) => continue,
                        };

                        let rows: Vec<String> = stmt.query_map(params![seed_id], |row| {
                            Ok(format!("{} -> {} -> {}",
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?))
                        }).map(|r| r.flatten().collect()).unwrap_or_default();

                        for r in rows {
                            let entry = format!("- {}\n", r);
                            if !context_str.contains(&entry) {
                                context_str.push_str(&entry);
                            }
                        }
                    }

                    if context_str.is_empty() {
                        context_str = "No hay recuerdos previos estructurados sobre este concepto.".to_string();
                    } else {
                        context_str = format!("Sabes que (cadena de conocimiento):\n{}", context_str);
                    }

                    let (current_pleasure, _, _) = self.get_pad();
                    let _ = reply_to.send((context_str, current_pleasure));
                }

                SystemMessage::QueryGaps { reply_to } => {
                    // Nodos con alta frecuencia pero baja densidad de aristas = huecos de conocimiento
                    let mut stmt = self.db_conn.prepare(
                        "SELECT n.concept
                         FROM nodes n
                         LEFT JOIN edges e ON e.source_id = n.id
                         GROUP BY n.id
                         ORDER BY (n.access_count * 1.0 / (COUNT(e.source_id) + 1)) DESC
                         LIMIT 3"
                    ).unwrap();

                    let gaps: Vec<String> = stmt
                        .query_map([], |row| row.get::<_, String>(0))
                        .map(|r| r.flatten().collect())
                        .unwrap_or_default();

                    let (_, arousal, _) = self.get_pad();
                    let _ = reply_to.send((gaps, arousal));
                }

                SystemMessage::Consolidate => {
                    let avg_valence: f64 = self.db_conn.query_row(
                        "SELECT COALESCE(AVG(emotional_valence), 0.0) FROM nodes", [], |row| row.get(0),
                    ).unwrap_or(0.0);

                    let avg_access: f64 = self.db_conn.query_row(
                        "SELECT COALESCE(AVG(access_count), 0.0) FROM nodes", [], |row| row.get(0),
                    ).unwrap_or(0.0);

                    let (p, a, d) = self.get_pad();
                    let new_pleasure  = 0.2 * avg_valence + 0.8 * p;
                    let new_arousal   = a * AROUSAL_DECAY;
                    let new_dominance = 0.1 * (avg_access / 100.0).min(1.0) + 0.9 * d;

                    self.set_pad(new_pleasure, new_arousal, new_dominance);
                    println!(
                        "[MEMORY] - Consolidación nocturna. PAD: [P: {:.4}, A: {:.4}, D: {:.4}]",
                        new_pleasure, new_arousal, new_dominance
                    );
                }

                SystemMessage::Shutdown => {
                    println!("[MEMORY] - Cerrando conexión a la base de datos y apagando actor...");
                    break;
                }

                _ => {}
            }
        }
    }
}
