use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time;
use crate::messages::SystemMessage;

/// Umbral de curiosidad: cuando el arousal supera esto el sistema genera pensamiento autónomo.
const CURIOSITY_THRESHOLD: f64 = 0.35;

/// Ciclos de enfriamiento entre pensamientos autónomos consecutivos.
const CURIOSITY_COOLDOWN_CYCLES: u32 = 6;

#[allow(dead_code)]
pub struct HeartbeatActor {
    tx_memory: mpsc::Sender<SystemMessage>,
    tx_cognition: mpsc::Sender<SystemMessage>,
    interval: Duration,
    tick_count: u32,
    last_curiosity_concept: Option<String>,
    curiosity_cooldown: u32,
}

impl HeartbeatActor {
    pub fn new(
        tx_memory: mpsc::Sender<SystemMessage>,
        tx_cognition: mpsc::Sender<SystemMessage>,
        interval: Duration,
    ) -> Self {
        Self {
            tx_memory,
            tx_cognition,
            interval,
            tick_count: 0,
            last_curiosity_concept: None,
            curiosity_cooldown: 0,
        }
    }

    pub async fn run(mut self) {
        let mut interval = time::interval(self.interval);

        loop {
            interval.tick().await;
            self.tick_count += 1;

            if self.tick_count >= 5 {
                // Consolidar el grafo de memoria
                if self.tx_memory.send(SystemMessage::Consolidate).await.is_err() {
                    break;
                }

                // Consultar huecos de conocimiento y arousal actual
                let (tx_gaps, rx_gaps) = oneshot::channel::<(Vec<String>, f64)>();
                if self.tx_memory.send(SystemMessage::QueryGaps { reply_to: tx_gaps }).await.is_err() {
                    break;
                }

                if let Ok((gaps, arousal)) = rx_gaps.await {
                    let curiosity = arousal + (gaps.len() as f64).ln_1p();

                    if self.curiosity_cooldown > 0 {
                        self.curiosity_cooldown -= 1;
                    }

                    let can_fire = self.curiosity_cooldown == 0;
                    let is_new_concept = gaps.first()
                        .map(|g| self.last_curiosity_concept.as_deref() != Some(g.as_str()))
                        .unwrap_or(false);

                    if curiosity > CURIOSITY_THRESHOLD && !gaps.is_empty() && can_fire && is_new_concept {
                        let gap_concept = gaps[0].clone();
                        println!("[ANDROIDE] - (autónomo) Curiosidad activada sobre: \"{}\"", gap_concept);

                        // Recuperar el contexto de memoria sobre ese concepto
                        // y pasarlo al motor de consciencia real del CognitionActor.
                        let (tx_ctx, rx_ctx) = oneshot::channel::<(String, f64)>();
                        let _ = self.tx_memory.send(SystemMessage::QueryContext {
                            concept: gap_concept.clone(),
                            reply_to: tx_ctx,
                        }).await;

                        if let Ok((context, pleasure)) = rx_ctx.await {
                            // Disparar AutonomousThought: el CognitionActor generará
                            // texto real con el LLM sobre este concepto, no un mensaje preprogramado.
                            let _ = self.tx_cognition.send(SystemMessage::AutonomousThought {
                                concept: gap_concept.clone(),
                                context,
                                pleasure,
                            }).await;
                        }

                        self.last_curiosity_concept = Some(gap_concept);
                        self.curiosity_cooldown = CURIOSITY_COOLDOWN_CYCLES;
                    }
                }

                self.tick_count = 0;
            } else {
                // Enviar señal de Tick para que el MemoryActor incremente el arousal por inactividad
                if self.tx_memory.send(SystemMessage::Tick).await.is_err() {
                    break;
                }
            }
        }
    }
}
