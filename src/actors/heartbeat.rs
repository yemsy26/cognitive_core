use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time;
use crate::messages::SystemMessage;

/// Umbral de curiosidad: curiosidad = arousal + ln(gap_count + 1)
const CURIOSITY_THRESHOLD: f64 = 0.4;

/// Ciclos de enfriamiento entre preguntas autónomas consecutivas.
const CURIOSITY_COOLDOWN_CYCLES: u32 = 4;

#[allow(dead_code)]
pub struct HeartbeatActor {
    tx_memory: mpsc::Sender<SystemMessage>,
    tx_perception: mpsc::Sender<SystemMessage>,
    interval: Duration,
    tick_count: u32,
    /// Concepto sobre el que se hizo la última pregunta autónoma.
    last_curiosity_concept: Option<String>,
    /// Ciclos restantes de enfriamiento antes de poder disparar otra pregunta.
    curiosity_cooldown: u32,
}

impl HeartbeatActor {
    pub fn new(
        tx_memory: mpsc::Sender<SystemMessage>,
        tx_perception: mpsc::Sender<SystemMessage>,
        interval: Duration,
    ) -> Self {
        Self {
            tx_memory,
            tx_perception,
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
                if self.tx_memory.send(SystemMessage::Consolidate).await.is_err() {
                    break;
                }

                let (tx_gaps, rx_gaps) = oneshot::channel::<(Vec<String>, f64)>();
                if self.tx_memory.send(SystemMessage::QueryGaps { reply_to: tx_gaps }).await.is_err() {
                    break;
                }

                if let Ok((gaps, arousal)) = rx_gaps.await {
                    let curiosity = arousal + (gaps.len() as f64).ln_1p();

                    // Bajar el contador de enfriamiento cada ciclo
                    if self.curiosity_cooldown > 0 {
                        self.curiosity_cooldown -= 1;
                    }

                    let can_fire = self.curiosity_cooldown == 0;
                    let is_new_concept = gaps.first()
                        .map(|g| self.last_curiosity_concept.as_deref() != Some(g.as_str()))
                        .unwrap_or(false);

                    if curiosity > CURIOSITY_THRESHOLD && !gaps.is_empty() && can_fire && is_new_concept {
                        let gap = gaps[0].clone();
                        println!("[ANDROIDE] - (autónomo) Curiosidad activada sobre: \"{}\"", gap);
                        let prompt = format!(
                            "¿Qué más puedes contarme sobre '{}' y cómo se relaciona con lo que ya sabes?",
                            gap
                        );
                        let _ = self.tx_perception.send(SystemMessage::Input(prompt)).await;
                        self.last_curiosity_concept = Some(gap);
                        self.curiosity_cooldown = CURIOSITY_COOLDOWN_CYCLES;
                    }
                }

                self.tick_count = 0;
            } else {
                if self.tx_memory.send(SystemMessage::Tick).await.is_err() {
                    break;
                }
            }
        }
    }
}
