use tokio::sync::oneshot;

#[allow(dead_code)]
#[derive(Debug)]
pub enum SystemMessage {
    Tick,
    Consolidate,
    /// Señal de actividad: incrementa arousal en MemoryActor.
    ActivityPulse,
    Input(String),
    StoreMemory {
        subject: String,
        predicate: String,
        object: String,
        valence: f64,
    },
    QueryContext {
        concept: String,
        reply_to: oneshot::Sender<(String, f64)>,
    },
    /// Solicita al MemoryActor los conceptos con mayor densidad de huecos
    /// y el arousal actual. Retorna (huecos, arousal).
    QueryGaps {
        reply_to: oneshot::Sender<(Vec<String>, f64)>,
    },
    /// Dispara el Motor de Consciencia Interna. El concepto es el núcleo de
    /// pensamiento sobre el cual el sistema generará un monólogo real con el LLM.
    AutonomousThought {
        concept: String,
        context: String,
        pleasure: f64,
    },
    Output(String),
    Shutdown,
}
