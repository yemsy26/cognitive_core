use tokio::sync::mpsc;
use crate::messages::SystemMessage;

#[allow(dead_code)]
pub struct PerceptionActor {
    rx: mpsc::Receiver<SystemMessage>,
    tx_cognition: mpsc::Sender<SystemMessage>,
}

impl PerceptionActor {
    pub fn new(rx: mpsc::Receiver<SystemMessage>, tx_cognition: mpsc::Sender<SystemMessage>) -> Self {
        Self { rx, tx_cognition }
    }

    pub async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                SystemMessage::Input(data) => {
                    println!("[PERCEPTION] - Input externo recibido: {}", data);
                    if self.tx_cognition.send(SystemMessage::Input(data)).await.is_err() {
                        println!("[PERCEPTION] - Error enviando al bus cognitivo.");
                    }
                }
                SystemMessage::Output(text) => {
                    println!("[ANDROIDE] - \"{}\"", text);
                }
                SystemMessage::Shutdown => {
                    println!("[PERCEPTION] - Apagando actor...");
                    break;
                }
                _ => {}
            }
        }
    }
}
