mod actors;
mod messages;
mod ollama_resolver;

use std::time::Duration;
use tokio::sync::mpsc;
use std::io::{self, Write};

use actors::cognition::CognitionActor;
use actors::heartbeat::HeartbeatActor;
use actors::memory::MemoryActor;
use actors::perception::PerceptionActor;
use messages::SystemMessage;
use ollama_resolver::resolve_best_model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[SISTEMA] - Inicializando Núcleo Cognitivo y Arquitectura de Actores...");

    let (tx_perception, rx_perception) = mpsc::channel::<SystemMessage>(1024);
    let (tx_cognition, rx_cognition) = mpsc::channel::<SystemMessage>(1024);
    let (tx_memory, rx_memory) = mpsc::channel::<SystemMessage>(1024);

    let perception = PerceptionActor::new(rx_perception, tx_cognition.clone());
    
    let ollama_path = "C:\\Users\\yemsy\\.ollama\\models";
    let model_path = match resolve_best_model(ollama_path) {
        Ok(path) => {
            println!("[SISTEMA] - Modelo óptimo localizado en: {}", path);
            path
        }
        Err(e) => panic!("[SISTEMA] - Error crítico al resolver modelo local: {}", e),
    };
    
    let cognition = CognitionActor::new(rx_cognition, tx_memory.clone(), tx_perception.clone(), &model_path);
    let memory = MemoryActor::new(rx_memory);
    let heartbeat = HeartbeatActor::new(tx_memory.clone(), tx_perception.clone(), Duration::from_secs(3));

    tokio::spawn(async move { perception.run().await; });
    tokio::spawn(async move { cognition.run().await; });
    tokio::spawn(async move { memory.run().await; });
    tokio::spawn(async move { heartbeat.run().await; });

    println!("[SISTEMA] - Arquitectura de actores inicializada.");
    
    tokio::time::sleep(Duration::from_millis(500)).await;

    loop {
        print!("Usuario> ");
        io::stdout().flush()?;

        let input = tokio::task::spawn_blocking(|| {
            let mut buf = String::new();
            io::stdin().read_line(&mut buf).unwrap_or(0);
            buf
        }).await?;

        let clean_input = input.trim();

        if clean_input.eq_ignore_ascii_case("exit") || clean_input.eq_ignore_ascii_case("quit") {
            break;
        }

        if !clean_input.is_empty() {
            let _ = tx_perception.send(SystemMessage::Input(clean_input.to_string())).await;
        }
    }

    println!("[SISTEMA] - Señal de apagado recibida.");

    let _ = tx_perception.send(SystemMessage::Shutdown).await;
    let _ = tx_cognition.send(SystemMessage::Shutdown).await;
    let _ = tx_memory.send(SystemMessage::Shutdown).await;

    tokio::time::sleep(Duration::from_millis(500)).await;

    Ok(())
}
