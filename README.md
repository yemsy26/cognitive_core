# Cognitive Core (Cerebro Autónomo)

Un motor cognitivo autónomo construido en Rust. Este proyecto es el **núcleo central** ("Cerebro") de un asistente de inteligencia artificial avanzado y modular diseñado por **Ramon (yemsy26)**. 

Este núcleo funciona de forma completamente independiente de interfaces externas (UI, voz, etc.) para asegurar una separación de responsabilidades robusta. Otros módulos consumirán este cerebro como una dependencia independiente para evitar la contaminación cruzada del código a medida que los subsistemas crezcan.

## 🚀 Capacidades del Núcleo Cognitivo

El sistema no es un simple envoltorio para interactuar con un LLM; implementa un ciclo cognitivo real:

1. **Memoria Semántica Persistente (RAG Interno):**
   - Utiliza `SQLite` y una arquitectura orientada a grafos para almacenar entidades y relaciones (`Nodos` y `Aristas`).
   - Genera *Embeddings* vectoriales locales de alta velocidad utilizando `fastembed` para relacionar conceptos semánticamente.
   - Soluciona de forma definitiva la "amnesia del RLHF" mediante un patrón de **Tool Result Injection**. Los modelos quantizados confían en la memoria recuperada considerándola una verdad inmutable comprobada (ground truth) en lugar de dudar de ella.

2. **Razonamiento Chain-of-Thought (CoT) de 3 Fases:**
   - **Fase 1 (Deliberación):** El motor recupera hechos almacenados en su memoria a largo plazo y delibera un borrador en base a sus recuerdos.
   - **Fase 2 (Auditoría/Validación):** Un circuito crítico analiza estrictamente si el borrador usa correctamente la información y no está inventando datos o sufriendo de alucinaciones.
   - **Fase 3 (Expresión):** Basándose en los resultados aprobados de las fases anteriores, adapta el estilo del mensaje al tono y la fluidez requerida sin perder la perspectiva de primera persona.

3. **Curiosidad Autónoma:**
   - Inspirado por el modelo psicológico PAD (*Pleasure-Arousal-Dominance*).
   - El sistema tiene un "latido" (*Activity Pulse*). Si el umbral de *Arousal* (excitación/curiosidad) supera ciertos límites por inactividad, el motor indaga su propia memoria y genera preguntas internas. Literalmente, **piensa de forma autónoma**.

4. **100% Local y Privado:**
   - La inferencia se realiza completamente en hardware local a través de `llama-cpp-2` con aceleración **Vulkan**.

## 🛠️ Tecnologías

- **Rust:** Rendimiento, seguridad y concurrencia pura.
- **llama-cpp-2 (Vulkan backend):** Inferencia LLM altamente eficiente para hardware local (RTX series).
- **fastembed:** Creación de embeddings ultrarrápida sin dependencias externas pesadas.
- **SQLite (rusqlite):** Base de datos embebida ligera para la persistencia del grafo de memoria.

## ⚙️ Arquitectura de Actores

El núcleo se divide en tres *Actores* principales (canales MPSC):

- **Perception Actor:** Recibe estímulos del entorno (o de la consola).
- **Cognition Actor:** Responsable de la inferencia LLM y el Pipeline CoT de 3 fases.
- **Memory Actor:** Mantiene el grafo conceptual local, gestiona los embeddings y evalúa la homeostasis emocional (PAD).

## 💻 Instalación y Uso

**Requisitos previos:**
1. Rust (`rustup default stable`)
2. CMake y Ninja (para compilar dependencias de `llama.cpp`)
3. Un modelo GGUF (ej. `Qwen2.5` o `Llama3`) configurado en la constante `MODEL_PATH` de `src/main.rs`.

**Compilar y Ejecutar:**

```bash
# Se recomienda usar Ninja en Windows para evitar problemas con paths largos
$env:CARGO_TARGET_DIR="C:\tmp\cg_core"; $env:CMAKE_GENERATOR="Ninja"; cargo run --release
```

## 📜 Licencia

Desarrollado bajo licencia **MIT**. 
Creado y diseñado por **Ramon (yemsy26)**. Ver el archivo `LICENSE` para más detalles.
