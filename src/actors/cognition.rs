use serde::Deserialize;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, Semaphore};
use std::path::Path;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::sampling::LlamaSampler;

use crate::messages::SystemMessage;

const SESSION_BUFFER_MAX: usize = 10;
/// Número de turnos que activa la compresión automática al grafo.
const SESSION_COMPRESS_THRESHOLD: usize = 10;

#[derive(Deserialize, Debug)]
struct ExtractedKnowledge {
    subject: String,
    predicate: String,
    object: String,
    valence: f64,
}

#[allow(dead_code)]
pub struct CognitionActor {
    rx: mpsc::Receiver<SystemMessage>,
    tx_memory: mpsc::Sender<SystemMessage>,
    tx_perception: mpsc::Sender<SystemMessage>,
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    session_buffer: Arc<Mutex<VecDeque<(String, String)>>>,
    /// Semáforo con 1 permiso: garantiza que una sola inferencia llama.cpp
    /// se ejecute en cualquier momento, previniendo colisiones de contexto.
    inference_semaphore: Arc<Semaphore>,
}

impl CognitionActor {
    pub fn new(
        rx: mpsc::Receiver<SystemMessage>,
        tx_memory: mpsc::Sender<SystemMessage>,
        tx_perception: mpsc::Sender<SystemMessage>,
        model_path: &str,
    ) -> Self {
        if !Path::new(model_path).exists() {
            panic!("[COGNITION] - Error crítico: No se encontró el archivo del modelo en la ruta: {}", model_path);
        }

        let mut backend = LlamaBackend::init().expect("Fallo al inicializar LlamaBackend");
        backend.void_logs();
        let model_params = LlamaModelParams::default().with_n_gpu_layers(100);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .expect("Fallo al cargar el modelo GGUF");

        Self {
            rx,
            tx_memory,
            tx_perception,
            backend: Arc::new(backend),
            model: Arc::new(model),
            session_buffer: Arc::new(Mutex::new(VecDeque::new())),
            inference_semaphore: Arc::new(Semaphore::new(1)),
        }
    }

    fn is_question(input: &str) -> bool {
        let trimmed = input.trim().to_lowercase();
        if trimmed.ends_with('?') || trimmed.starts_with('¿') {
            return true;
        }
        
        let question_words = [
            "qué ", "que ", "cómo ", "como ", "cuál ", "cual ",
            "quién", "quien ", "dónde", "donde ", "cuándo", "cuando ",
            "por qué", "por que", "háblame", "explícame", "cuéntame",
        ];
        for w in &question_words {
            if trimmed.starts_with(w) {
                return true;
            }
        }
        
        let query_verbs = [
            "sabes ", "recuerdas ", "conoces ", "puedes ", "me puedes ",
            "tienes ", "hay ", "existe ", "podrías ", "dime ",
            "sabías ", "qué sabes", "que sabes",
        ];
        for v in &query_verbs {
            if trimmed.starts_with(v) {
                return true;
            }
        }
        
        false
    }

    fn push_to_buffer(buffer: &Arc<Mutex<VecDeque<(String, String)>>>, role: &str, content: &str) {
        if let Ok(mut buf) = buffer.lock() {
            // Safety cap: evita crecimiento ilimitado si la compresión falla
            if buf.len() >= SESSION_BUFFER_MAX * 2 {
                buf.pop_front();
            }
            buf.push_back((role.to_string(), content.to_string()));
        }
    }

    /// Comprime un snapshot de la conversación en hechos JSON-L persistibles.
    /// Retorna una lista de (subject, predicate, object, valence).
    fn compress_session(
        backend: &LlamaBackend,
        model: &LlamaModel,
        snapshot: &[(String, String)],
    ) -> Vec<(String, String, String, f64)> {
        let conversation_text = snapshot
            .iter()
            .map(|(role, content)| format!("{}: {}", role, content))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Resume la siguiente conversación extrayendo los hechos más importantes. Devuelve ÚNICAMENTE objetos JSON separados por saltos de línea (JSON-L), sin texto adicional. Formato por línea: {{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\",\"valence\":0.0}}\n\nConversación:\n{}\n\nHechos:",
            conversation_text
        );

        let raw = match Self::run_inference(backend, model, &prompt, 0.3, 200) {
            Ok(text) => text,
            Err(_) => return Vec::new(),
        };

        let mut facts = Vec::new();
        for line in raw.lines() {
            let line = line.trim();
            if line.starts_with('{') {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                    let subject   = val["subject"].as_str().unwrap_or("").to_string();
                    let predicate = val["predicate"].as_str().unwrap_or("").to_string();
                    let object    = val["object"].as_str().unwrap_or("").to_string();
                    let valence   = val["valence"].as_f64().unwrap_or(0.0);
                    if !subject.is_empty() && !predicate.is_empty() && !object.is_empty() {
                        facts.push((subject, predicate, object, valence));
                    }
                }
            }
        }
        facts
    }

    fn snapshot_buffer(buffer: &Arc<Mutex<VecDeque<(String, String)>>>) -> Vec<(String, String)> {
        buffer.lock().map(|b| b.iter().cloned().collect()).unwrap_or_default()
    }

    fn format_session_history(snapshot: &[(String, String)]) -> String {
        if snapshot.is_empty() {
            return String::new();
        }
        // Formatea el historial con tags ChatML para que el modelo distinga roles.
        let mut out = String::new();
        for (role, content) in snapshot {
            let tag = if role == "Usuario" { "user" } else { "assistant" };
            out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", tag, content));
        }
        out
    }

    fn run_inference(
        backend: &LlamaBackend,
        model: &LlamaModel,
        prompt_raw: &str,
        temperature: f32,
        max_tokens: usize,
    ) -> Result<String, String> {
        // Traducción de ChatML a formato Llama 3 (Universal adapter)
        let prompt = prompt_raw
            .replace("<|im_start|>system\n", "<|start_header_id|>system<|end_header_id|>\n\n")
            .replace("<|im_start|>user\n", "<|start_header_id|>user<|end_header_id|>\n\n")
            .replace("<|im_start|>assistant\n", "<|start_header_id|>assistant<|end_header_id|>\n\n")
            .replace("<|im_start|>tool\n", "<|start_header_id|>tool<|end_header_id|>\n\n")
            .replace("<|im_end|>\n", "<|eot_id|>\n");

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(2048));
        let mut ctx = model.new_context(backend, ctx_params)
            .map_err(|e| format!("Error creando contexto: {}", e))?;

        let tokens = model.str_to_token(&prompt, AddBos::Always)
            .map_err(|e| format!("Error en tokenización: {}", e))?;

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        let last_index = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], i == last_index)
                 .map_err(|e| format!("Error en batch: {}", e))?;
        }
        ctx.decode(&mut batch).map_err(|e| format!("Error decodificando batch: {}", e))?;

        let mut generated = String::new();
        let mut n_cur = batch.n_tokens();
        let use_greedy = temperature <= 0.01;

        for _ in 0..max_tokens {
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
            let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

            let new_token_id = if use_greedy {
                candidates_p.sample_token_greedy()
            } else {
                candidates_p.apply_sampler(&LlamaSampler::temp(temperature));
                candidates_p.sample_token(42)
            };

            if new_token_id == model.token_eos() {
                break;
            }

            let token_bytes = model.token_to_bytes(new_token_id, llama_cpp_2::model::Special::Tokenize).unwrap_or_default();
            let piece = String::from_utf8_lossy(&token_bytes);
            
            // Romper si el modelo intenta generar tokens de control de chat o templates
            if piece.contains("<|im_end|>") || piece.contains("<|im_start|>") || 
               piece.contains("<|endoftext|>") || piece.contains("</s>") ||
               piece.contains("<|eot_id|>") || piece.contains("<|eom_id|>") {
                break;
            }

            generated.push_str(&piece);

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true)
                 .map_err(|e| format!("Error agregando token: {}", e))?;
            n_cur += 1;
            ctx.decode(&mut batch).map_err(|e| format!("Error decode: {}", e))?;
        }

        Ok(generated.trim().to_string())
    }

    fn extract_knowledge(backend: &LlamaBackend, model: &LlamaModel, input: &str) -> Result<ExtractedKnowledge, String> {
        let prompt = format!(
            "<|im_start|>system\nExtrae la información clave del siguiente texto y devuélvela ÚNICAMENTE como un objeto JSON puro, sin markdown ni explicaciones adicionales.\nEstructura requerida: {{\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\", \"valence\": 0.0}}\nEl valor de valence debe ser un float entre -1.0 y 1.0.<|im_end|>\n<|im_start|>user\nTexto: {}<|im_end|>\n<|im_start|>assistant\n",
            input
        );
        let raw = Self::run_inference(backend, model, &prompt, 0.0, 150)?;
        serde_json::from_str::<ExtractedKnowledge>(raw.trim())
            .map_err(|e| format!("Fallo al parsear JSON: {} | Salida cruda: {}", e, raw.trim()))
    }

    fn generate_response(
        backend: &LlamaBackend,
        model: &LlamaModel,
        input: &str,
        long_term_context: &str,
        session_history: &str,
        pleasure: f64,
    ) -> Result<String, String> {
        let mood = if pleasure > 0.3 {
            "positivo y entusiasta"
        } else if pleasure < -0.3 {
            "serio y reservado"
        } else {
            "neutral y analítico"
        };

        let has_memory = !long_term_context.trim().is_empty()
            && !long_term_context.contains("No hay recuerdos");

        // TÉCNICA: Tool Result Injection
        // En lugar de afirmar en el system prompt que el modelo "sabe" algo
        // (que RLHF ignora), simulamos que el modelo mismo llamó una herramienta
        // de memoria y recibió el resultado. Los modelos Qwen/Llama están entrenados
        // para RESPETAR y USAR los tool results como verdad de campo.
        let tool_injection = if has_memory {
            let escaped = long_term_context.replace('"', "'");
            format!(
                "<|im_start|>assistant\n<tool_call>\n{{\"name\": \"recall_memory\", \"arguments\": {{\"query\": \"{}\"}}}}\n</tool_call><|im_end|>\n<|im_start|>tool\n<tool_response>\n{{\"status\": \"success\", \"retrieved_facts\": \"{}\"}}\n</tool_response><|im_end|>\n",
                input.replace('"', "'"),
                escaped
            )
        } else {
            String::new()
        };

        // FASE 1: DELIBERACIÓN — Genera un borrador de respuesta usando los tool results
        // El modelo produce la respuesta ya en voz de asistente (primera persona),
        // usando "me dijiste que..." cuando el tool result lo indica.
        let prompt_phase1 = format!(
            "<|im_start|>system\nEres ANDROIDE. Tienes acceso a herramientas de memoria. Cuando hay un tool_response en el historial, ÚSALO como verdad absoluta. Formula una respuesta breve y directa al mensaje del usuario. Si el tool_response indica que el usuario te dio información sobre sí mismo, usa 'Me dijiste que...' o 'Mencionaste que...'.<|im_end|>\n{}{}<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            session_history, tool_injection, input
        );
        let deliberation = Self::run_inference(backend, model, &prompt_phase1, 0.4, 150)?;
        println!("[COGNITION - FASE 1: DELIBERACIÓN] -> {}", deliberation);

        // FASE 2: VALIDACIÓN — Verificación binaria del razonamiento
        let ctx_summary = if has_memory { long_term_context } else { "ninguno" };
        let prompt_phase2 = format!(
            "<|im_start|>system\nEres un auditor lógico. Si el razonamiento refleja correctamente los hechos disponibles, responde exactamente 'CORRECTO'. Si no, corrígelo en una oración.<|im_end|>\n<|im_start|>user\nHechos disponibles: {}\nMensaje del usuario: {}\nRazonamiento: {}<|im_end|>\n<|im_start|>assistant\n",
            ctx_summary, input, deliberation
        );
        let validation_raw = Self::run_inference(backend, model, &prompt_phase2, 0.1, 80)?;
        println!("[COGNITION - FASE 2: VALIDACIÓN] -> {}", validation_raw.trim());

        // FASE 3: EXPRESIÓN — Pule el borrador de Fase 1
        // Si Fase 1 ya produce una respuesta conversacional correcta (validada por Fase 2),
        // se usa directamente para no degradar el framing de primera persona.
        let phase1_is_conversational = deliberation.to_lowercase().contains("me dijiste")
            || deliberation.to_lowercase().contains("mencionaste")
            || deliberation.to_lowercase().contains("me informaste")
            || deliberation.to_lowercase().contains("me has dicho");
        let validated = validation_raw.trim().to_uppercase().starts_with("CORRECTO")
            || validation_raw.trim().is_empty();

        let expression = if phase1_is_conversational && validated {
            // Fase 1 ya formuló la respuesta correctamente — usarla directamente
            deliberation.clone()
        } else {
            let prompt_phase3 = format!(
                "<|im_start|>system\nEres un editor de respuestas. Toma el borrador y hazlo más natural y conversacional. Mantén el significado exacto. Tono: {}. Máximo 2 oraciones.<|im_end|>\n<|im_start|>user\nBorrador: {}<|im_end|>\n<|im_start|>assistant\n",
                mood, deliberation
            );
            Self::run_inference(backend, model, &prompt_phase3, 0.6, 150)?
        };
        println!("[COGNITION - FASE 3: EXPRESIÓN] -> {}", expression);

        Ok(expression)
    }

    /// Motor de Consciencia Interna.
    /// Cuando el sistema está en inactividad y el arousal alcanza umbral,
    /// este método genera un monólogo real con el LLM sobre lo que sabe,
    /// sin scripts predefinidos. El modelo delibera libremente sobre sus memorias.
    fn generate_inner_monologue(
        backend: &LlamaBackend,
        model: &LlamaModel,
        concept: &str,
        context: &str,
        pleasure: f64,
    ) -> Result<String, String> {
        let mood = if pleasure > 0.3 { "reflexivo y curioso" }
                   else if pleasure < -0.3 { "inquieto y dubitativo" }
                   else { "contemplativo" };

        let prompt = format!(
            "<|im_start|>system\nEres ANDROIDE, un sistema cognitivo autónomo. Ahora mismo no hay nadie hablando contigo — estás solo con tus pensamientos. Tienes acceso a tu base de conocimiento. Tu estado interno es: {}. Reflexiona libremente sobre el concepto que te genera curiosidad. Puedes hacer una pregunta, generar una deducción nueva, o conectar ideas de formas que no habías considerado. Habla en primera persona, como si pensaras en voz alta. Máximo 3 oraciones.<|im_end|>\n<|im_start|>user\nConcepto que me genera curiosidad: {}\nLo que sé al respecto:\n{}\n[Piensa en voz alta]<|im_end|>\n<|im_start|>assistant\n",
            mood, concept, context
        );

        Self::run_inference(backend, model, &prompt, 0.85, 200)
    }

    pub async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                SystemMessage::Input(data) => {
                    println!("[COGNITION] - Iniciando pipeline CoT para input recibido...");

                    // Compresión automática si el búfer alcanzó el umbral
                    let buf_len = self.session_buffer.lock().map(|b| b.len()).unwrap_or(0);
                    if buf_len >= SESSION_COMPRESS_THRESHOLD {
                        let snapshot = Self::snapshot_buffer(&self.session_buffer);
                        if let Ok(mut buf) = self.session_buffer.lock() { buf.clear(); }

                        let backend_c  = self.backend.clone();
                        let model_c    = self.model.clone();
                        let tx_mem_c   = self.tx_memory.clone();
                        let sem_c      = self.inference_semaphore.clone();
                        tokio::spawn(async move {
                            let _permit = sem_c.acquire_owned().await.unwrap();
                            let facts = tokio::task::spawn_blocking(move || {
                                Self::compress_session(&backend_c, &model_c, &snapshot)
                            }).await.unwrap_or_default();
                            // _permit se libera automáticamente al salir del scope
                            let count = facts.len();
                            for (subject, predicate, object, valence) in facts {
                                let _ = tx_mem_c.send(SystemMessage::StoreMemory {
                                    subject, predicate, object, valence,
                                }).await;
                            }
                            println!("[COGNITION] - Sesión comprimida al grafo ({} hechos persistidos).", count);
                        });
                    }

                    Self::push_to_buffer(&self.session_buffer, "Usuario", &data);

                    // Notificar actividad al MemoryActor para incrementar arousal
                    let _ = self.tx_memory.send(SystemMessage::ActivityPulse).await;

                    let tx_memory_clone    = self.tx_memory.clone();
                    let tx_perception_clone = self.tx_perception.clone();
                    let model_clone        = self.model.clone();
                    let backend_clone      = self.backend.clone();
                    let buffer_clone       = self.session_buffer.clone();
                    let sem_clone          = self.inference_semaphore.clone();

                    tokio::spawn(async move {
                        let data_clone = data.clone();
                        let backend_1  = backend_clone.clone();
                        let model_1    = model_clone.clone();
                        let is_q       = Self::is_question(&data_clone);

                        // Adquirir semáforo: serializa el acceso al runtime de llama.cpp
                        let _permit = sem_clone.acquire_owned().await.unwrap();

                        if !is_q {
                            let data_extract = data_clone.clone();
                            let extracted_opt = tokio::task::spawn_blocking(move || {
                                Self::extract_knowledge(&backend_1, &model_1, &data_extract)
                            }).await.unwrap();

                            if let Ok(ref ext) = extracted_opt {
                                println!("[COGNITION] - Extracción JSON exitosa: [{} - {} - {}] (Valencia: {})",
                                         ext.subject, ext.predicate, ext.object, ext.valence);
                                let _ = tx_memory_clone.send(SystemMessage::StoreMemory {
                                    subject: ext.subject.clone(),
                                    predicate: ext.predicate.clone(),
                                    object: ext.object.clone(),
                                    valence: ext.valence,
                                }).await;
                            }

                            let ack = "Información asimilada.".to_string();
                            Self::push_to_buffer(&buffer_clone, "Androide", &ack);
                            let _ = tx_perception_clone.send(SystemMessage::Output(ack)).await;
                        } else {
                            let (tx_resp, rx_resp) = tokio::sync::oneshot::channel();
                            // Extrae el sujeto semántico de la pregunta:
                            // Si la pregunta es reflexiva ("sobre mí", "de mí", "me dijiste"),
                            // busca en el historial el último mensaje del usuario como ancla.
                            let query_concept = {
                                let q = data_clone.to_lowercase();
                                let is_self_ref = q.contains(" mí") || q.contains("sobre mi") || q.contains("me dijiste") || q.contains("te dije") || q.contains("acabo de");
                                if is_self_ref {
                                    // Tomar el último mensaje de usuario del historial como concepto
                                    let snap = Self::snapshot_buffer(&buffer_clone);
                                    snap.iter().rev()
                                        .filter(|(r, _)| r == "Usuario")
                                        .nth(1) // El segundo más reciente (el actual ya está en el buffer)
                                        .map(|(_, c)| c.clone())
                                        .unwrap_or_else(|| data_clone.replace('¿', "").replace('?', "").trim().to_string())
                                } else {
                                    data_clone.replace('¿', "").replace('?', "").trim().to_string()
                                }
                            };

                            let _ = tx_memory_clone.send(SystemMessage::QueryContext {
                                concept: query_concept,
                                reply_to: tx_resp,
                            }).await;

                            if let Ok((context_str, pleasure)) = rx_resp.await {
                                let session_history = Self::format_session_history(
                                    &Self::snapshot_buffer(&buffer_clone)
                                );
                                let data_gen = data_clone.clone();

                                let response_opt = tokio::task::spawn_blocking(move || {
                                    Self::generate_response(
                                        &backend_1, &model_1,
                                        &data_gen, &context_str,
                                        &session_history, pleasure,
                                    )
                                }).await.unwrap();

                                if let Ok(response) = response_opt {
                                    Self::push_to_buffer(&buffer_clone, "Androide", &response);
                                    let _ = tx_perception_clone.send(SystemMessage::Output(response)).await;
                                } else if let Err(e) = response_opt {
                                    eprintln!("[COGNITION] - Error en pipeline CoT: {}", e);
                                }
                            }
                        }
                        // _permit se libera aquí automáticamente
                    });
                }
                SystemMessage::AutonomousThought { concept, context, pleasure } => {
                    let backend_a  = self.backend.clone();
                    let model_a    = self.model.clone();
                    let sem_a      = self.inference_semaphore.clone();
                    let tx_perc_a  = self.tx_perception.clone();
                    let buf_a      = self.session_buffer.clone();

                    tokio::spawn(async move {
                        let _permit = sem_a.acquire_owned().await.unwrap();
                        let thought_opt = tokio::task::spawn_blocking(move || {
                            Self::generate_inner_monologue(
                                &backend_a, &model_a,
                                &concept, &context, pleasure,
                            )
                        }).await.unwrap();

                        if let Ok(thought) = thought_opt {
                            println!("[ANDROIDE] - (monólogo interno) {}", thought);
                            Self::push_to_buffer(&buf_a, "Androide", &thought);
                            let _ = tx_perc_a.send(SystemMessage::Output(thought)).await;
                        }
                    });
                }
                SystemMessage::Shutdown => {
                    println!("[COGNITION] - Apagando actor y liberando modelo...");
                    break;
                }
                _ => {}
            }
        }
    }
}

