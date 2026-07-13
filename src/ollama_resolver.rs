use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

pub fn resolve_best_model(base_path: &str) -> Result<String, String> {
    let base = Path::new(base_path);
    let manifests_dir = base.join("manifests").join("registry.ollama.ai").join("library");

    if !manifests_dir.exists() {
        return Err(format!("El directorio de manifiestos no existe: {}", manifests_dir.display()));
    }

    let mut potential_models: Vec<(i32, PathBuf)> = Vec::new();

    // Explorar directorio local de manifiestos
    if let Ok(entries) = fs::read_dir(&manifests_dir) {
        for entry in entries.flatten() {
            let model_name = entry.file_name().into_string().unwrap_or_default();
            
            // Priorización por familias de modelos eficientes para extracción semántica
            let lower_name = model_name.to_lowercase();
            
            // Score-based priority: higher score means checked first
            let mut priority_score = 0;
            if lower_name.contains("llama3.1") {
                priority_score = 100; // Prefer Llama 3.1
            } else if lower_name.contains("llama") {
                priority_score = 80;
            } else if lower_name.contains("qwen") {
                priority_score = 50;
            } else if lower_name.contains("phi") || lower_name.contains("mistral") {
                priority_score = 30;
            }

            if let Ok(tags) = fs::read_dir(entry.path()) {
                for tag_entry in tags.flatten() {
                    if tag_entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                        potential_models.push((priority_score, tag_entry.path()));
                    }
                }
            }
        }
    }

    if potential_models.is_empty() {
        return Err("No se encontraron manifiestos de modelos locales.".to_string());
    }

    // Sort by priority score (descending)
    potential_models.sort_by(|a, b| b.0.cmp(&a.0));
    let sorted_models: Vec<PathBuf> = potential_models.into_iter().map(|(_, p)| p).collect();

    for manifest_path in sorted_models {
        if let Ok(content) = fs::read_to_string(&manifest_path) {
            if let Ok(json) = serde_json::from_str::<Value>(&content) {
                if let Some(layers) = json.get("layers").and_then(|l| l.as_array()) {
                    for layer in layers {
                        if let Some(media_type) = layer.get("mediaType").and_then(|m| m.as_str()) {
                            // Localizar la capa binaria del modelo
                            if media_type == "application/vnd.ollama.image.model" {
                                if let Some(digest) = layer.get("digest").and_then(|d| d.as_str()) {
                                    // Adaptar el hash digest a formato físico de archivo en Windows
                                    let blob_name = digest.replace("sha256:", "sha256-");
                                    // La estructura real de blobs de ollama: models/blobs/<sha256-...>
                                    let blob_path = base.join("blobs").join(blob_name);
                                    
                                    if blob_path.exists() {
                                        return Ok(blob_path.to_string_lossy().into_owned());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Err("Se encontraron manifiestos, pero ningún blob físico GGUF válido asociado en la ruta esperada.".to_string())
}
