//! LLM Service for AI-powered decision making
//!
//! Provides a unified interface to OpenAI's API for agent intelligence.

use crate::config::OpenAIConfig;
use async_openai::{
    config::OpenAIConfig as AsyncOpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
        CreateEmbeddingRequestArgs,
    },
    Client,
};
use tracing::{debug, error, info};

/// LLM service for agent intelligence
pub struct LLMService {
    client: Client<AsyncOpenAIConfig>,
    model: String,
    embedding_model: String,
    max_tokens: u32,
    temperature: f32,
}

impl LLMService {
    /// Create a new LLM service from configuration
    pub fn new(config: &OpenAIConfig) -> anyhow::Result<Self> {
        let openai_config = AsyncOpenAIConfig::new().with_api_key(&config.api_key);
        let client = Client::with_config(openai_config);

        Ok(Self {
            client,
            model: config.model.clone(),
            embedding_model: config.embedding_model.clone(),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
        })
    }

    /// Create with explicit parameters
    pub fn with_params(api_key: &str, model: &str) -> anyhow::Result<Self> {
        let openai_config = AsyncOpenAIConfig::new().with_api_key(api_key);
        let client = Client::with_config(openai_config);

        Ok(Self {
            client,
            model: model.to_string(),
            embedding_model: "text-embedding-3-small".to_string(),
            max_tokens: 2048,
            temperature: 0.7,
        })
    }

    /// Generate a chat completion with timeout
    pub async fn chat_completion(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> anyhow::Result<String> {
        debug!("Chat completion request - model: {}", self.model);

        let messages = vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt)
                    .build()?
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user_message)
                    .build()?
            ),
        ];

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(messages)
            .max_tokens(self.max_tokens as u16)
            .temperature(self.temperature)
            .build()?;

        // Add timeout to prevent hanging on slow LLM responses
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            self.client.chat().create(request)
        )
        .await
        .map_err(|_| anyhow::anyhow!("LLM chat completion timed out after 60 seconds"))?
        .map_err(|e| anyhow::anyhow!("LLM API error: {}", e))?;

        let content = response.choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!(
                "No response content from LLM. Choices count: {}, finish reason: {:?}",
                response.choices.len(),
                response.choices.first().map(|c| &c.finish_reason)
            ))?;

        debug!("Chat completion response length: {} chars", content.len());

        Ok(content)
    }

    /// Generate a chat completion with conversation history
    pub async fn chat_completion_with_history(
        &self,
        system_prompt: &str,
        history: Vec<(String, String)>, // (role, content) pairs
        user_message: &str,
    ) -> anyhow::Result<String> {
        let mut messages = vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt)
                    .build()?
            ),
        ];

        // Add history
        for (role, content) in history {
            let msg = match role.as_str() {
                "user" => ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(content)
                        .build()?
                ),
                "assistant" => ChatCompletionRequestMessage::Assistant(
                    async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                        .content(content)
                        .build()?
                ),
                _ => continue,
            };
            messages.push(msg);
        }

        // Add current message
        messages.push(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_message)
                .build()?
        ));

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(messages)
            .max_tokens(self.max_tokens as u16)
            .temperature(self.temperature)
            .build()?;

        // Add timeout to prevent hanging on slow LLM responses
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(120), // Longer timeout for history-based calls
            self.client.chat().create(request)
        )
        .await
        .map_err(|_| anyhow::anyhow!("LLM chat completion with history timed out after 120 seconds"))?
        .map_err(|e| anyhow::anyhow!("LLM API error: {}", e))?;

        let content = response.choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("No response content from LLM"))?;

        Ok(content)
    }

    /// Generate embeddings for text
    pub async fn get_embedding(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.embedding_model)
            .input(text)
            .build()?;

        let response = self.client.embeddings().create(request).await?;

        let embedding = response.data
            .first()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?
            .embedding
            .clone();

        Ok(embedding)
    }

    /// Generate embeddings for multiple texts
    pub async fn get_embeddings(&self, texts: Vec<&str>) -> anyhow::Result<Vec<Vec<f32>>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.embedding_model)
            .input(texts)
            .build()?;

        let response = self.client.embeddings().create(request).await?;

        let embeddings: Vec<Vec<f32>> = response.data
            .into_iter()
            .map(|e| e.embedding)
            .collect();

        Ok(embeddings)
    }

    /// Analyze text and return structured JSON
    pub async fn analyze_json<T: serde::de::DeserializeOwned>(
        &self,
        prompt: &str,
        data: &str,
    ) -> anyhow::Result<T> {
        let system = format!(
            "{}\n\nRespond with valid JSON only. Do not include markdown code blocks.",
            prompt
        );

        let response = self.chat_completion(&system, data).await?;

        // Try to extract JSON if wrapped in code blocks
        // Ensure bounds are valid: start <= end to prevent slice panic
        let json_str = if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                if start <= end {
                    &response[start..=end]
                } else {
                    // Malformed JSON: opening brace appears after closing brace
                    &response
                }
            } else {
                &response
            }
        } else {
            &response
        };

        serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse LLM JSON response: {} - Response: {}", e, response))
    }

    /// Get the model being used
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the embedding model being used
    pub fn embedding_model(&self) -> &str {
        &self.embedding_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_creation() {
        // This test would require a valid API key
        // Just testing the struct creation logic
        let config = OpenAIConfig {
            api_key: "test-key".to_string(),
            model: "gpt-4o-mini".to_string(),
            embedding_model: "text-embedding-3-small".to_string(),
            max_tokens: 2048,
            temperature: 0.7,
        };

        let service = LLMService::new(&config).unwrap();
        assert_eq!(service.model(), "gpt-4o-mini");
    }
}
