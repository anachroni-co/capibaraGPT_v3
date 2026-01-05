const axios = require('axios');

class OllamaClient {
  constructor(config) {
    this.endpoint = config.api_settings.ollama_endpoint;
    this.models = config.models;
    this.fallbackEnabled = config.fallback_strategy.enabled;
    this.fallbackOrder = config.fallback_strategy.order;
  }
  
  async generateWithFallback(prompt, options = {}) {
    const preferredModelTier = options.modelTier || 'auto';
    let modelToUse;
    
    if (preferredModelTier === 'auto') {
      modelToUse = TaskClassifier.classifyTask(prompt);
    } else {
      modelToUse = preferredModelTier;
    }
    
    // Intentar con el modelo clasificado
    const result = await this.generate(prompt, modelToUse, options);
    
    // Si falla y está habilitado el fallback, intentar con el siguiente modelo
    if (!result.success && this.fallbackEnabled) {
      const currentIndex = this.fallbackOrder.indexOf(modelToUse);
      for (let i = currentIndex + 1; i < this.fallbackOrder.length; i++) {
        console.log(`Fallback al modelo: ${this.fallbackOrder[i]}`);
        try {
          return await this.generate(prompt, this.fallbackOrder[i], options);
        } catch (error) {
          console.log(`Fallback fallido para: ${this.fallbackOrder[i]}`);
          continue;
        }
      }
    }
    
    return result;
  }
  
  async generate(prompt, modelTier, options = {}) {
    const modelConfig = this.models[modelTier];
    if (!modelConfig) {
      throw new Error(`Modelo no encontrado: ${modelTier}`);
    }
    
    const requestData = {
      model: modelConfig.name,
      prompt: prompt,
      stream: options.streaming || false,
      options: {
        temperature: options.temperature || 0.7,
        num_predict: Math.min(options.max_tokens || modelConfig.max_tokens, 2048),
        top_p: options.top_p || 0.9,
        top_k: options.top_k || 40
      }
    };
    
    try {
      const timeout = options.timeout || modelConfig.timeout;
      
      const response = await axios.post(
        `${this.endpoint}/api/generate`,
        requestData,
        { 
          timeout: timeout,
          headers: { 'Content-Type': 'application/json' }
        }
      );
      
      return {
        success: true,
        model: modelConfig.name,
        response: response.data.response,
        eval_duration: response.data.eval_duration,
        total_duration: response.data.total_duration,
        token_count: response.data.eval_count
      };
    } catch (error) {
      console.error(`Error con modelo ${modelConfig.name}:`, error.message);
      return {
        success: false,
        error: error.message,
        model: modelConfig.name
      };
    }
  }
  
  async streamWithModel(prompt, modelTier, onData, onEnd) {
    const modelConfig = this.models[modelTier];
    if (!modelConfig) {
      throw new Error(`Modelo no encontrado: ${modelTier}`);
    }
    
    const response = await fetch(`${this.endpoint}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelConfig.name,
        prompt: prompt,
        stream: true,
        options: {
          temperature: 0.7,
          num_predict: modelConfig.max_tokens
        }
      })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep the incomplete line in buffer
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              if (data.response) {
                onData(data.response);
              }
              if (data.done) {
                onEnd();
                break;
              }
            } catch (e) {
              console.error('Error parsing stream data:', e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}