// Ejemplo de endpoint Express.js
const express = require('express');
const app = express();

// Cargar configuración
const modelConfig = require('./model_config.json');
const { VLLMClient } = require('./ollama_client.js');  // Updated import
const { TaskClassifier } = require('./task_classifier.js');

const vllmClient = new VLLMClient(modelConfig);  // Updated client

app.use(express.json());

// Endpoint para generar texto con selección automática de modelo
app.post('/api/ai/generate', async (req, res) => {
  try {
    const { prompt, modelPreference = 'auto', streaming = false, ...options } = req.body;
    
    if (streaming) {
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      const modelTier = modelPreference === 'auto' ? TaskClassifier.classifyTask(prompt) : modelPreference;
      
      await vllmClient.streamWithModel(
        prompt,
        modelTier,
        (chunk) => res.write(chunk),
        () => res.end()
      );
    } else {
      const result = await vllmClient.generateWithFallback(prompt, {
        modelTier: modelPreference,
        ...options
      });
      
      if (result.success) {
        res.json({
          success: true,
          response: result.response,
          model_used: result.model,
          processing_time: result.total_duration,
          token_count: result.token_count
        });
      } else {
        res.status(500).json({
          success: false,
          error: result.error,
          message: 'Error al procesar la solicitud'
        });
      }
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Endpoint para clasificar tarea sin ejecutarla
app.post('/api/ai/classify', (req, res) => {
  const { prompt } = req.body;
  const modelTier = TaskClassifier.classifyTask(prompt);
  const estimatedTime = TaskClassifier.estimateResponseTime(modelTier);
  
  res.json({
    model_recommendation: modelTier,
    model_name: modelConfig.models[modelTier].name,
    estimated_response_time: estimatedTime,
    confidence: 'high'
  });
});

// Endpoint para uso específico de modelo
app.post('/api/ai/:modelTier/generate', async (req, res) => {
  try {
    const { modelTier } = req.params;
    const { prompt, ...options } = req.body;
    
    const result = await vllmClient.generate(prompt, modelTier, options);
    
    if (result.success) {
      res.json({
        success: true,
        response: result.response,
        model_used: result.model,
        processing_time: result.total_duration
      });
    } else {
      res.status(500).json({
        success: false,
        error: result.error
      });
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});