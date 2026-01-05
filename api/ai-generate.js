/**
 * Vercel Serverless Function - AI Generate
 * Endpoint principal para generación de respuestas de IA
 *
 * Este es el endpoint que el frontend llama para chat completions
 * Usa vLLM como principal y Ollama como fallback
 *
 * Actualizado: 2025-12-01
 */

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const {
      message,
      prompt,
      text,
      model,
      temperature = 0.7,
      max_tokens = 500,
      stream = false
    } = req.body;

    const userMessage = message || prompt || text || '';

    if (!userMessage) {
      return res.status(400).json({
        error: 'Message, prompt or text is required'
      });
    }

    // URLs de servicios (usando gateway server en VM services - IP externa)
    const CHAT_URL = process.env.CHAT_URL || 'http://34.175.255.139:8080/api/chat';

    console.log(`📨 Mensaje recibido: ${userMessage.substring(0, 50)}...`);
    console.log(`🎯 Modelo solicitado: ${model || 'auto'}`);

    // Llamada al gateway server
    try {
      console.log('📡 Enviando solicitud al gateway server...');

      const chatResponse = await fetch(CHAT_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          model: model || 'aya_expanse_multilingual',
          temperature: temperature,
          max_tokens: max_tokens,
          use_semantic_router: true
        }),
        signal: AbortSignal.timeout(60000) // Aumentar timeout para modelo grande
      });

      const data = await chatResponse.json();

      console.log('✅ Gateway server respondió');

      // Verificar si el gateway ya devolvió una respuesta simulada
      if (data.status && data.status.includes('simulated')) {
        console.log('⚠️ Gateway devolvió respuesta simulada, retornando tal cual');
        return res.status(200).json(data);
      }

      // Formatear la respuesta para que sea compatible con el frontend
      if (data.choices && data.choices[0]) {
        return res.status(200).json({
          response: data.choices[0].message?.content || data.response || data.content,
          content: data.choices[0].message?.content || data.response || data.content,
          model: data.model || model || 'aya_expanse_multilingual',
          provider: 'Capibara6 Gateway',
          tokens: data.tokens,
          finish_reason: data.choices[0].finish_reason,
          routing_info: data.routing_info
        });
      } else {
        // Si no tiene el formato de choices, usar directamente la respuesta
        return res.status(200).json({
          response: data.response || data.content || (data.choices && data.choices[0]?.message?.content),
          content: data.response || data.content || (data.choices && data.choices[0]?.message?.content),
          model: data.model || model || 'aya_expanse_multilingual',
          provider: 'Capibara6 Gateway',
          tokens: data.tokens,
          routing_info: data.routing_info
        });
      }
    } catch (error) {
      console.error('❌ Error al conectar con gateway server:', error);
      throw new Error('Gateway server no disponible');
    }

  } catch (error) {
    console.error('❌ Error en AI Generate:', error);

    // En lugar de devolver fallback aquí, dejar que el gateway server lo maneje
    return res.status(503).json({
      error: 'Servicios de IA temporalmente no disponibles',
      message: 'Por favor, intenta de nuevo en unos momentos.',
      details: error.message,
      provider: 'Capibara6 Gateway (error)'
    });
  }
}
