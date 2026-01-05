/**
 * Vercel Serverless Function - Chat Completions con Streaming
 * Proxy HTTPS para vLLM Multi-Model Server con streaming y contexto
 */

export const config = {
  api: {
    responseLimit: false,
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
};

export default async function handler(req, res) {
  // CORS headers (se aplican siempre)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight (OPTIONS) immediately
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Solo permitir POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Configurar para streaming
  res.writeHead(200, {
    'Content-Type': 'text/plain; charset=utf-8',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Transfer-Encoding': 'chunked',
  });

  try {
    // URLs de servicios (usar gateway server en VM services - IP externa)
    const CHAT_URL = process.env.CHAT_URL || 'http://34.175.255.139:8080/api/chat';

    // Extraer parámetros
    const prompt = req.body.prompt || req.body.message || (req.body.messages && req.body.messages[req.body.messages.length - 1]?.content);
    const model = req.body.model || 'phi4_fast';
    const temperature = req.body.temperature || 0.7;
    const max_tokens = req.body.max_tokens || 200;
    
    // Preparar payload para el gateway server sin streaming
    const gatewayPayload = {
      message: prompt,
      model: model,
      temperature: temperature,
      max_tokens: max_tokens,
      use_semantic_router: true
    };

    // Hacer solicitud al gateway server
    const gatewayResponse = await fetch(CHAT_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(gatewayPayload),
      signal: AbortSignal.timeout(60000) // Aumentar timeout para modelo grande
    });

    const data = await gatewayResponse.json();

    // Verificar si el gateway ya devolvió una respuesta simulada
    if (data.status && data.status.includes('simulated')) {
      // Enviar la respuesta simulada directamente
      res.write(`data: ${JSON.stringify(data)}\n\n`);
      res.write(`data: ${JSON.stringify({ type: 'done', finish_reason: 'stop' })}\n\n`);
      res.end();
      return;
    }

    // Si hay respuesta con choices (formato OpenAI), usarla
    if (data.choices && data.choices[0]) {
      const content = data.choices[0].message?.content || data.response || data.content;

      // Enviar la respuesta completa como un solo chunk
      res.write(`data: ${JSON.stringify({ type: 'content', content: content })}\n\n`);
      res.write(`data: ${JSON.stringify({ type: 'done', finish_reason: data.choices[0].finish_reason || 'stop' })}\n\n`);
      res.end();
    } else {
      // Formato alternativo
      const content = data.response || data.content || (data.choices && data.choices[0]?.message?.content);

      // Enviar la respuesta completa
      res.write(`data: ${JSON.stringify({ type: 'content', content: content })}\n\n`);
      res.write(`data: ${JSON.stringify({ type: 'done', finish_reason: 'stop' })}\n\n`);
      res.end();
    }
  } catch (error) {
    console.error('❌ Error en proxy streaming:', error);
    res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
    res.end();
  }
}