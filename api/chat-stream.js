/**
 * Vercel Serverless Function - Chat Stream
 * Streaming de respuestas para experiencia de texto apareciendo letra por letra
 * Usa Server-Sent Events (SSE) para streaming en tiempo real
 */

export default async function handler(req, res) {
  // Configurar CORS y headers de streaming
  const allowedOrigin = process.env.NODE_ENV === 'production'
    ? 'https://www.capibara6.com'
    : 'http://localhost:3000';

  res.setHeader('Access-Control-Allow-Origin', allowedOrigin);
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Configurar headers para Server-Sent Events
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Transfer-Encoding', 'chunked');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Método no permitido' });
  }

  try {
    const { message, model, temperature, max_tokens, use_semantic_router } = req.body;

    // Preparar payload para conectar con VM services gateway (formato esperado por gateway server)
    const payload = {
      message: message || '', // Campo requerido por el gateway server (no messages)
      model: model || 'aya_expanse_multilingual',
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 200,
      // El streaming se manejará internamente en el gateway server
      use_semantic_router: use_semantic_router || false
    };

    // Conectar al gateway server en VM services (IP externa)
    const gatewayResponse = await fetch('http://34.175.255.139:8080/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(60000) // Aumentar timeout para modelos grandes
    });

    if (!gatewayResponse.ok) {
      // Enviar error como evento SSE
      res.write(`data: ${JSON.stringify({ type: 'error', error: 'Error en gateway' })}\n\n`);
      res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
      res.end();
      return;
    }

    // Intentar leer como stream si el gateway server devuelve stream
    const reader = gatewayResponse.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Iniciar el stream
    res.write(`data: ${JSON.stringify({ type: 'start' })}\n\n`);

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Procesar líneas completas
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || ''; // Mantener última línea incompleta

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6); // Remover 'data: '
            if (data === '[DONE]') {
              res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
              break;
            }

            try {
              const parsed = JSON.parse(data);
              
              // Enviar contenido como evento de streaming
              if (parsed.choices?.[0]?.delta?.content) {
                res.write(`data: ${JSON.stringify({ 
                  type: 'content', 
                  content: parsed.choices[0].delta.content 
                })}\n\n`);
                res.flush ? res.flush() : void 0; // Forzar envío si disponible
              } else if (parsed.choices?.[0]?.finish_reason) {
                res.write(`data: ${JSON.stringify({ 
                  type: 'done', 
                  finish_reason: parsed.choices[0].finish_reason 
                })}\n\n`);
              }
            } catch (e) {
              // Si no es JSON válido, enviar como evento de contenido simple
              if (data.trim()) {
                res.write(`data: ${JSON.stringify({ 
                  type: 'content', 
                  content: data 
                })}\n\n`);
                res.flush ? res.flush() : void 0;
              }
            }
          }
        }
      }
    } catch (streamError) {
      console.error('Error en streaming:', streamError);
      res.write(`data: ${JSON.stringify({ type: 'error', error: streamError.message })}\n\n`);
    } finally {
      reader.releaseLock();
    }

    // Finalizar el stream
    res.write(`data: ${JSON.stringify({ type: 'complete' })}\n\n`);
    res.end();

  } catch (error) {
    console.error('Error en proxy streaming:', error);

    // Enviar error como evento SSE
    res.write(`data: ${JSON.stringify({ type: 'error', error: error.message })}\n\n`);
    res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
    res.end();
  }
}

// Configurar para streaming
export const config = {
  api: {
    responseLimit: false,
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
};