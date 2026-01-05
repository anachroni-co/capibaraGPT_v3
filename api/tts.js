/**
 * Vercel Serverless Function - TTS Proxy
 * Proxy ultra-ligero a servidor TTS en VM
 * JavaScript es más ligero que Python para proxies simples
 */

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle OPTIONS (preflight)
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  // Only accept POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { text, language = 'es' } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }

    // Limitar caracteres
    const truncatedText = text.length > 3000 ? text.substring(0, 3000) : text;

    // URL del servidor Kyutai TTS en VM services (variable de entorno)
    const TTS_URL = process.env.KYUTAI_TTS_URL || 'http://34.175.255.139:5001/tts';

    console.log(`📝 Proxy TTS: ${truncatedText.length} chars -> ${TTS_URL}`);

    // Reenviar request a la VM
    const response = await fetch(TTS_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: truncatedText,
        language: language
      }),
      signal: AbortSignal.timeout(30000) // 30 segundos timeout
    });

    if (!response.ok) {
      throw new Error(`VM responded with status ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ TTS exitoso desde VM');
    
    return res.status(200).json(data);

  } catch (error) {
    console.error('❌ Error en proxy TTS:', error.message);

    // Devolver fallback para que el frontend use Web Speech API
    return res.status(200).json({
      error: error.message,
      fallback: true,
      provider: 'Web Speech API (VM unavailable)'
    });
  }
}

