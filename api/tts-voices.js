/**
 * Vercel Serverless Function - TTS Voices Proxy
 * Proxy para listar voces disponibles en el servidor TTS
 */

export default async function handler(req, res) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const TTS_URL = process.env.KYUTAI_TTS_URL || 'http://34.175.255.139:5001/tts';
    const BASE_URL = TTS_URL.replace('/tts', '');
    const VOICES_URL = `${BASE_URL}/voices`;

    console.log(`🔍 Fetching voices from: ${VOICES_URL}`);

    const response = await fetch(VOICES_URL, {
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      },
      signal: AbortSignal.timeout(5000)
    });

    if (!response.ok) {
      throw new Error(`TTS server responded with ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ Voices retrieved');

    return res.status(200).json(data);

  } catch (error) {
    console.error('❌ TTS Voices Proxy error:', error.message);

    return res.status(503).json({
      status: 'error',
      error: error.message,
      fallback: true
    });
  }
}

