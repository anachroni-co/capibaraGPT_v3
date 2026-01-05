/**
 * Vercel Serverless Function - TTS Voice Cloning Proxy
 * Proxy para clonar voces desde el servidor TTS
 */

export default async function handler(req, res) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const TTS_URL = process.env.KYUTAI_TTS_URL || 'http://34.175.255.139:5001/tts';
    const BASE_URL = TTS_URL.replace('/tts', '');
    const CLONE_URL = `${BASE_URL}/clone`;

    console.log(`🎭 Proxying clone request to: ${CLONE_URL}`);

    // Vercel no soporta multipart/form-data directamente en serverless
    // Necesitamos pasar la request tal cual
    
    // Para este caso específico, recomendamos conectarse directo al backend desde el frontend
    // en desarrollo (localhost), o implementar un edge function más complejo
    
    // Temporal: indicar que se debe usar conexión directa
    return res.status(501).json({
      status: 'error',
      error: 'Voice cloning debe hacerse directamente al servidor TTS',
      message: 'Use localhost:5002/clone en desarrollo, o configure un edge function',
      direct_url: CLONE_URL
    });

  } catch (error) {
    console.error('❌ TTS Clone Proxy error:', error.message);

    return res.status(503).json({
      status: 'error',
      error: error.message
    });
  }
}

