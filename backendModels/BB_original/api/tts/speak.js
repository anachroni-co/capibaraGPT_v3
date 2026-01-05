// api/tts/speak.js - Proxy a VM TTS
export default async function handler(request, response) {
  if (request.method !== 'POST') {
    return response.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const backendResponse = await fetch('http://34.175.215.109:5004/api/tts/speak', {  // Puerto 5004 para TTS
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request.body),
    });

    const data = await backendResponse.json();
    response.status(backendResponse.status).json(data);
  } catch (error) {
    console.error('Error proxy TTS:', error);
    response.status(500).json({ error: 'Error interno del servidor' });
  }
}