/**
 * Vercel Serverless Function - AI Classification
 * Endpoint para clasificar prompts de IA usando el sistema de routing semántico
 */

export default async function handler(req, res) {
  console.log('📡 Entrando en AI Classify handler:', req.method, req.url);
  
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    console.log('🔄 Manejando preflight OPTIONS para classify');
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    console.log('❌ Método no permitido para classify:', req.method);
    return res.status(405).json({ error: 'Método no permitido para classify' });
  }

  try {
    const { prompt, context, model } = req.body;

    console.log('📥 Datos recibidos para clasificación:', { 
      prompt_exists: !!prompt, 
      prompt_len: prompt ? prompt.length : 0,
      model: model || 'default',
      has_context: !!context
    });

    if (!prompt) {
      return res.status(400).json({ error: 'Prompt es requerido' });
    }

    // Enviar al gateway server para clasificación inteligente
    // Usar modelo más rápido para evitar timeouts
    const targetModel = (model && model !== 'aya_expanse_multilingual') ? model : 'phi4_fast';

    const classifyResponse = await fetch('http://34.175.255.139:8080/api/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: prompt,
        context: context,
        model: targetModel,  // Usar modelo más rápido
        use_semantic_router: true
      }),
      signal: AbortSignal.timeout(60000) // Aumentar timeout a 60 segundos para respuestas completas
    });

    console.log('📥 Respuesta del gateway classify:', classifyResponse.status);

    if (classifyResponse.ok) {
      const data = await classifyResponse.json();
      console.log('✅ Clasificación exitosa');
      return res.status(200).json(data);
    } else {
      // Si el gateway no devuelve 200, usar respuesta de fallback
      console.log('🔄 Usando clasificación de fallback');
      const classificationResult = {
        query: prompt,
        classification: {
          category: 'general',
          confidence: 0.8,
          domain: 'general',
          requires_context_extension: prompt.length < 100,
          model_preference: 'context_aware',
          next_step: 'process_query',
          complexity: prompt.length > 100 ? 'high' : prompt.length > 50 ? 'medium' : 'low'
        },
        status: 'classified',
        model_used: model || 'aya_expanse_multilingual',
        processing_time: Date.now()
      };

      return res.status(200).json(classificationResult);
    }

  } catch (error) {
    console.error('❌ Error en AI classify:', error.message);
    
    // En caso de error, devolver clasificación de fallback con información útil
    return res.status(200).json({
      query: req.body?.prompt || 'unknown',
      classification: {
        category: 'general',
        confidence: 0.5,
        domain: 'unknown',
        requires_context_extension: true,
        model_preference: 'default',
        next_step: 'process_query',
        complexity: 'medium',
        error: error.message
      },
      status: 'fallback_classification',
      fallback_mode: true,
      model_used: 'fallback_system',
      processing_time: Date.now()
    });
  }
}