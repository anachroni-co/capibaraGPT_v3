class TaskClassifier {
  static classifyTask(prompt) {
    const promptLower = prompt.toLowerCase();
    
    // Clasificación basada en palabras clave y longitud
    const complexityIndicators = {
      'complex': ['análisis', 'razonamiento', 'comparación', 'evaluar', 'estrategia', 'planificación', 'investigación', 'profundo', 'detalle', 'complejo', 'técnico'],
      'balanced': ['explicar', 'qué es', 'cómo funciona', 'describir', 'resumen', 'breve', 'ejemplo', 'definir'],
      'simple': ['qué', 'quién', 'cuál', 'cuándo', 'dónde', 'chiste', 'broma', 'saludo', 'ayuda']
    };
    
    // Contar indicadores de cada tipo
    let complexScore = 0, balancedScore = 0, simpleScore = 0;
    
    complexityIndicators.complex.forEach(indicator => {
      if (promptLower.includes(indicator)) complexScore += 2;
    });
    
    complexityIndicators.balanced.forEach(indicator => {
      if (promptLower.includes(indicator)) balancedScore += 1;
    });
    
    complexityIndicators.simple.forEach(indicator => {
      if (promptLower.includes(indicator)) simpleScore += 1;
    });
    
    // Considerar la longitud del prompt
    if (prompt.length > 100) balancedScore += 1;
    if (prompt.length > 200) complexScore += 1;
    
    // Determinar el modelo más apropiado
    if (complexScore >= Math.max(balancedScore, simpleScore)) {
      return 'complex';
    } else if (balancedScore >= Math.max(complexScore, simpleScore)) {
      return 'balanced';
    } else {
      return 'fast_response';
    }
  }
  
  static estimateResponseTime(modelTier) {
    const responseTimes = {
      'fast_response': 2000,    // ~2 segundos
      'balanced': 4000,         // ~4 segundos
      'complex': 120000         // ~2 minutos
    };
    return responseTimes[modelTier];
  }
}