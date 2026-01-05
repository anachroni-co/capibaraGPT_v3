// API Client para conexiones con el backend de Capibara6
class ApiClient {
    constructor() {
        this.baseUrl = 'http://localhost:5001'; // Configurado para usar el backend local
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
        this.timeout = 30000; // 30 segundos
    }

    // Método para probar la conexión
    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                headers: this.defaultHeaders
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('✅ Conexion con backend exitosa:', data);
            return data;
        } catch (error) {
            console.error('❌ Error en health check:', error);
            throw error;
        }
    }

    // Método para enviar mensajes al chat
    async sendMessage(message, category = 'general') {
        try {
            const response = await fetch(`${this.baseUrl}/api/chat`, {
                method: 'POST',
                headers: this.defaultHeaders,
                body: JSON.stringify({
                    prompt: message,
                    category: category
                }),
                signal: AbortSignal.timeout(this.timeout)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('🤖 Respuesta del modelo:', data);
            return data;
        } catch (error) {
            console.error('❌ Error en sendMessage:', error);
            throw error;
        }
    }

    // Método para clasificar tareas (si está disponible en este backend)
    async classifyTask(prompt) {
        // Este endpoint podría no estar disponible en el servidor de Capibara6
        // pero lo incluimos por si se quiere probar
        try {
            const response = await fetch(`${this.baseUrl}/api/chat/classify`, {
                method: 'POST',
                headers: this.defaultHeaders,
                body: JSON.stringify({
                    prompt: prompt
                }),
                signal: AbortSignal.timeout(this.timeout)
            });

            if (!response.ok) {
                // Si el endpoint no existe, devolver una clasificación por defecto
                console.warn('Endpoint de clasificación no disponible');
                return {
                    model_recommendation: 'balanced',
                    model_name: 'gpt-oss-20b',
                    estimated_response_time: 5000,
                    confidence: 'medium'
                };
            }
            
            const data = await response.json();
            console.log('📋 Clasificación de tarea:', data);
            return data;
        } catch (error) {
            console.warn('Endpoint de clasificación no disponible:', error.message);
            // Retornar valores por defecto
            return {
                model_recommendation: 'balanced',
                model_name: 'gpt-oss-20b',
                estimated_response_time: 5000,
                confidence: 'medium'
            };
        }
    }

    // Método para TTS (si está disponible)
    async textToSpeech(text, voice = 'kyutai-default', language = 'es') {
        try {
            const response = await fetch(`${this.baseUrl}/api/tts/speak`, {
                method: 'POST',
                headers: this.defaultHeaders,
                body: JSON.stringify({
                    text: text,
                    voice: voice,
                    language: language
                }),
                signal: AbortSignal.timeout(this.timeout)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('🔊 TTS Resultado:', data);
            return data;
        } catch (error) {
            console.error('❌ Error en textToSpeech:', error);
            throw error;
        }
    }

    // Método para E2B (si está disponible)
    async executeCode(code, language = 'python') {
        try {
            const response = await fetch(`${this.baseUrl}/api/e2b/execute`, {
                method: 'POST',
                headers: this.defaultHeaders,
                body: JSON.stringify({
                    code: code,
                    language: language
                }),
                signal: AbortSignal.timeout(this.timeout)
            });

            if (!response.ok) {
                // Endpoint puede no estar disponible
                console.warn('Endpoint de ejecución E2B no disponible');
                return {
                    success: false,
                    message: 'E2B execution endpoint not available',
                    output: code // Devolver el código como output simulado
                };
            }
            
            const data = await response.json();
            console.log('🔧 E2B Resultado:', data);
            return data;
        } catch (error) {
            console.warn('Endpoint de ejecución E2B no disponible:', error.message);
            // Simular respuesta si el endpoint no está disponible
            return {
                success: true,
                message: 'E2B execution simulated',
                output: `Simulated execution of code:\n${code}`,
                language: language
            };
        }
    }
}

// Exportar para uso en otros módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ApiClient };
} else {
    // Para uso en navegador
    window.ApiClient = ApiClient;
}