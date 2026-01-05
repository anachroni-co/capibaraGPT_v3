// Script de prueba para la API GPT-OSS-20B
// Uso: Abrir DevTools (F12) y ejecutar este script

const API_URL = 'http://34.175.215.109/api/chat';
const HEALTH_URL = 'http://34.175.215.109/health';

// Función para probar el health check
async function testHealth() {
    console.log('🔍 Probando health check...');
    try {
        const response = await fetch(HEALTH_URL, {
            method: 'GET',
            timeout: 5000
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Health Check OK:', data);
            return true;
        } else {
            console.error('❌ Health Check Error:', response.status, response.statusText);
            return false;
        }
    } catch (error) {
        console.error('❌ Error de conexión health:', error.message);
        return false;
    }
}

// Función para probar el chat
async function testChat(message = "Hola, ¿cómo estás?") {
    console.log('💬 Probando chat con mensaje:', message);
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                max_tokens: 100
            }),
            timeout: 300000 // 5 minutos
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Chat OK:', data);
            console.log('🤖 Respuesta:', data.response);
            console.log('📊 Modelo:', data.model);
            console.log('🎯 Tokens:', data.tokens);
            return data;
        } else {
            console.error('❌ Chat Error:', response.status, response.statusText);
            const errorText = await response.text();
            console.error('📄 Error details:', errorText);
            return null;
        }
    } catch (error) {
        console.error('❌ Error de conexión chat:', error.message);
        return null;
    }
}

// Función para ejecutar todas las pruebas
async function runAllTests() {
    console.log('🚀 Iniciando pruebas de la API GPT-OSS-20B...');
    console.log('📍 URL:', API_URL);
    
    // Test 1: Health Check
    const healthOk = await testHealth();
    
    if (healthOk) {
        // Test 2: Chat básico
        await testChat();
        
        // Test 3: Chat con pregunta específica
        await testChat("Explica qué es la inteligencia artificial en una frase");
        
        console.log('✅ Todas las pruebas completadas');
    } else {
        console.error('❌ Health check falló, no se ejecutaron las pruebas de chat');
    }
}

// Exportar funciones para uso manual
window.testHealth = testHealth;
window.testChat = testChat;
window.runAllTests = runAllTests;

console.log('📝 Funciones disponibles:');
console.log('  - testHealth()');
console.log('  - testChat("mensaje")');
console.log('  - runAllTests()');
console.log('💡 Ejecuta runAllTests() para probar todo');
