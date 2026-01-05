#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enviar email de prueba rápido
"""

import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Cargar variables de entorno
load_dotenv()

def send_test():
    print("Enviando email de prueba desde capibara6...\n")
    
    SMTP_SERVER = os.getenv('SMTP_SERVER')
    SMTP_PORT = int(os.getenv('SMTP_PORT'))
    SMTP_USER = os.getenv('SMTP_USER')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    FROM_EMAIL = os.getenv('FROM_EMAIL')
    
    print(f"Servidor: {SMTP_SERVER}:{SMTP_PORT}")
    print(f"Usuario: {SMTP_USER}")
    print(f"Destino: electrohipy@gmail.com\n")
    
    try:
        print("Conectando al servidor SMTP...")
        msg = MIMEMultipart('alternative')
        msg['Subject'] = '🦫 Email de prueba - capibara6 Backend'
        msg['From'] = FROM_EMAIL
        msg['To'] = 'electrohipy@gmail.com'
        
        text = """
¡Hola desde capibara6!

Este es un email de prueba del backend de capibara6 configurado para ES2030.

✅ Configuración SMTP exitosa
✅ Servidor: smtp.dondominio.com
✅ Email: es2030@capibaragpt.com

El sistema está listo para enviar emails automáticamente cuando los usuarios dejen su correo en el chatbot.

Saludos,
Backend capibara6 🦫
        """
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            background: #f4f4f4; 
            padding: 20px; 
        }
        .container { 
            background: white; 
            padding: 40px; 
            border-radius: 10px; 
            max-width: 600px; 
            margin: 0 auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #667eea; 
            margin: 0 0 20px 0;
        }
        .success { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0; 
        }
        .info {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🦫 capibara6</h1>
        <h2>Email de Prueba - ES2030</h2>
        <p>¡Hola!</p>
        <p>Este es un email de prueba del backend de capibara6 configurado para el evento ES2030.</p>
        
        <div class="success">
            <h3 style="margin: 0 0 10px 0;">✅ ¡Configuración Exitosa!</h3>
            <p style="margin: 0;">El sistema está funcionando correctamente y listo para enviar emails.</p>
        </div>
        
        <div class="info">
            <strong>📋 Configuración actual:</strong><br>
            ✅ Servidor: smtp.dondominio.com<br>
            ✅ Email: es2030@capibaragpt.com<br>
            ✅ Puerto: 587 (SSL/TLS)
        </div>
        
        <h3>🎯 ¿Qué significa esto?</h3>
        <p>El backend está completamente configurado y listo. Cuando un usuario deje su email en el chatbot:</p>
        <ul>
            <li>📧 Recibirá un email de confirmación automáticamente</li>
            <li>📊 Los datos se guardarán en el servidor</li>
            <li>🔔 Recibirás una notificación con la información</li>
        </ul>
        
        <div class="footer">
            <p><strong>Backend capibara6 🦫</strong><br>
            Sistema de IA Conversacional Avanzado</p>
            <p style="font-size: 12px; color: #999;">
                Este es un email de prueba automático del sistema.
            </p>
        </div>
    </div>
</body>
</html>
        """
        
        part1 = MIMEText(text, 'plain', 'utf-8')
        part2 = MIMEText(html, 'html', 'utf-8')
        msg.attach(part1)
        msg.attach(part2)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.starttls()
            print("Iniciando TLS...")
            server.login(SMTP_USER, SMTP_PASSWORD)
            print("Autenticacion exitosa...")
            server.send_message(msg)
        
        print("\nEmail enviado exitosamente!")
        print("Revisa la bandeja de entrada de electrohipy@gmail.com")
        print("(Tambien revisa la carpeta de spam por si acaso)\n")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"\nERROR: Autenticacion fallida")
        print(f"   {e}")
        print("\nVerifica que las credenciales en .env sean correctas")
        return False
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nVerifica la configuracion SMTP en .env")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("CAPIBARA6 - TEST DE EMAIL")
    print("=" * 60)
    print()
    send_test()
    print("=" * 60)

