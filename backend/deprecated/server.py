#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend de capibara6 - Servidor Flask para gestión de emails
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from dotenv import load_dotenv
import json

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir peticiones desde el frontend

# Configuración SMTP
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', 'info@anachroni.co')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
FROM_EMAIL = os.getenv('FROM_EMAIL', 'info@anachroni.co')

# Archivo para guardar datos
DATA_FILE = 'user_data/conversations.json'

def ensure_data_dir():
    """Crear directorio de datos si no existe"""
    os.makedirs('user_data', exist_ok=True)

def save_to_file(data):
    """Guardar datos en archivo JSON"""
    ensure_data_dir()
    
    # Leer datos existentes
    existing_data = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except:
            existing_data = []
    
    # Agregar nuevos datos
    existing_data.append({
        'timestamp': datetime.now().isoformat(),
        'email': data.get('email'),
        'conversations': data.get('conversations', []),
        'user_agent': request.headers.get('User-Agent'),
        'ip': request.remote_addr
    })
    
    # Guardar
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    # También guardar en txt
    txt_file = f'user_data/user_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write('=== CAPIBARA6 - DATOS DE USUARIO ===\n\n')
        f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Email: {data.get("email")}\n')
        f.write(f'IP: {request.remote_addr}\n\n')
        f.write('--- CONVERSACIONES ---\n\n')
        for conv in data.get('conversations', []):
            f.write(f'[{conv.get("timestamp")}]\n')
            f.write(f'{conv.get("message")}\n\n')

def send_email(to_email, conversations):
    """Enviar email de confirmación al usuario"""
    try:
        # Crear mensaje
        msg = MIMEMultipart('alternative')
        msg['Subject'] = '¡Gracias por tu interés en capibara6! 🦫'
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        
        # Contenido del email
        text_content = f"""
¡Hola!

Gracias por tu interés en capibara6, nuestro sistema de IA conversacional avanzado.

Hemos recibido tu mensaje y nos pondremos en contacto contigo muy pronto.

Mientras tanto, puedes:
- Visitar nuestro repositorio: https://github.com/anachroni-co/capibara6
- Explorar la documentación en nuestra web
- Seguirnos en nuestras redes sociales

Un saludo,
Equipo Anachroni
https://www.anachroni.co

---
Este es un email automático. Si necesitas ayuda inmediata, responde a este correo.
        """
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 28px; }}
        .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
        .button {{ display: inline-block; padding: 12px 30px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 14px; }}
        .links {{ margin: 20px 0; }}
        .links a {{ color: #667eea; text-decoration: none; margin: 0 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦫 capibara6</h1>
            <p>Sistema de IA Conversacional Avanzado</p>
        </div>
        <div class="content">
            <h2>¡Hola!</h2>
            <p>Gracias por tu interés en <strong>capibara6</strong>, nuestro sistema de IA conversacional de última generación.</p>
            <p>Hemos recibido tu mensaje y nos pondremos en contacto contigo muy pronto para darte más información.</p>
            
            <h3>Mientras tanto, puedes:</h3>
            <ul>
                <li>🔗 <a href="https://github.com/anachroni-co/capibara6">Explorar nuestro repositorio en GitHub</a></li>
                <li>📚 Revisar nuestra documentación técnica</li>
                <li>🚀 Probar nuestras demos interactivas</li>
            </ul>
            
            <div style="text-align: center;">
                <a href="https://github.com/anachroni-co/capibara6" class="button">Ver en GitHub</a>
            </div>
            
            <div class="footer">
                <p><strong>Equipo Anachroni</strong><br>
                <a href="https://www.anachroni.co">www.anachroni.co</a></p>
                <p style="font-size: 12px; color: #999;">
                    Este es un email automático. Si necesitas ayuda inmediata, responde a este correo.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        # Adjuntar contenido
        part1 = MIMEText(text_content, 'plain', 'utf-8')
        part2 = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(part1)
        msg.attach(part2)
        
        # Enviar email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f'Error enviando email: {e}')
        return False

def send_notification_to_admin(user_email, conversations):
    """Enviar notificación al admin con los datos del usuario"""
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'Nuevo contacto desde capibara6: {user_email}'
        msg['From'] = FROM_EMAIL
        msg['To'] = FROM_EMAIL
        
        # Preparar conversaciones
        conv_text = '\n'.join([f"[{c.get('timestamp')}] {c.get('message')}" for c in conversations])
        
        text_content = f"""
NUEVO CONTACTO DESDE CAPIBARA6

Email: {user_email}
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

CONVERSACIONES:
{conv_text}

---
IP: {request.remote_addr}
User Agent: {request.headers.get('User-Agent')}
        """
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: monospace; background: #1a1a1a; color: #00ff00; padding: 20px; }}
        .container {{ background: #0a0a0a; padding: 20px; border: 2px solid #00ff00; border-radius: 5px; }}
        .email {{ color: #00ffff; font-size: 18px; font-weight: bold; }}
        .conversation {{ background: #151515; padding: 15px; margin: 10px 0; border-left: 3px solid #667eea; }}
        .timestamp {{ color: #888; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🦫 NUEVO CONTACTO DESDE CAPIBARA6</h2>
        <p><strong>Email:</strong> <span class="email">{user_email}</span></p>
        <p><strong>Fecha:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h3>CONVERSACIONES:</h3>
        {''.join([f'<div class="conversation"><div class="timestamp">{c.get("timestamp")}</div><div>{c.get("message")}</div></div>' for c in conversations])}
        
        <hr>
        <p style="color: #666; font-size: 12px;">
            IP: {request.remote_addr}<br>
            User Agent: {request.headers.get('User-Agent')}
        </p>
    </div>
</body>
</html>
        """
        
        part1 = MIMEText(text_content, 'plain', 'utf-8')
        part2 = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(part1)
        msg.attach(part2)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f'Error enviando notificación al admin: {e}')
        return False

@app.route('/api/save-conversation', methods=['POST'])
def save_conversation():
    """Endpoint para guardar conversación y enviar email"""
    try:
        data = request.get_json()
        
        email = data.get('email')
        conversations = data.get('conversations', [])
        
        if not email:
            return jsonify({'success': False, 'error': 'Email requerido'}), 400
        
        # Guardar en archivo
        save_to_file(data)
        
        # Enviar email al usuario
        email_sent = send_email(email, conversations)
        
        # Enviar notificación al admin
        admin_notified = send_notification_to_admin(email, conversations)
        
        return jsonify({
            'success': True,
            'email_sent': email_sent,
            'admin_notified': admin_notified,
            'message': 'Datos guardados correctamente'
        })
    
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/', methods=['GET'])
def index():
    """Página principal"""
    return '''
    <html>
        <head>
            <title>capibara6 Backend</title>
            <style>
                body { font-family: monospace; background: #0a0a0a; color: #00ff00; padding: 40px; }
                h1 { color: #00ffff; }
                .status { color: #00ff00; }
            </style>
        </head>
        <body>
            <h1>capibara6 Backend</h1>
            <p class="status">Servidor funcionando correctamente</p>
            <p>Endpoints disponibles:</p>
            <ul>
                <li>POST /api/save-conversation - Guardar conversacion y enviar email</li>
                <li>GET /api/health - Health check</li>
            </ul>
        </body>
    </html>
    '''

if __name__ == '__main__':
    ensure_data_dir()
    print('capibara6 Backend iniciado')
    print(f'Email configurado: {FROM_EMAIL}')
    
    # Puerto para Railway (usa variable de entorno PORT)
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

