# Servicios systemd para capibara6-consensus

Este directorio contiene los archivos de servicio systemd para el despliegue de capibara6-consensus en producción.

## Servicios Disponibles

### n8n.service
Servicio de automatización de workflows n8n integrado con capibara6-consensus.

**Instalación:**
```bash
sudo cp n8n.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable n8n.service
sudo systemctl start n8n.service
```

**Verificar estado:**
```bash
sudo systemctl status n8n.service
```

**Logs:**
```bash
sudo journalctl -u n8n.service -f
```

## Requisitos Previos

### Para n8n
- Node.js >= 20.19
- npm >= 10.x
- Usuario: `elect` (o modificar el campo `User=` en el servicio)
- Directorio: `/home/elect/.n8n/` para datos y logs

### Instalación de n8n
```bash
# Instalar Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs -y

# Instalar n8n globalmente
sudo npm install -g n8n

# Crear directorios necesarios
mkdir -p /home/elect/.n8n/logs
```

## Configuración

### Variables de Entorno de n8n

El servicio n8n utiliza las siguientes variables de entorno:

- `N8N_HOST=0.0.0.0` - Host de escucha
- `N8N_PORT=5678` - Puerto de escucha
- `N8N_PROTOCOL=http` - Protocolo (cambiar a https en producción)
- `NODE_ENV=production` - Entorno de ejecución
- `WEBHOOK_URL=http://localhost:5678/` - URL base para webhooks
- `N8N_USER_FOLDER=/home/elect/.n8n` - Carpeta de datos de usuario
- `N8N_LOG_LEVEL=info` - Nivel de logging
- `N8N_LOG_OUTPUT=console,file` - Salida de logs
- `N8N_LOG_FILE_LOCATION=/home/elect/.n8n/logs/` - Ubicación de archivos de log

### Personalización

Para modificar la configuración, editar el archivo de servicio:

```bash
sudo systemctl edit n8n.service
```

O editar directamente:
```bash
sudo nano /etc/systemd/system/n8n.service
```

Después de modificar:
```bash
sudo systemctl daemon-reload
sudo systemctl restart n8n.service
```

## Integración con Nginx

Para acceder a n8n a través de nginx, añadir al archivo de configuración:

```nginx
location /n8n/ {
    proxy_pass http://localhost:5678/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    proxy_buffering off;
}
```

## Monitoreo

### Ver logs en tiempo real
```bash
sudo journalctl -u n8n.service -f
```

### Ver logs desde archivo
```bash
tail -f /home/elect/.n8n/logs/*.log
```

### Ver estado de todos los servicios
```bash
systemctl list-units --type=service --state=running | grep -E "(n8n|flask|llama|coqui)"
```

## Troubleshooting

### El servicio no inicia
1. Verificar que n8n esté instalado: `which n8n`
2. Verificar permisos del directorio: `ls -la /home/elect/.n8n/`
3. Verificar logs: `sudo journalctl -u n8n.service -n 50`

### Puerto en uso
Si el puerto 5678 está ocupado:
1. Modificar `N8N_PORT` en el servicio
2. Actualizar configuración de nginx
3. Reiniciar servicios

### Problemas de memoria
Si n8n consume mucha memoria, ajustar límites en el servicio:

```ini
[Service]
MemoryMax=512M
MemoryHigh=384M
```

## Backup

### Backup manual
```bash
tar -czf n8n-backup-$(date +%Y%m%d).tar.gz /home/elect/.n8n/
```

### Restaurar backup
```bash
sudo systemctl stop n8n.service
tar -xzf n8n-backup-YYYYMMDD.tar.gz -C /
sudo systemctl start n8n.service
```

## Seguridad

### Recomendaciones
1. Usar HTTPS en producción (configurar certificados SSL)
2. Configurar autenticación robusta en n8n (primer acceso)
3. Restringir acceso al puerto 5678 mediante firewall
4. Usar variables de entorno para secrets
5. Mantener n8n actualizado: `sudo npm update -g n8n`

---

Para más información sobre la integración de n8n con capibara6-consensus, ver:
- [Documentación de n8n](/docs/n8n/README.md)
- [Documentación oficial de n8n](https://docs.n8n.io)
