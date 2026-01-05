# 🚀 Guía Rápida de Alias - Docker Manager Capibara6

## Activar los Alias

Los alias están configurados en `~/.bashrc`. Para activarlos en la sesión actual:

```bash
source ~/.bashrc
```

O simplemente abre una nueva terminal.

---

## 📋 Comandos Principales

### Ver Estado
```bash
dstatus
```
Muestra el estado de todos los contenedores con colores y organizado por categorías.

### Iniciar Todo
```bash
dstart
```
Inicia todos los contenedores en el orden correcto (bases de datos → nebula → monitoreo → aplicación).

### Detener Todo
```bash
dstop
```
Detiene todos los contenedores en orden inverso.

### Reiniciar Todo
```bash
drestart
```
Detiene y vuelve a iniciar todos los contenedores rápidamente.

### Verificar Salud
```bash
dhealth
```
Muestra el estado de healthcheck de todos los servicios.

---

## 🔧 Comandos Específicos

### Ver Logs
```bash
dlogs capibara6-api          # Logs de un servicio específico
dlogs capibara6-postgres     # Logs de PostgreSQL
dlogs nebula-docker-compose-graphd-1  # Logs de Nebula
```

**Atajos predefinidos:**
```bash
dapi-logs       # Ver logs del API
dworker-logs    # Ver logs del worker 1
```

### Reconstruir Servicios
```bash
drebuild capibara6-api       # Reconstruir el API
drebuild capibara6-nginx     # Reconstruir Nginx
```

**Atajo predefinido:**
```bash
dapi-rebuild    # Reconstruir API rápidamente
```

### Limpiar Recursos
```bash
dclean
```
Elimina contenedores detenidos, imágenes sin usar y volúmenes no utilizados.

### Ayuda
```bash
dhelp
```
Muestra la ayuda completa del Docker Manager.

---

## 📂 Navegación Rápida

```bash
cdcapi      # cd ~/capibara6
cdcback     # cd ~/capibara6/backend
cdcweb      # cd ~/capibara6/web
```

---

## 💡 Ejemplos de Uso Común

### Workflow de Desarrollo Típico

1. **Ver estado actual:**
   ```bash
   dstatus
   ```

2. **Ver logs del API mientras desarrollas:**
   ```bash
   dapi-logs
   ```

3. **Después de modificar código, reconstruir:**
   ```bash
   dapi-rebuild
   ```

4. **Verificar que esté healthy:**
   ```bash
   dhealth
   ```

### Reinicio Rápido

```bash
drestart  # Todo en uno!
```

### Troubleshooting

```bash
# Ver estado
dstatus

# Verificar salud
dhealth

# Ver logs del servicio problemático
dlogs capibara6-api

# Si hay problema, reconstruir
dapi-rebuild
```

### Mantenimiento

```bash
# Limpiar espacio en disco
dclean

# Ver estado después de limpiar
dstatus
```

---

## 🎨 Interpretación de Colores

Cuando ejecutes `dstatus` o `dhealth`, verás colores:

- 🟢 **Verde (✓)**: Servicio healthy/funcionando correctamente
- 🟡 **Amarillo (⚠)**: Warning o unhealthy
- 🔵 **Azul (●)**: Running pero sin healthcheck
- 🔴 **Rojo (✗)**: Detenido o error

---

## 📝 Notas Importantes

1. **Los alias funcionan en cualquier directorio** - No necesitas estar en ~/capibara6

2. **Los cambios en docker-compose.yml requieren reiniciar** los contenedores:
   ```bash
   drestart
   ```

3. **Para ver todos los alias disponibles**:
   ```bash
   alias | grep "^d"
   ```

4. **Los logs muestran últimas 50 líneas por defecto** - Para más:
   ```bash
   python3 /home/elect/docker_manager.py logs capibara6-api  # Personalizar en el script
   ```

---

## 🔗 Comandos Equivalentes

| Alias | Comando Completo |
|-------|------------------|
| `dstatus` | `python3 /home/elect/docker_manager.py status` |
| `drestart` | `python3 /home/elect/docker_manager.py restart` |
| `dhealth` | `python3 /home/elect/docker_manager.py health` |
| `dapi-logs` | `python3 /home/elect/docker_manager.py logs capibara6-api` |

---

## ⚡ Tips Pro

1. **Combinación con watch para monitoreo continuo:**
   ```bash
   watch -n 5 'source ~/.bashrc && dhealth'
   ```

2. **Ver logs en tiempo real:**
   ```bash
   docker logs -f capibara6-api
   ```

3. **Reinicio selectivo de servicios:**
   ```bash
   cd ~/capibara6
   docker compose restart capibara6-api
   ```

---

*Creado para Capibara6 - Anachroni s.coop*
*Última actualización: 2025-11-11*
