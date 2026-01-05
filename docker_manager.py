#!/usr/bin/env python3
"""
Docker Manager - Gestor de contenedores Docker para Capibara6
Permite gestionar todos los contenedores de forma rápida y eficiente.

Uso:
    python docker_manager.py status          # Ver estado de todos los contenedores
    python docker_manager.py stop            # Detener todos los contenedores
    python docker_manager.py start           # Iniciar todos los contenedores
    python docker_manager.py restart         # Reiniciar todos los contenedores
    python docker_manager.py rebuild [service]  # Reconstruir y reiniciar servicio
    python docker_manager.py logs [service]  # Ver logs de un servicio
    python docker_manager.py clean           # Limpiar contenedores detenidos
    python docker_manager.py health          # Ver salud de todos los servicios
"""

import subprocess
import sys
import time
from typing import List, Dict, Optional
from datetime import datetime
import json


class Colors:
    """Colores ANSI para terminal."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DockerManager:
    """Gestor de contenedores Docker."""

    def __init__(self):
        self.compose_files = [
            "capibara6/docker-compose.yml",
            "nebula-docker-compose/docker-compose.yaml"
        ]

        # Grupos de servicios
        self.service_groups = {
            "databases": [
                "capibara6-postgres",
                "capibara6-timescaledb",
                "capibara6-redis",
                "milvus-standalone",
                "milvus-etcd",
                "milvus-minio"
            ],
            "nebula": [
                "nebula-docker-compose-metad0-1",
                "nebula-docker-compose-metad1-1",
                "nebula-docker-compose-metad2-1",
                "nebula-docker-compose-storaged0-1",
                "nebula-docker-compose-storaged1-1",
                "nebula-docker-compose-storaged2-1",
                "nebula-docker-compose-graphd-1",
                "nebula-docker-compose-graphd1-1",
                "nebula-docker-compose-graphd2-1",
                "nebula-docker-compose-studio-1"
            ],
            "application": [
                "capibara6-api",
                "capibara6-worker-1",
                "capibara6-worker-2",
                "capibara6-worker-3",
                "capibara6-nginx",
                "capibara6-n8n"
            ],
            "monitoring": [
                "capibara6-prometheus",
                "capibara6-grafana",
                "capibara6-jaeger"
            ]
        }

    def run_command(self, command: List[str], capture_output=True) -> subprocess.CompletedProcess:
        """Ejecuta un comando y retorna el resultado."""
        try:
            if capture_output:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False
                )
            else:
                result = subprocess.run(command, check=False)
            return result
        except Exception as e:
            print(f"{Colors.FAIL}Error ejecutando comando: {e}{Colors.ENDC}")
            sys.exit(1)

    def get_all_containers(self) -> List[Dict]:
        """Obtiene información de todos los contenedores."""
        result = self.run_command([
            "docker", "ps", "-a",
            "--format", "{{json .}}"
        ])

        containers = []
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        containers.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        return containers

    def print_status(self):
        """Muestra el estado de todos los contenedores."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}        ESTADO DE CONTENEDORES DOCKER - CAPIBARA6{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}═══════════════════════════════════════════════════════════════════{Colors.ENDC}\n")

        containers = self.get_all_containers()

        if not containers:
            print(f"{Colors.WARNING}No hay contenedores Docker en ejecución.{Colors.ENDC}")
            return

        # Agrupar por categorías
        for group_name, service_names in self.service_groups.items():
            group_containers = [c for c in containers if c['Names'] in service_names]

            if group_containers:
                print(f"{Colors.OKCYAN}{Colors.BOLD}▶ {group_name.upper()}{Colors.ENDC}")
                print(f"{Colors.OKCYAN}{'─' * 67}{Colors.ENDC}")

                for container in group_containers:
                    status = container['Status']
                    name = container['Names']

                    # Colorear según estado
                    if 'Up' in status and 'healthy' in status.lower():
                        status_color = Colors.OKGREEN
                        icon = "✓"
                    elif 'Up' in status and 'unhealthy' in status.lower():
                        status_color = Colors.WARNING
                        icon = "⚠"
                    elif 'Up' in status:
                        status_color = Colors.OKBLUE
                        icon = "●"
                    else:
                        status_color = Colors.FAIL
                        icon = "✗"

                    print(f"  {status_color}{icon}{Colors.ENDC} {name:<40} {status_color}{status}{Colors.ENDC}")

                print()

        # Mostrar contenedores no agrupados
        all_grouped = []
        for services in self.service_groups.values():
            all_grouped.extend(services)

        ungrouped = [c for c in containers if c['Names'] not in all_grouped]

        if ungrouped:
            print(f"{Colors.OKCYAN}{Colors.BOLD}▶ OTROS SERVICIOS{Colors.ENDC}")
            print(f"{Colors.OKCYAN}{'─' * 67}{Colors.ENDC}")

            for container in ungrouped:
                status = container['Status']
                name = container['Names']
                status_color = Colors.OKGREEN if 'Up' in status else Colors.FAIL
                icon = "●" if 'Up' in status else "○"
                print(f"  {status_color}{icon}{Colors.ENDC} {name:<40} {status_color}{status}{Colors.ENDC}")

            print()

        # Resumen
        total = len(containers)
        running = len([c for c in containers if 'Up' in c['Status']])
        healthy = len([c for c in containers if 'healthy' in c['Status'].lower()])
        unhealthy = len([c for c in containers if 'unhealthy' in c['Status'].lower()])

        print(f"{Colors.BOLD}RESUMEN:{Colors.ENDC}")
        print(f"  Total: {total} | Running: {Colors.OKGREEN}{running}{Colors.ENDC} | ", end="")
        print(f"Healthy: {Colors.OKGREEN}{healthy}{Colors.ENDC} | ", end="")
        print(f"Unhealthy: {Colors.WARNING}{unhealthy}{Colors.ENDC}")
        print()

    def stop_all(self):
        """Detiene todos los contenedores."""
        print(f"{Colors.WARNING}Deteniendo todos los contenedores...{Colors.ENDC}\n")

        # Detener servicios de aplicación primero
        print("1. Deteniendo servicios de aplicación...")
        for service in self.service_groups["application"]:
            result = self.run_command(["docker", "stop", service])
            if result.returncode == 0:
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {service}")

        # Detener monitoreo
        print("\n2. Deteniendo servicios de monitoreo...")
        for service in self.service_groups["monitoring"]:
            result = self.run_command(["docker", "stop", service])
            if result.returncode == 0:
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {service}")

        # Detener Nebula
        print("\n3. Deteniendo Nebula Graph...")
        for service in self.service_groups["nebula"]:
            result = self.run_command(["docker", "stop", service])
            if result.returncode == 0:
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {service}")

        # Detener bases de datos
        print("\n4. Deteniendo bases de datos...")
        for service in self.service_groups["databases"]:
            result = self.run_command(["docker", "stop", service])
            if result.returncode == 0:
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {service}")

        print(f"\n{Colors.OKGREEN}Todos los contenedores han sido detenidos.{Colors.ENDC}")

    def start_all(self):
        """Inicia todos los contenedores."""
        print(f"{Colors.OKGREEN}Iniciando todos los contenedores...{Colors.ENDC}\n")

        # Iniciar bases de datos primero
        print("1. Iniciando bases de datos...")
        for service in self.service_groups["databases"]:
            result = self.run_command(["docker", "start", service])
            if result.returncode == 0:
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {service}")

        # Esperar un poco para que las bases de datos estén listas
        print(f"\n{Colors.OKCYAN}Esperando 10 segundos para que las bases de datos inicien...{Colors.ENDC}")
        time.sleep(10)

        # Iniciar Nebula
        print("\n2. Iniciando Nebula Graph...")
        for service in self.service_groups["nebula"]:
            result = self.run_command(["docker", "start", service])
            if result.returncode == 0:
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {service}")

        # Esperar un poco
        print(f"\n{Colors.OKCYAN}Esperando 5 segundos...{Colors.ENDC}")
        time.sleep(5)

        # Iniciar monitoreo
        print("\n3. Iniciando servicios de monitoreo...")
        for service in self.service_groups["monitoring"]:
            result = self.run_command(["docker", "start", service])
            if result.returncode == 0:
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {service}")

        # Iniciar aplicación
        print("\n4. Iniciando servicios de aplicación...")
        for service in self.service_groups["application"]:
            result = self.run_command(["docker", "start", service])
            if result.returncode == 0:
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {service}")

        print(f"\n{Colors.OKGREEN}Todos los contenedores han sido iniciados.{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Espera 30 segundos para que los servicios estén completamente listos.{Colors.ENDC}")

    def restart_all(self):
        """Reinicia todos los contenedores rápidamente."""
        print(f"{Colors.BOLD}Reiniciando todos los contenedores...{Colors.ENDC}\n")
        self.stop_all()
        print()
        time.sleep(2)
        self.start_all()

    def rebuild_service(self, service: Optional[str] = None):
        """Reconstruye y reinicia un servicio específico."""
        if not service:
            print(f"{Colors.FAIL}Error: Debes especificar un servicio.{Colors.ENDC}")
            print(f"\nUso: python docker_manager.py rebuild <servicio>")
            print(f"\nEjemplo: python docker_manager.py rebuild capibara6-api")
            return

        print(f"{Colors.OKCYAN}Reconstruyendo servicio: {service}{Colors.ENDC}\n")

        # Determinar el docker-compose correcto
        if "capibara6" in service:
            compose_file = "capibara6/docker-compose.yml"
            service_name = service.replace("capibara6-", "")
        elif "nebula" in service:
            compose_file = "nebula-docker-compose/docker-compose.yaml"
            service_name = service
        else:
            print(f"{Colors.FAIL}Servicio no reconocido.{Colors.ENDC}")
            return

        # Detener
        print(f"1. Deteniendo {service}...")
        self.run_command(["docker", "compose", "-f", compose_file, "stop", service_name])

        # Eliminar
        print(f"2. Eliminando contenedor...")
        self.run_command(["docker", "compose", "-f", compose_file, "rm", "-f", service_name])

        # Reconstruir
        print(f"3. Reconstruyendo imagen...")
        self.run_command(["docker", "compose", "-f", compose_file, "build", service_name], capture_output=False)

        # Iniciar
        print(f"4. Iniciando {service}...")
        self.run_command(["docker", "compose", "-f", compose_file, "up", "-d", service_name])

        print(f"\n{Colors.OKGREEN}Servicio {service} reconstruido y reiniciado.{Colors.ENDC}")

    def show_logs(self, service: Optional[str] = None, lines: int = 50):
        """Muestra logs de un servicio."""
        if not service:
            print(f"{Colors.FAIL}Error: Debes especificar un servicio.{Colors.ENDC}")
            print(f"\nUso: python docker_manager.py logs <servicio>")
            return

        print(f"{Colors.OKCYAN}Mostrando últimas {lines} líneas de logs de {service}:{Colors.ENDC}\n")
        self.run_command(["docker", "logs", "--tail", str(lines), service], capture_output=False)

    def clean(self):
        """Limpia contenedores detenidos y recursos no utilizados."""
        print(f"{Colors.WARNING}Limpiando contenedores detenidos y recursos no utilizados...{Colors.ENDC}\n")

        # Limpiar contenedores
        print("1. Eliminando contenedores detenidos...")
        result = self.run_command(["docker", "container", "prune", "-f"])
        print(result.stdout)

        # Limpiar imágenes sin usar
        print("2. Eliminando imágenes sin usar...")
        result = self.run_command(["docker", "image", "prune", "-f"])
        print(result.stdout)

        # Limpiar volúmenes sin usar
        print("3. Eliminando volúmenes sin usar...")
        result = self.run_command(["docker", "volume", "prune", "-f"])
        print(result.stdout)

        print(f"{Colors.OKGREEN}Limpieza completada.{Colors.ENDC}")

    def check_health(self):
        """Verifica la salud de todos los servicios."""
        print(f"{Colors.HEADER}{Colors.BOLD}VERIFICACIÓN DE SALUD DE SERVICIOS{Colors.ENDC}\n")

        containers = self.get_all_containers()
        running = [c for c in containers if 'Up' in c['Status']]

        for container in running:
            name = container['Names']

            # Verificar healthcheck
            result = self.run_command([
                "docker", "inspect", "--format", "{{json .State.Health}}", name
            ])

            if result.returncode == 0 and result.stdout.strip() != "null":
                try:
                    health = json.loads(result.stdout)
                    status = health.get('Status', 'unknown')

                    if status == 'healthy':
                        print(f"{Colors.OKGREEN}✓{Colors.ENDC} {name:<40} {Colors.OKGREEN}HEALTHY{Colors.ENDC}")
                    elif status == 'unhealthy':
                        print(f"{Colors.FAIL}✗{Colors.ENDC} {name:<40} {Colors.FAIL}UNHEALTHY{Colors.ENDC}")
                    else:
                        print(f"{Colors.WARNING}?{Colors.ENDC} {name:<40} {Colors.WARNING}{status.upper()}{Colors.ENDC}")
                except json.JSONDecodeError:
                    print(f"{Colors.OKCYAN}●{Colors.ENDC} {name:<40} {Colors.OKCYAN}NO HEALTHCHECK{Colors.ENDC}")
            else:
                print(f"{Colors.OKCYAN}●{Colors.ENDC} {name:<40} {Colors.OKCYAN}NO HEALTHCHECK{Colors.ENDC}")


def print_help():
    """Muestra la ayuda del script."""
    print(f"""
{Colors.BOLD}{Colors.HEADER}Docker Manager - Capibara6{Colors.ENDC}

{Colors.BOLD}USO:{Colors.ENDC}
    python docker_manager.py <comando> [opciones]

{Colors.BOLD}COMANDOS:{Colors.ENDC}
    {Colors.OKGREEN}status{Colors.ENDC}              Ver estado de todos los contenedores
    {Colors.OKGREEN}start{Colors.ENDC}               Iniciar todos los contenedores
    {Colors.OKGREEN}stop{Colors.ENDC}                Detener todos los contenedores
    {Colors.OKGREEN}restart{Colors.ENDC}             Reiniciar todos los contenedores
    {Colors.OKGREEN}rebuild{Colors.ENDC} <servicio>  Reconstruir y reiniciar un servicio
    {Colors.OKGREEN}logs{Colors.ENDC} <servicio>     Ver logs de un servicio
    {Colors.OKGREEN}health{Colors.ENDC}              Verificar salud de todos los servicios
    {Colors.OKGREEN}clean{Colors.ENDC}               Limpiar contenedores y recursos no utilizados
    {Colors.OKGREEN}help{Colors.ENDC}                Mostrar esta ayuda

{Colors.BOLD}EJEMPLOS:{Colors.ENDC}
    python docker_manager.py status
    python docker_manager.py restart
    python docker_manager.py rebuild capibara6-api
    python docker_manager.py logs capibara6-api
""")


def main():
    """Función principal."""
    manager = DockerManager()

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "status":
        manager.print_status()

    elif command == "stop":
        manager.stop_all()

    elif command == "start":
        manager.start_all()

    elif command == "restart":
        manager.restart_all()

    elif command == "rebuild":
        service = sys.argv[2] if len(sys.argv) > 2 else None
        manager.rebuild_service(service)

    elif command == "logs":
        service = sys.argv[2] if len(sys.argv) > 2 else None
        manager.show_logs(service)

    elif command == "health":
        manager.check_health()

    elif command == "clean":
        manager.clean()

    elif command == "help":
        print_help()

    else:
        print(f"{Colors.FAIL}Comando no reconocido: {command}{Colors.ENDC}")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
