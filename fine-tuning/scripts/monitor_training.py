#!/usr/bin/env python3
"""
Script de monitoreo para el entrenamiento de GPT-OSS-20B
Proporciona métricas en tiempo real y alertas
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

class TrainingMonitor:
    def __init__(self, model_dir, check_interval=60):
        self.model_dir = model_dir
        self.check_interval = check_interval
        self.start_time = datetime.now()
        
    def get_latest_checkpoint(self):
        """Obtener el checkpoint más reciente"""
        try:
            result = subprocess.run([
                'gsutil', 'ls', f'{self.model_dir}/checkpoint_*'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                checkpoints = result.stdout.strip().split('\n')
                if checkpoints and checkpoints[0]:
                    # Ordenar por número de step
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1]))
                    return checkpoints[-1]
            return None
        except Exception as e:
            print(f"❌ Error obteniendo checkpoints: {e}")
            return None
    
    def get_training_metrics(self):
        """Obtener métricas de entrenamiento"""
        try:
            # Buscar archivos de log más recientes
            result = subprocess.run([
                'gsutil', 'ls', f'{self.model_dir}/logs/*.log'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                log_files = result.stdout.strip().split('\n')
                if log_files and log_files[0]:
                    latest_log = sorted(log_files)[-1]
                    
                    # Leer las últimas líneas del log
                    result = subprocess.run([
                        'gsutil', 'cat', latest_log, '|', 'tail', '-50'
                    ], shell=True, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        return self.parse_metrics(result.stdout)
            
            return {}
        except Exception as e:
            print(f"❌ Error obteniendo métricas: {e}")
            return {}
    
    def parse_metrics(self, log_content):
        """Parsear métricas del log"""
        metrics = {}
        lines = log_content.split('\n')
        
        for line in lines:
            if 'loss' in line.lower() and 'step' in line.lower():
                try:
                    # Buscar patrones como "step: 1000, loss: 2.5"
                    parts = line.split(',')
                    for part in parts:
                        if 'step' in part:
                            step = int(part.split(':')[-1].strip())
                            metrics['step'] = step
                        elif 'loss' in part:
                            loss = float(part.split(':')[-1].strip())
                            metrics['loss'] = loss
                except:
                    pass
        
        return metrics
    
    def check_tpu_health(self):
        """Verificar salud de la TPU"""
        try:
            result = subprocess.run([
                'gcloud', 'compute', 'tpus', 'list', '--filter=name:tx-5-oss-20b'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    status = lines[1].split()[2]  # Status column
                    return status == 'READY'
            return False
        except Exception as e:
            print(f"❌ Error verificando TPU: {e}")
            return False
    
    def calculate_eta(self, current_step, total_steps):
        """Calcular tiempo estimado de finalización"""
        if current_step == 0:
            return "Calculando..."
        
        elapsed = datetime.now() - self.start_time
        steps_per_second = current_step / elapsed.total_seconds()
        remaining_steps = total_steps - current_step
        eta_seconds = remaining_steps / steps_per_second
        
        return str(timedelta(seconds=int(eta_seconds)))
    
    def print_status(self, metrics, checkpoint):
        """Imprimir estado actual"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🔍 Monitoreo de Entrenamiento GPT-OSS-20B")
        print("=" * 50)
        print(f"⏰ Iniciado: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Model Dir: {self.model_dir}")
        print()
        
        if metrics:
            step = metrics.get('step', 0)
            loss = metrics.get('loss', 0.0)
            
            print(f"📊 Paso actual: {step:,}")
            print(f"📉 Pérdida: {loss:.4f}")
            
            if step > 0:
                eta = self.calculate_eta(step, 200000)  # Total steps
                print(f"⏱️ ETA: {eta}")
                
                # Calcular throughput aproximado
                elapsed = datetime.now() - self.start_time
                throughput = step / elapsed.total_seconds() * 60  # steps/min
                print(f"⚡ Throughput: {throughput:.1f} steps/min")
        else:
            print("⏳ Esperando métricas...")
        
        print()
        
        if checkpoint:
            print(f"💾 Último checkpoint: {checkpoint.split('/')[-1]}")
        else:
            print("💾 No hay checkpoints aún")
        
        print()
        
        # Estado de TPU
        tpu_healthy = self.check_tpu_health()
        if tpu_healthy:
            print("✅ TPU: Saludable")
        else:
            print("❌ TPU: Problemas detectados")
        
        print()
        print("🔄 Actualizando cada 60 segundos... (Ctrl+C para salir)")
    
    def run(self):
        """Ejecutar monitoreo continuo"""
        print("🚀 Iniciando monitoreo de entrenamiento...")
        print("   Presiona Ctrl+C para salir")
        
        try:
            while True:
                metrics = self.get_training_metrics()
                checkpoint = self.get_latest_checkpoint()
                
                self.print_status(metrics, checkpoint)
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoreo detenido por el usuario")
        except Exception as e:
            print(f"\n❌ Error en monitoreo: {e}")

def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print("Uso: python monitor_training.py <model_dir>")
        print("Ejemplo: python monitor_training.py gs://bucket/gpt_oss_20b_finetune_model_dir")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    monitor = TrainingMonitor(model_dir)
    monitor.run()

if __name__ == "__main__":
    main()
