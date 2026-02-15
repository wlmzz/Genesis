#!/usr/bin/env python3
"""
Genesis - Unified CLI
Interfaccia unificata per tutte le funzionalità Genesis
"""
import argparse
import sys
import subprocess
from pathlib import Path

BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    ██████╗ ███████╗███╗   ██╗███████╗███████╗██╗███████╗      ║
║   ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝██║██╔════╝      ║
║   ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ███████╗██║███████╗      ║
║   ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ╚════██║██║╚════██║      ║
║   ╚██████╔╝███████╗██║ ╚████║███████╗███████║██║███████║      ║
║    ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝╚══════╝      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""

def main():
    parser = argparse.ArgumentParser(
        description="Genesis - Unified Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  genesis.py register --person john_doe          # Registra nuovo volto
  genesis.py zones --source 0                    # Definisci zone
  genesis.py run --cam 0                         # Avvia analisi live
  genesis.py run --video sample.mp4              # Analizza video
  genesis.py dashboard                           # Apri dashboard
  genesis.py report --date 2026-02-15            # Genera report
  genesis.py api                                 # Avvia API server
  genesis.py status                              # System status

For full documentation: README.md, QUICK_START.md, ADVANCED_FEATURES.md
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Comando da eseguire')

    # Register face
    register_parser = subparsers.add_parser('register', help='Registra nuovo volto nel database')
    register_parser.add_argument('--person', required=True, help='ID persona (es: john_doe)')
    register_parser.add_argument('--cam', type=int, default=0, help='ID webcam')

    # Define zones
    zones_parser = subparsers.add_parser('zones', help='Editor visuale zone')
    zones_parser.add_argument('--source', default='0', help='0 per webcam o path video')
    zones_parser.add_argument('--width', type=int, default=960, help='Larghezza frame')

    # Run analysis
    run_parser = subparsers.add_parser('run', help='Avvia analisi')
    run_group = run_parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument('--cam', type=int, help='Analisi live da webcam (ID)')
    run_group.add_argument('--video', help='Analisi da file video')
    run_parser.add_argument('--settings', default='configs/settings.yaml', help='File config')
    run_parser.add_argument('--zones', default='configs/zones.json', help='File zone')

    # Dashboard
    dashboard_parser = subparsers.add_parser('dashboard', help='Avvia dashboard Streamlit')

    # Generate report
    report_parser = subparsers.add_parser('report', help='Genera report automatico')
    report_parser.add_argument('--date', help='Data report (YYYY-MM-DD)')
    report_parser.add_argument('--outdir', default='data/outputs/reports', help='Directory output')

    # API Server
    api_parser = subparsers.add_parser('api', help='Avvia REST API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host')
    api_parser.add_argument('--port', type=int, default=8000, help='Porta')

    # Status
    status_parser = subparsers.add_parser('status', help='System status & health check')

    # Setup
    setup_parser = subparsers.add_parser('setup', help='Setup iniziale Genesis')

    # Version
    version_parser = subparsers.add_parser('version', help='Mostra versione')

    args = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
        return

    print(BANNER)

    # Execute command
    try:
        if args.command == 'register':
            print(f"📸 Registrazione volto: {args.person}\n")
            subprocess.run([
                sys.executable, 'app/face_register.py',
                '--person_id', args.person,
                '--cam', str(args.cam)
            ], check=True)

        elif args.command == 'zones':
            print(f"📍 Editor zone\n")
            subprocess.run([
                sys.executable, 'app/zone_editor.py',
                '--source', args.source,
                '--width', str(args.width)
            ], check=True)

        elif args.command == 'run':
            if args.cam is not None:
                print(f"🎥 Analisi live da webcam {args.cam}\n")
                subprocess.run([
                    sys.executable, 'app/run_camera.py',
                    '--cam', str(args.cam),
                    '--settings', args.settings,
                    '--zones', args.zones
                ], check=True)
            else:
                print(f"📹 Analisi video: {args.video}\n")
                subprocess.run([
                    sys.executable, 'app/run_video.py',
                    '--video', args.video,
                    '--settings', args.settings,
                    '--zones', args.zones
                ], check=True)

        elif args.command == 'dashboard':
            print("📊 Avvio dashboard...\n")
            print("Dashboard disponibile su: http://localhost:8501\n")
            subprocess.run([
                'streamlit', 'run', 'app/dashboard.py'
            ], check=True)

        elif args.command == 'report':
            print("📝 Generazione report...\n")
            cmd = [sys.executable, 'app/report_generator.py']
            if args.date:
                cmd.extend(['--date', args.date])
            cmd.extend(['--outdir', args.outdir])
            subprocess.run(cmd, check=True)

        elif args.command == 'api':
            print(f"🌐 Avvio API server su {args.host}:{args.port}\n")
            print(f"API disponibile su: http://localhost:{args.port}")
            print(f"Docs interattivi: http://localhost:{args.port}/docs\n")
            subprocess.run([
                sys.executable, 'app/api_server.py'
            ], check=True)

        elif args.command == 'status':
            print("🔍 System Status Check\n")
            check_system_status()

        elif args.command == 'setup':
            print("⚙️  Setup Genesis\n")
            subprocess.run(['bash', 'setup.sh'], check=True)

        elif args.command == 'version':
            print("Genesis Version 1.0.0")
            print("Facial Recognition & Analytics System")
            print("Python modules:")
            try:
                import ultralytics
                print(f"  - YOLO: {ultralytics.__version__}")
            except:
                pass
            try:
                import deepface
                print(f"  - DeepFace: OK")
            except:
                pass
            try:
                import streamlit
                print(f"  - Streamlit: {streamlit.__version__}")
            except:
                pass

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error executing command: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

def check_system_status():
    """Check system health"""
    import os
    import requests

    checks = []

    # Check files
    checks.append(("Config file", os.path.exists("configs/settings.yaml")))
    checks.append(("Zones file", os.path.exists("configs/zones.json")))
    checks.append(("Data directory", os.path.exists("data")))

    # Check database
    db_exists = os.path.exists("data/outputs/identities.db")
    checks.append(("Identity database", db_exists))
    if db_exists:
        import sqlite3
        conn = sqlite3.connect("data/outputs/identities.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM identities")
        event_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        conn.close()
        print(f"  📊 Database: {event_count} events, {session_count} sessions")

    # Check metrics
    csv_exists = os.path.exists("data/outputs/metrics.csv")
    checks.append(("Metrics file", csv_exists))
    if csv_exists:
        import pandas as pd
        df = pd.read_csv("data/outputs/metrics.csv")
        print(f"  📈 Metrics: {len(df)} records")

    # Check Ollama
    ollama_ok = False
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_ok = response.status_code == 200
        if ollama_ok:
            models = response.json().get("models", [])
            print(f"  🧠 Ollama: {len(models)} models available")
    except:
        pass
    checks.append(("Ollama server", ollama_ok))

    # Check face database
    faces_dir = Path("data/faces")
    if faces_dir.exists():
        known_faces = len([d for d in faces_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
        checks.append(("Known faces", known_faces > 0))
        print(f"  👤 Faces registered: {known_faces}")

    # Print summary
    print("\nSystem Status:")
    for name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {name}")

    all_ok = all(status for _, status in checks)
    if all_ok:
        print("\n🎉 System ready!")
    else:
        print("\n⚠️  Some components missing. Run: ./setup.sh")

if __name__ == "__main__":
    main()
