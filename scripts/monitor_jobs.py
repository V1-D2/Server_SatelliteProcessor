#!/usr/bin/env python3
"""
Job monitoring script for Server_SatelliteProcessor
Provides real-time status of jobs and system resources
"""

import os
import sys
import json
import time
import psutil
import pathlib
import subprocess
from datetime import datetime, timedelta
from collections import Counter

# Add parent directory to path
SERVER_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))


class JobMonitor:
    """Monitor jobs and system status"""

    def __init__(self):
        self.server_root = SERVER_ROOT
        self.jobs_dir = self.server_root / 'jobs'
        self.results_dir = self.server_root / 'results'
        self.logs_dir = self.server_root / 'logs'

    def get_job_counts(self):
        """Get count of jobs in each status"""
        counts = {}
        for status in ['pending', 'running', 'completed', 'failed']:
            status_dir = self.jobs_dir / status
            if status_dir.exists():
                counts[status] = len(list(status_dir.glob('*.json')))
            else:
                counts[status] = 0
        return counts

    def get_recent_jobs(self, status='completed', limit=5):
        """Get recent jobs of given status"""
        status_dir = self.jobs_dir / status
        if not status_dir.exists():
            return []

        jobs = []
        for job_file in sorted(status_dir.glob('*.json'),
                               key=lambda p: p.stat().st_mtime,
                               reverse=True)[:limit]:
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                    jobs.append({
                        'id': job_file.stem,
                        'function': job_data.get('function', 'unknown'),
                        'status': job_data.get('status', status),
                        'time': datetime.fromtimestamp(job_file.stat().st_mtime)
                    })
            except:
                pass

        return jobs

    def get_slurm_status(self):
        """Get SLURM job status"""
        try:
            result = subprocess.run(
                ['squeue', '-u', 'vdidur', '-o', '%j %T %M %R'],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                jobs = []
                for line in lines:
                    if line:
                        parts = line.split()
                        if len(parts) >= 4:
                            jobs.append({
                                'name': parts[0],
                                'state': parts[1],
                                'time': parts[2],
                                'reason': ' '.join(parts[3:])
                            })
                return jobs
            return []
        except:
            return []

    def get_gpu_status(self):
        """Get GPU utilization"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpus.append({
                                'name': parts[0],
                                'memory_used': int(parts[1]),
                                'memory_total': int(parts[2]),
                                'utilization': int(parts[3])
                            })
                return gpus
            return []
        except:
            return []

    def get_disk_usage(self):
        """Get disk usage for key directories"""
        usage = {}

        for name, path in [
            ('Results', self.results_dir),
            ('Logs', self.logs_dir),
            ('Jobs', self.jobs_dir)
        ]:
            if path.exists():
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                usage[name] = total_size / (1024 * 1024 * 1024)  # GB
            else:
                usage[name] = 0

        # Overall disk usage
        disk_usage = psutil.disk_usage('/home/vdidur')
        usage['Total Used'] = disk_usage.used / (1024 * 1024 * 1024)
        usage['Total Free'] = disk_usage.free / (1024 * 1024 * 1024)
        usage['Percent'] = disk_usage.percent

        return usage

    def get_processing_stats(self, hours=24):
        """Get processing statistics for last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        stats = {
            'total_jobs': 0,
            'successful': 0,
            'failed': 0,
            'by_function': Counter(),
            'avg_duration': 0
        }

        # Check completed and failed jobs
        for status in ['completed', 'failed']:
            status_dir = self.jobs_dir / status
            if not status_dir.exists():
                continue

            for job_file in status_dir.glob('*.json'):
                if datetime.fromtimestamp(job_file.stat().st_mtime) > cutoff_time:
                    try:
                        with open(job_file, 'r') as f:
                            job_data = json.load(f)

                        stats['total_jobs'] += 1
                        if status == 'completed':
                            stats['successful'] += 1
                        else:
                            stats['failed'] += 1

                        function = job_data.get('function', 'unknown')
                        stats['by_function'][function] += 1

                        # Calculate duration if available
                        if 'start_time' in job_data and 'end_time' in job_data:
                            start = datetime.fromisoformat(job_data['start_time'])
                            end = datetime.fromisoformat(job_data['end_time'])
                            duration = (end - start).total_seconds()
                            stats['avg_duration'] += duration
                    except:
                        pass

        # Calculate average duration
        if stats['successful'] > 0:
            stats['avg_duration'] /= stats['successful']

        return stats

    def print_status(self):
        """Print formatted status report"""
        os.system('clear')

        print("=" * 80)
        print(f"SatelliteProcessor Server Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Job counts
        counts = self.get_job_counts()
        print("\n📊 JOB STATUS:")
        print(f"  Pending:   {counts['pending']:3d}")
        print(f"  Running:   {counts['running']:3d}")
        print(f"  Completed: {counts['completed']:3d}")
        print(f"  Failed:    {counts['failed']:3d}")

        # SLURM jobs
        slurm_jobs = self.get_slurm_status()
        print("\n🖥️  SLURM JOBS:")
        if slurm_jobs:
            for job in slurm_jobs:
                print(f"  {job['name']:<20} {job['state']:<10} {job['time']:<10}")
        else:
            print("  No active SLURM jobs")

        # GPU status
        gpus = self.get_gpu_status()
        if gpus:
            print("\n🎮 GPU STATUS:")
            for i, gpu in enumerate(gpus):
                mem_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                print(f"  GPU {i}: {gpu['name']}")
                print(f"    Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({mem_percent:.1f}%)")
                print(f"    Utilization: {gpu['utilization']}%")

        # Recent jobs
        print("\n📝 RECENT COMPLETED JOBS:")
        recent = self.get_recent_jobs('completed', 5)
        if recent:
            for job in recent:
                print(f"  {job['time'].strftime('%H:%M:%S')} - {job['function']:<20} - {job['id'][:20]}...")
        else:
            print("  No recent completions")

        # Failed jobs
        failed = self.get_recent_jobs('failed', 3)
        if failed:
            print("\n❌ RECENT FAILED JOBS:")
            for job in failed:
                print(f"  {job['time'].strftime('%H:%M:%S')} - {job['function']:<20} - {job['id'][:20]}...")

        # Disk usage
        disk = self.get_disk_usage()
        print(f"\n💾 DISK USAGE:")
        print(f"  Results:    {disk['Results']:.2f} GB")
        print(f"  Logs:       {disk['Logs']:.2f} GB")
        print(f"  Jobs:       {disk['Jobs']:.2f} GB")
        print(f"  Total Used: {disk['Total Used']:.1f} GB ({disk['Percent']:.1f}%)")
        print(f"  Free Space: {disk['Total Free']:.1f} GB")

        # Processing stats
        stats = self.get_processing_stats(24)
        if stats['total_jobs'] > 0:
            print(f"\n📈 LAST 24 HOURS:")
            print(f"  Total Jobs: {stats['total_jobs']}")
            print(f"  Success Rate: {(stats['successful'] / stats['total_jobs'] * 100):.1f}%")
            if stats['avg_duration'] > 0:
                print(f"  Avg Duration: {int(stats['avg_duration'] // 60)}m {int(stats['avg_duration'] % 60)}s")

            print("  By Function:")
            for func, count in stats['by_function'].most_common():
                print(f"    {func}: {count}")

        print("\n" + "=" * 80)
        print("Press Ctrl+C to exit")

    def run(self, refresh_interval=5):
        """Run continuous monitoring"""
        try:
            while True:
                self.print_status()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


if __name__ == "__main__":
    monitor = JobMonitor()

    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        # Single status report
        monitor.print_status()
    else:
        # Continuous monitoring
        monitor.run()