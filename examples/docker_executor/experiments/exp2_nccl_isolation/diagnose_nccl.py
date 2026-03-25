#!/usr/bin/env python3
"""
NCCL Socket Diagnostic Script - Extended
Tests socket visibility across containers
"""
import os
import sys
import time
import glob
import socket
import subprocess
from pathlib import Path

def check_tmp_contents():
    """List all files in /tmp"""
    print("\n" + "="*70)
    print("Contents of /tmp:")
    print("="*70)
    
    try:
        all_files = list(Path("/tmp").glob("*"))
        if all_files:
            for f in sorted(all_files):
                try:
                    stat = os.stat(f)
                    print(f"  {f}")
                    print(f"    Mode: {oct(stat.st_mode)}, UID: {stat.st_uid}, Size: {stat.st_size}, Inode: {stat.st_ino}")
                except Exception as e:
                    print(f"  {f} - Error: {e}")
        else:
            print("  Empty directory")
    except Exception as e:
        print(f"  Error listing /tmp: {e}")
    
    print()

def check_namespaces():
    """Check current namespace information"""
    print("\n" + "="*70)
    print("Namespace Information:")
    print("="*70)
    
    namespaces = ['ipc', 'pid', 'uts', 'mnt', 'net']
    for ns in namespaces:
        try:
            ns_link = os.readlink(f'/proc/self/ns/{ns}')
            print(f"  {ns.upper():6s} Namespace: {ns_link}")
        except Exception as e:
            print(f"  {ns.upper():6s} Namespace: Error - {e}")
    
    print(f"  Hostname: {socket.gethostname()}")
    print(f"  PID: {os.getpid()}")
    print()

def create_test_socket():
    """Create a test socket and report its path"""
    rank = os.environ.get('RANK', '0')
    socket_path = f"/tmp/nccl-test-rank{rank}-{os.getpid()}"
    
    print("\n" + "="*70)
    print(f"Creating test socket: {socket_path}")
    print("="*70)
    
    try:
        if os.path.exists(socket_path):
            os.remove(socket_path)
        
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(socket_path)
        
        if os.path.exists(socket_path):
            stat = os.stat(socket_path)
            print(f"  ✓ Socket created successfully")
            print(f"    Path: {socket_path}")
            print(f"    Mode: {oct(stat.st_mode)}, Inode: {stat.st_ino}")
            
            # List all sockets to see if others are visible
            print(f"\n  All NCCL test sockets in /tmp:")
            test_sockets = glob.glob("/tmp/nccl-test-*")
            for s in test_sockets:
                st = os.stat(s)
                print(f"    {s} (inode: {st.st_ino})")
        else:
            print(f"  ✗ Socket file NOT visible after creation!")
        
        return sock, socket_path
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None, None
    
    print()

def check_mount_info():
    """Check mount information for /tmp"""
    print("\n" + "="*70)
    print("/tmp Mount Information:")
    print("="*70)
    
    try:
        with open('/proc/self/mountinfo', 'r') as f:
            for line in f:
                if '/tmp' in line:
                    print(f"  {line.strip()}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()

def main():
    rank = os.environ.get('RANK', 'unknown')
    
    print("\n" + "="*70)
    print(f"NCCL Diagnostic Report - Rank {rank}")
    print("="*70)
    
    check_namespaces()
    check_mount_info()
    check_tmp_contents()
    
    sock, sock_path = create_test_socket()
    
    print("\n" + "="*70)
    print(f"Rank {rank}: Sleeping 10 seconds to allow other containers to start...")
    print("="*70)
    time.sleep(10)
    
    print("\n" + "="*70)
    print("Final check - looking for sockets from other ranks:")
    print("="*70)
    check_tmp_contents()
    
    # Cleanup
    if sock:
        sock.close()
    if sock_path and os.path.exists(sock_path):
        os.remove(sock_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
