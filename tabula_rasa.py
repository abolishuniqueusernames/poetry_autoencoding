#!/usr/bin/env python3
"""
Tabula Rasa - Clean Slate Testing Script
Safely removes audit logs, PDFs, and test files from /data directory for reproducibility testing.
"""

import os
import sys
import argparse
from pathlib import Path
import fnmatch

def confirm_deletion(file_path, dry_run=False):
    """Confirm deletion with user (unless dry run)"""
    if dry_run:
        print(f"[DRY RUN] Would delete: {file_path}")
        return True
    
    response = input(f"Delete {file_path}? (y/N): ")
    return response.lower() in ['y', 'yes']

def find_files_to_delete(data_dir="/data"):
    """
    Find files matching deletion criteria:
    - Audit logs (*.log, *audit*, *log*)
    - PDFs (*.pdf)
    - Files containing "test" in name
    """
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist")
        return []
    
    files_to_delete = []
    data_path = Path(data_dir)
    
    # Recursively find matching files
    for file_path in data_path.rglob('*'):
        if file_path.is_file():
            filename = file_path.name.lower()
            
            # Check deletion criteria
            should_delete = False
            reason = ""
            
            # Audit logs
            if (filename.endswith('.log') or 
                'audit' in filename or 
                'log' in filename):
                should_delete = True
                reason = "audit/log file"
            
            # PDFs
            elif filename.endswith('.pdf'):
                should_delete = True
                reason = "PDF file"
            
            # Test files
            elif 'test' in filename:
                should_delete = True
                reason = "test file"
            
            if should_delete:
                files_to_delete.append((file_path, reason))
    
    return files_to_delete

def tabula_rasa(data_dir="/data", dry_run=False, force=False, interactive=True):
    """
    Main cleanup function with safety checks
    
    Args:
        data_dir: Directory to clean (default: /data)
        dry_run: Only show what would be deleted
        force: Skip individual confirmations
        interactive: Ask for overall confirmation
    """
    print(f"🧹 TABULA RASA - Clean Slate Script")
    print(f"Target directory: {data_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE DELETION'}")
    print("-" * 50)
    
    # Safety check: Ensure we're not targeting critical directories
    critical_dirs = ['/', '/home', '/usr', '/bin', '/etc', '/var', '/sys', '/proc']
    if data_dir in critical_dirs:
        print(f"❌ ERROR: Cannot target critical system directory: {data_dir}")
        return False
    
    # Find files to delete
    files_to_delete = find_files_to_delete(data_dir)
    
    if not files_to_delete:
        print(f"✅ No files found matching deletion criteria in {data_dir}")
        return True
    
    print(f"📋 Found {len(files_to_delete)} files matching deletion criteria:")
    print()
    
    # Group by type for display
    by_type = {}
    for file_path, reason in files_to_delete:
        if reason not in by_type:
            by_type[reason] = []
        by_type[reason].append(file_path)
    
    # Display files by type
    for file_type, files in by_type.items():
        print(f"📁 {file_type.upper()}: {len(files)} files")
        for file_path in files[:5]:  # Show first 5
            print(f"   {file_path}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more")
        print()
    
    # Overall confirmation
    if interactive and not dry_run:
        print(f"⚠️  This will permanently delete {len(files_to_delete)} files!")
        response = input("Continue with deletion? (yes/NO): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Operation cancelled by user")
            return False
    
    # Delete files
    deleted_count = 0
    skipped_count = 0
    
    for file_path, reason in files_to_delete:
        try:
            if dry_run:
                print(f"[DRY RUN] Would delete: {file_path} ({reason})")
                deleted_count += 1
            else:
                if force or not interactive or confirm_deletion(file_path, dry_run):
                    file_path.unlink()
                    print(f"✓ Deleted: {file_path} ({reason})")
                    deleted_count += 1
                else:
                    print(f"⏭️  Skipped: {file_path}")
                    skipped_count += 1
        
        except Exception as e:
            print(f"❌ Error deleting {file_path}: {e}")
            skipped_count += 1
    
    print()
    print("=" * 50)
    if dry_run:
        print(f"🔍 DRY RUN COMPLETE: Would have deleted {deleted_count} files")
    else:
        print(f"✅ CLEANUP COMPLETE: {deleted_count} files deleted, {skipped_count} skipped")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Tabula Rasa - Clean slate testing script for reproducibility"
    )
    parser.add_argument(
        '--data-dir', 
        default='/data',
        help='Target directory for cleanup (default: /data)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Skip individual file confirmations (still asks for overall confirmation)'
    )
    parser.add_argument(
        '--yes', 
        action='store_true',
        help='Skip all confirmations (DANGEROUS - use with caution)'
    )
    
    args = parser.parse_args()
    
    # Safety check for --yes flag
    if args.yes and not args.dry_run:
        print("⚠️  WARNING: --yes flag will delete files without confirmation!")
        response = input("Are you absolutely sure? Type 'CONFIRM' to proceed: ")
        if response != 'CONFIRM':
            print("❌ Operation cancelled")
            return 1
    
    success = tabula_rasa(
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        force=args.force,
        interactive=not args.yes
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())