import yaml
import subprocess
import json
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'permissions.yaml'
LOG_PATH = Path(__file__).parent.parent / 'logs' / 'action_log.jsonl'

# Cache config at module load
_config: Optional[dict] = None

def load_config() -> dict:
    """Load and cache permissions config."""
    global _config
    if _config is None:
        with open(CONFIG_PATH) as f:
            _config = yaml.safe_load(f)
    return _config

def reload_config() -> dict:
    """Force reload of config (useful after editing permissions.yaml)."""
    global _config
    _config = None
    return load_config()

def expand_path(path: str) -> str:
    """Expand ~ and environment variables in paths."""
    return os.path.expanduser(os.path.expandvars(path))

def is_trusted_path(cmd: str) -> bool:
    """Check if command operates on a trusted path."""
    config = load_config()
    trusted = config.get('trusted_paths', [])

    # Extract paths from command (simplified heuristic)
    parts = cmd.split()
    for part in parts[1:]:  # Skip the command itself
        if part.startswith('-'):
            continue
        expanded = expand_path(part)
        for trusted_path in trusted:
            trusted_expanded = expand_path(trusted_path)
            if expanded.startswith(trusted_expanded):
                return True
    return False

def get_task_command(user_input: str) -> Optional[str]:
    """Check if input matches a task shortcut."""
    config = load_config()
    tasks = config.get('tasks', {})
    key = user_input.strip().lower()
    return tasks.get(key)

def classify_command(cmd: str, llm_tier: Optional[str] = None) -> str:
    """
    Classify command into permission tier.

    Priority order: blocked -> confirm -> standard -> auto -> unknown
    LLM suggestion is considered but guard has final say on blocked/confirm.
    """
    config = load_config()
    tiers = config.get('tiers', {})
    cmd = cmd.strip()

    # Check tiers in security order (most restrictive first)
    tier_order = ['blocked', 'confirm', 'standard', 'auto']

    for tier in tier_order:
        patterns = tiers.get(tier, [])
        for pattern in patterns:
            try:
                if re.search(pattern, cmd):
                    # If command is in trusted path and tier is standard, upgrade to auto
                    if tier == 'standard' and is_trusted_path(cmd):
                        return 'auto'
                    return tier
            except re.error:
                # Invalid regex pattern, skip it
                continue

    # No pattern matched - use LLM suggestion if provided, else default to confirm
    if llm_tier and llm_tier in tier_order:
        # But never allow LLM to override to auto for unmatched commands
        if llm_tier == 'auto':
            return 'standard'
        return llm_tier

    return 'confirm'  # Default to confirm for unknown commands

def validate_command(cmd: str) -> Tuple[bool, str]:
    """
    Validate command for dangerous patterns before execution.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for unfixed placeholders that could cause shell interpretation issues
    dangerous_patterns = [
        (r'<[a-zA-Z_]+>', 'Unfixed placeholder detected (angle brackets)'),
        (r'\[[a-zA-Z_]+\](?!/)', 'Unfixed placeholder detected (square brackets)'),
        (r'\{[a-zA-Z_]+\}', 'Unfixed placeholder detected (curly braces)'),
    ]

    for pattern, message in dangerous_patterns:
        match = re.search(pattern, cmd)
        if match:
            return False, f'{message}: {match.group()}'

    # Check for empty/malformed commands
    if not cmd or not cmd.strip():
        return False, 'Empty command'

    # Check for commands that are just whitespace and special chars
    if re.match(r'^[\s\-]+$', cmd):
        return False, 'Malformed command'

    return True, ''


def log_action(cmd: str, tier: str, approved: bool, output: str = ''):
    """Log command execution to JSONL file."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'command': cmd,
        'tier': tier,
        'approved': approved,
        'output': output[:1000] if output else ''  # Truncate long output
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')

def execute(cmd: str, llm_tier: Optional[str] = None, dry_run: bool = False) -> dict:
    """
    Execute command with permission checking.

    Args:
        cmd: The shell command to execute
        llm_tier: The tier suggested by the LLM (optional)
        dry_run: If True, don't actually execute, just classify

    Returns:
        dict with status, message, stdout, stderr, tier
    """
    # Validate command before doing anything
    is_valid, error_msg = validate_command(cmd)
    if not is_valid:
        return {
            'status': 'error',
            'tier': 'blocked',
            'command': cmd,
            'stdout': '',
            'stderr': error_msg,
            'message': f'Command validation failed: {error_msg}'
        }

    tier = classify_command(cmd, llm_tier)

    result = {
        'status': 'pending',
        'tier': tier,
        'command': cmd,
        'stdout': '',
        'stderr': '',
        'message': ''
    }

    # Handle blocked commands
    if tier == 'blocked':
        log_action(cmd, tier, False)
        result['status'] = 'blocked'
        result['message'] = f'Command blocked for safety: {cmd}'
        return result

    # Handle dry run
    if dry_run:
        result['status'] = 'dry_run'
        result['message'] = f'Would execute ({tier}): {cmd}'
        return result

    # Handle confirm tier - requires user approval
    if tier == 'confirm':
        confirm = input(f'\n[CONFIRM {tier.upper()}] Execute: {cmd}\n(yes/no): ').strip().lower()
        if confirm != 'yes':
            log_action(cmd, tier, False)
            result['status'] = 'denied'
            result['message'] = 'User denied execution'
            return result

    # Execute the command
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60  # Increased timeout
        )
        log_action(cmd, tier, True, proc.stdout)
        result['status'] = 'ok'
        result['stdout'] = proc.stdout
        result['stderr'] = proc.stderr
        if proc.returncode != 0:
            result['status'] = 'error'
            result['message'] = f'Exit code: {proc.returncode}'
    except subprocess.TimeoutExpired:
        log_action(cmd, tier, True, 'TIMEOUT')
        result['status'] = 'timeout'
        result['message'] = 'Command timed out (60s limit)'
    except Exception as e:
        log_action(cmd, tier, False, str(e))
        result['status'] = 'error'
        result['message'] = str(e)

    return result
