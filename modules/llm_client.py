"""
LLM Client Module - Multi-provider support with command intelligence.

Supports:
- Portkey: NYU AI Gateway with Claude Sonnet (preferred)
- Local: Ollama (Mistral, Llama, etc.)
- Cloud: OpenAI, Anthropic (optional, requires API keys)

Features:
- Automatic workflow classification for optimal model selection
- Command intelligence layer that validates and fixes commands
- Provider fallback chain for reliability
"""

import httpx
import json
import re
import os
import shlex
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    OLLAMA = 'ollama'
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    PORTKEY = 'portkey'  # NYU AI Gateway (Claude Sonnet)


class WorkflowType(Enum):
    """Workflow types for model selection optimization."""
    SIMPLE = 'simple'       # Quick commands, file ops
    STANDARD = 'standard'   # General queries
    COMPLEX = 'complex'     # Multi-step, reasoning-heavy
    CREATIVE = 'creative'   # Generation, writing tasks


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: LLMProvider
    model: str
    api_url: str
    api_key: Optional[str] = None
    timeout: int = 60
    temperature: float = 0.3
    max_tokens: int = 512
    # For providers with multiple model tiers
    alt_models: Optional[Dict[str, str]] = None


# Default configurations
PROVIDER_CONFIGS = {
    LLMProvider.OLLAMA: LLMConfig(
        provider=LLMProvider.OLLAMA,
        model=os.environ.get('OLLAMA_MODEL', 'mistral'),
        api_url=os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate'),
    ),
    LLMProvider.OPENAI: LLMConfig(
        provider=LLMProvider.OPENAI,
        model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
        api_url='https://api.openai.com/v1/chat/completions',
        api_key=os.environ.get('OPENAI_API_KEY'),
    ),
    LLMProvider.ANTHROPIC: LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model=os.environ.get('ANTHROPIC_MODEL', 'claude-3-haiku-20240307'),
        api_url='https://api.anthropic.com/v1/messages',
        api_key=os.environ.get('ANTHROPIC_API_KEY'),
    ),
    LLMProvider.PORTKEY: LLMConfig(
        provider=LLMProvider.PORTKEY,
        model=os.environ.get('PORTKEY_MODEL', '@anthropic-prod/claude-sonnet-4-5-20250929'),
        api_url=os.environ.get('PORTKEY_URL', 'https://ai-gateway.apps.cloud.rt.nyu.edu/v1/chat/completions'),
        api_key=os.environ.get('PORTKEY_API_KEY'),
        timeout=90,  # Slightly longer for complex queries
        max_tokens=1024,  # Better for versatile workflows
        alt_models={
            'fast': '@anthropic-prod/claude-3-5-haiku-20241022',
            'default': '@anthropic-prod/claude-sonnet-4-5-20250929',
            'powerful': '@anthropic-prod/claude-sonnet-4-5-20250929',
        }
    ),
}

# Provider priority for fallback (Portkey/Sonnet preferred for quality)
PROVIDER_PRIORITY = [LLMProvider.PORTKEY, LLMProvider.OLLAMA, LLMProvider.OPENAI, LLMProvider.ANTHROPIC]


def get_system_context() -> str:
    """Gather current system context to help LLM make better decisions."""
    context_parts = []
    home = Path.home()

    # Current directory
    context_parts.append(f"Current directory: {os.getcwd()}")

    # Home directory contents (top-level only)
    try:
        home_dirs = [d.name for d in home.iterdir() if d.is_dir() and not d.name.startswith('.')][:15]
        context_parts.append(f"Home directories: {', '.join(sorted(home_dirs))}")
    except OSError:
        pass

    # Check for common project directories
    common_paths = ['~/projects', '~/Documents', '~/Downloads', '~/interruption']
    existing = []
    for p in common_paths:
        expanded = Path(p).expanduser()
        if expanded.exists():
            existing.append(p)
    if existing:
        context_parts.append(f"Existing paths: {', '.join(existing)}")

    return '\n'.join(context_parts)


SYSTEM_PROMPT = '''You are a local Arch Linux system assistant. Convert user requests into shell commands.

CRITICAL RULES:
1. Always use `mkdir -p` for creating directories (creates parents automatically)
2. Always use ~ for home directory paths
3. Never use placeholders like <username> - use actual paths
4. Check the system context below to understand what directories exist

SYSTEM CONTEXT:
{context}

Respond with a JSON object:
{{
  "action": "shell command to execute",
  "explanation": "brief explanation",
  "tier": "auto|standard|confirm|blocked"
}}

Tier guidelines:
- auto: Safe read-only (ls, cat, df, pacman -Q)
- standard: User space modifications (mkdir -p, touch, pip install)
- confirm: Potentially impactful (rm -r, pacman -R, sudo)
- blocked: Dangerous (rm -rf /, dd, mkfs)

Respond with valid JSON only.'''


def fix_placeholders(cmd: str) -> str:
    """Replace common LLM placeholders with actual values."""
    if not cmd:
        return cmd

    home = os.path.expanduser('~')
    user = os.environ.get('USER', os.environ.get('USERNAME', 'user'))

    replacements = [
        (r'/home/<[^>]+>', home),
        (r'/home/\[username\]', home),
        (r'/home/\[user\]', home),
        (r'/home/\{username\}', home),
        (r'/home/\{user\}', home),
        (r'/home/\$USER', home),
        (r'/home/\$\{USER\}', home),
        (r'/home/\$\(whoami\)', home),
        (r'<username>', user),
        (r'<user>', user),
        (r'\[username\]', user),
        (r'\[user\]', user),
        (r'\{username\}', user),
        (r'\{user\}', user),
        (r'<home>', home),
        (r'\[home\]', home),
        (r'\{home\}', home),
        (r'<service>', ''),
        (r'<package>', ''),
        (r'<file>', ''),
        (r'<path>', ''),
        (r'<directory>', ''),
        (r'<dir>', ''),
        (r'\[service\]', ''),
        (r'\[package\]', ''),
    ]

    for pattern, replacement in replacements:
        cmd = re.sub(pattern, replacement, cmd, flags=re.IGNORECASE)

    # Expand ~ and environment variables while preserving quoted strings
    try:
        parts = shlex.split(cmd)
        expanded_parts = []
        for part in parts:
            if part.startswith('~') or '$' in part:
                expanded_parts.append(shlex.quote(os.path.expanduser(os.path.expandvars(part))))
            else:
                # Re-quote if part contains spaces or special chars
                expanded_parts.append(shlex.quote(part) if ' ' in part or any(c in part for c in ';&|<>') else part)

        return ' '.join(expanded_parts)
    except ValueError:
        # shlex.split can fail on malformed strings, fall back to simple expansion
        return os.path.expanduser(os.path.expandvars(cmd))


def apply_command_intelligence(cmd: str) -> Tuple[str, List[str]]:
    """
    Apply smart fixes to commands based on common issues.

    Returns:
        Tuple of (fixed_command, list_of_fixes_applied)
    """
    if not cmd:
        return cmd, []

    fixes = []
    original = cmd

    # Fix 1: mkdir without -p flag
    if re.match(r'^mkdir\s+(?!.*-p)', cmd):
        cmd = re.sub(r'^mkdir\s+', 'mkdir -p ', cmd)
        fixes.append('Added -p flag to mkdir (creates parent directories)')

    # Fix 2: rm without -f when dealing with paths that might not exist
    # (be careful - only add -f for specific safe patterns)

    # Fix 3: Ensure cp -r for directories
    # This would need path checking, skip for now

    # Fix 4: Add --noconfirm for pacman -S in non-interactive contexts
    # Skip - user should confirm package installs

    # Fix 5: Use -p with touch for nested paths (touch doesn't support -p, but we can mkdir first)
    touch_match = re.match(r'^touch\s+(.+)$', cmd)
    if touch_match:
        paths_str = touch_match.group(1).strip()
        try:
            paths = shlex.split(paths_str)
        except ValueError:
            paths = paths_str.split()

        missing_parents = []
        for path in paths:
            expanded = os.path.expanduser(path)
            parent = os.path.dirname(expanded)
            if parent and not os.path.exists(parent):
                missing_parents.append(os.path.dirname(path))

        if missing_parents:
            # Deduplicate and create mkdir commands for each missing parent
            unique_parents = list(dict.fromkeys(missing_parents))
            mkdir_cmd = 'mkdir -p ' + ' '.join(shlex.quote(p) for p in unique_parents)
            cmd = f'{mkdir_cmd} && touch {paths_str}'
            fixes.append('Added mkdir -p for parent directories')

    return cmd, fixes


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling common formatting issues."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    match = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {'action': '', 'explanation': text, 'tier': 'blocked'}


def query_ollama(user_input: str, config: LLMConfig, context: str) -> dict:
    """Query Ollama API."""
    prompt = SYSTEM_PROMPT.format(context=context)
    payload = {
        'model': config.model,
        'prompt': f'{prompt}\n\nUser request: {user_input}',
        'stream': False,
        'options': {
            'temperature': config.temperature,
            'num_predict': config.max_tokens
        }
    }

    response = httpx.post(config.api_url, json=payload, timeout=config.timeout)
    response.raise_for_status()
    return extract_json(response.json().get('response', ''))


def query_openai(user_input: str, config: LLMConfig, context: str) -> dict:
    """Query OpenAI API."""
    if not config.api_key:
        raise ValueError("OpenAI API key not configured")

    prompt = SYSTEM_PROMPT.format(context=context)
    payload = {
        'model': config.model,
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_input}
        ],
        'temperature': config.temperature,
        'max_tokens': config.max_tokens
    }

    headers = {
        'Authorization': f'Bearer {config.api_key}',
        'Content-Type': 'application/json'
    }

    response = httpx.post(config.api_url, json=payload, headers=headers, timeout=config.timeout)
    response.raise_for_status()
    content = response.json()['choices'][0]['message']['content']
    return extract_json(content)


def query_anthropic(user_input: str, config: LLMConfig, context: str) -> dict:
    """Query Anthropic API."""
    if not config.api_key:
        raise ValueError("Anthropic API key not configured")

    prompt = SYSTEM_PROMPT.format(context=context)
    payload = {
        'model': config.model,
        'max_tokens': config.max_tokens,
        'temperature': config.temperature,
        'system': prompt,
        'messages': [
            {'role': 'user', 'content': user_input}
        ]
    }

    headers = {
        'x-api-key': config.api_key,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
    }

    response = httpx.post(config.api_url, json=payload, headers=headers, timeout=config.timeout)
    response.raise_for_status()
    content = response.json()['content'][0]['text']
    return extract_json(content)


def classify_workflow(user_input: str) -> WorkflowType:
    """Classify the user request to select optimal model tier."""
    input_lower = user_input.lower()

    # Simple: basic file ops, status checks
    simple_patterns = ['ls', 'pwd', 'cd', 'cat', 'echo', 'df', 'du', 'which', 'whereis']
    if any(p in input_lower for p in simple_patterns) and len(user_input.split()) <= 5:
        return WorkflowType.SIMPLE

    # Complex: multi-step, analysis, debugging, planning
    complex_patterns = [
        'analyze', 'debug', 'explain', 'compare', 'optimize', 'refactor',
        'why', 'how does', 'what if', 'step by step', 'plan', 'design',
        'troubleshoot', 'investigate', 'review'
    ]
    if any(p in input_lower for p in complex_patterns):
        return WorkflowType.COMPLEX

    # Creative: generation, writing
    creative_patterns = ['write', 'create', 'generate', 'draft', 'compose', 'script']
    if any(p in input_lower for p in creative_patterns):
        return WorkflowType.CREATIVE

    return WorkflowType.STANDARD


def query_portkey(user_input: str, config: LLMConfig, context: str, workflow: Optional[WorkflowType] = None) -> dict:
    """
    Query Portkey AI Gateway (NYU) with Claude Sonnet.

    Uses OpenAI-compatible API format. Supports workflow-based model selection
    for optimized performance across different task types.
    """
    if not config.api_key:
        raise ValueError("Portkey API key not configured (set PORTKEY_API_KEY)")

    # Select model based on workflow type
    model = config.model
    if workflow and config.alt_models:
        if workflow == WorkflowType.SIMPLE:
            model = config.alt_models.get('fast', config.model)
        elif workflow in (WorkflowType.COMPLEX, WorkflowType.CREATIVE):
            model = config.alt_models.get('powerful', config.model)

    prompt = SYSTEM_PROMPT.format(context=context)

    # Adjust parameters based on workflow
    temperature = config.temperature
    max_tokens = config.max_tokens

    if workflow == WorkflowType.CREATIVE:
        temperature = 0.7  # More creative responses
        max_tokens = 1024
    elif workflow == WorkflowType.COMPLEX:
        max_tokens = 1024  # More room for detailed explanations

    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_input}
        ],
        'temperature': temperature,
        'max_tokens': max_tokens
    }

    headers = {
        'Authorization': f'Bearer {config.api_key}',
        'Content-Type': 'application/json'
    }

    response = httpx.post(config.api_url, json=payload, headers=headers, timeout=config.timeout)
    response.raise_for_status()

    result = response.json()
    content = result['choices'][0]['message']['content']
    parsed = extract_json(content)
    parsed['model_used'] = model
    parsed['workflow_type'] = workflow.value if workflow else 'auto'
    return parsed


def query(user_input: str, provider: Optional[LLMProvider] = None, workflow: Optional[WorkflowType] = None) -> dict:
    """
    Query LLM for a command with automatic fallback and workflow optimization.

    Args:
        user_input: Natural language request
        provider: Specific provider to use (None = auto with fallback)
        workflow: Workflow type for model optimization (None = auto-detect)

    Returns:
        dict with action, explanation, tier, provider_used, fixes_applied, workflow_type
    """
    context = get_system_context()
    providers_to_try = [provider] if provider else PROVIDER_PRIORITY
    last_error = None

    # Auto-detect workflow type if not specified
    if workflow is None:
        workflow = classify_workflow(user_input)

    for prov in providers_to_try:
        if prov is None:
            continue

        config = PROVIDER_CONFIGS.get(prov)
        if not config:
            continue

        # Skip cloud providers without API keys
        if prov in (LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.PORTKEY) and not config.api_key:
            continue

        try:
            if prov == LLMProvider.OLLAMA:
                result = query_ollama(user_input, config, context)
            elif prov == LLMProvider.OPENAI:
                result = query_openai(user_input, config, context)
            elif prov == LLMProvider.ANTHROPIC:
                result = query_anthropic(user_input, config, context)
            elif prov == LLMProvider.PORTKEY:
                result = query_portkey(user_input, config, context, workflow)
            else:
                continue

            # Normalize tier
            tier = result.get('tier', 'confirm').lower()
            if tier not in ('auto', 'standard', 'confirm', 'blocked'):
                tier = 'confirm'
            result['tier'] = tier

            # Fix placeholders
            if result.get('action'):
                result['action'] = fix_placeholders(result['action'])

                # Apply command intelligence
                fixed_cmd, fixes = apply_command_intelligence(result['action'])
                result['action'] = fixed_cmd
                result['fixes_applied'] = fixes

            result['provider_used'] = prov.value
            result['workflow_type'] = result.get('workflow_type', workflow.value if workflow else 'standard')
            return result

        except httpx.TimeoutException:
            last_error = f'{prov.value}: timeout'
        except httpx.HTTPError as e:
            last_error = f'{prov.value}: {e}'
        except Exception as e:
            last_error = f'{prov.value}: {e}'

    # All providers failed
    return {
        'action': '',
        'explanation': f'All LLM providers failed. Last error: {last_error}',
        'tier': 'blocked',
        'provider_used': None,
        'fixes_applied': [],
        'workflow_type': workflow.value if workflow else 'unknown'
    }


def list_available_providers() -> List[str]:
    """List which providers are currently available."""
    available = []

    # Check Portkey (NYU AI Gateway) first - preferred provider
    if os.environ.get('PORTKEY_API_KEY'):
        available.append('portkey')

    # Check Ollama
    try:
        response = httpx.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            available.append('ollama')
    except httpx.HTTPError:
        pass

    # Check cloud providers by API key presence
    if os.environ.get('OPENAI_API_KEY'):
        available.append('openai')
    if os.environ.get('ANTHROPIC_API_KEY'):
        available.append('anthropic')

    return available
