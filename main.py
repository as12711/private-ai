import sys
import argparse
sys.path.insert(0, '.')

from modules.llm_client import query, list_available_providers
from modules.permission_guard import execute, get_task_command, load_config
from modules.file_manager import file_manager
from modules.workflow import (
    Workflow, create_project_workflow, clear_cache_workflow, backup_config_workflow
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

TIER_STYLES = {
    'auto': 'green',
    'standard': 'cyan',
    'confirm': 'yellow',
    'blocked': 'red'
}

# Built-in workflows
WORKFLOWS = {
    'new-project': lambda args: create_project_workflow(args[0] if args else 'my_project'),
    'clear-cache': lambda args: clear_cache_workflow(include_pip='--pip' in args, include_npm='--npm' in args),
    'backup-config': lambda args: backup_config_workflow(args[0] if args else '~/backups/config'),
}

def show_tasks():
    """Display available task shortcuts."""
    config = load_config()
    tasks = config.get('tasks', {})

    if tasks:
        table = Table(title='Task Shortcuts', show_header=True)
        table.add_column('Shortcut', style='cyan')
        table.add_column('Command', style='dim')

        for name, cmd in sorted(tasks.items()):
            table.add_row(name, cmd)

        console.print(table)

    # Show workflows
    wf_table = Table(title='Workflows', show_header=True)
    wf_table.add_column('Name', style='magenta')
    wf_table.add_column('Usage', style='dim')

    wf_table.add_row('new-project', 'workflow new-project <name>')
    wf_table.add_row('clear-cache', 'workflow clear-cache [--pip] [--npm]')
    wf_table.add_row('backup-config', 'workflow backup-config [path]')

    console.print(wf_table)
    console.print()


def show_providers():
    """Display available LLM providers."""
    available = list_available_providers()

    table = Table(title='LLM Providers', show_header=True)
    table.add_column('Provider', style='cyan')
    table.add_column('Model', style='magenta')
    table.add_column('Status', style='dim')
    table.add_column('Config', style='dim')

    providers = [
        ('portkey', 'Claude Sonnet 4.5', 'PORTKEY_API_KEY, PORTKEY_MODEL'),
        ('ollama', 'Mistral (local)', 'OLLAMA_URL, OLLAMA_MODEL'),
        ('openai', 'GPT-4o-mini', 'OPENAI_API_KEY, OPENAI_MODEL'),
        ('anthropic', 'Claude Haiku', 'ANTHROPIC_API_KEY, ANTHROPIC_MODEL'),
    ]

    for name, model, config in providers:
        status = '[green]Available[/green]' if name in available else '[red]Not configured[/red]'
        table.add_row(name, model, status, config)

    console.print(table)

    # Show workflow types
    wf_table = Table(title='Workflow Optimization', show_header=True)
    wf_table.add_column('Type', style='cyan')
    wf_table.add_column('Description', style='dim')
    wf_table.add_column('Model Selection', style='magenta')

    workflows = [
        ('simple', 'Basic file ops, status checks', 'Fast model (Haiku)'),
        ('standard', 'General shell commands', 'Default model (Sonnet)'),
        ('complex', 'Analysis, debugging, planning', 'Powerful model (Sonnet)'),
        ('creative', 'Writing, generation tasks', 'Creative mode (higher temp)'),
    ]

    for wf_type, desc, model_sel in workflows:
        wf_table.add_row(wf_type, desc, model_sel)

    console.print(wf_table)
    console.print('\n[dim]Set environment variables to enable cloud providers.[/dim]')
    console.print('[dim]Priority: portkey → ollama → openai → anthropic (fallback chain)[/dim]\n')


def show_file_commands():
    """Display available file manager commands."""
    table = Table(title='File Manager Commands', show_header=True)
    table.add_column('Command', style='cyan')
    table.add_column('Description', style='dim')

    commands = [
        ('mkdir <path>', 'Create directory'),
        ('ls [path] [pattern]', 'List directory contents'),
        ('rm <path>', 'Delete file'),
        ('rmdir <path>', 'Delete directory'),
        ('cp <src> <dst>', 'Copy file or directory'),
        ('mv <src> <dst>', 'Move file or directory'),
        ('search <path> <pattern>', 'Search for files'),
        ('backup <path>', 'Create timestamped backup'),
        ('organize <path>', 'Organize files by extension'),
        ('disk <path>', 'Show disk usage'),
        ('dupes <path>', 'Find duplicate files'),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)
    console.print()


def handle_file_command(cmd_parts: list) -> bool:
    """Handle built-in file manager commands. Returns True if handled."""
    if not cmd_parts:
        return False

    cmd = cmd_parts[0].lower()
    args = cmd_parts[1:]

    if cmd == 'mkdir' and args:
        result = file_manager.create_directory(args[0])
        console.print(f'[{"green" if result.success else "red"}]{result.message}[/]')
        return True

    elif cmd == 'ls':
        path = args[0] if args else '.'
        pattern = args[1] if len(args) > 1 else '*'
        result = file_manager.list_directory(path, pattern, show_hidden='-a' in args)
        if result.success and result.details:
            for entry in result.details['entries'][:20]:  # Limit display
                icon = '📁' if entry['is_dir'] else '📄'
                console.print(f"  {icon} {entry['name']}")
            if result.details['count'] > 20:
                console.print(f'  ... and {result.details["count"] - 20} more')
        else:
            console.print(f'[red]{result.message}[/]')
        return True

    elif cmd == 'rm' and args:
        result = file_manager.delete_file(args[0])
        console.print(f'[{"green" if result.success else "red"}]{result.message}[/]')
        return True

    elif cmd == 'rmdir' and args:
        recursive = '-r' in args
        path = [a for a in args if not a.startswith('-')][0]
        result = file_manager.delete_directory(path, recursive=recursive)
        console.print(f'[{"green" if result.success else "red"}]{result.message}[/]')
        return True

    elif cmd == 'cp' and len(args) >= 2:
        result = file_manager.copy(args[0], args[1])
        console.print(f'[{"green" if result.success else "red"}]{result.message}[/]')
        return True

    elif cmd == 'mv' and len(args) >= 2:
        result = file_manager.move(args[0], args[1])
        console.print(f'[{"green" if result.success else "red"}]{result.message}[/]')
        return True

    elif cmd == 'search' and len(args) >= 2:
        result = file_manager.search(args[0], args[1])
        if result.success and result.details:
            for match in result.details['matches'][:20]:
                console.print(f"  {match['path']}")
            if result.details.get('truncated'):
                console.print('  ... results truncated')
        else:
            console.print(f'[red]{result.message}[/]')
        return True

    elif cmd == 'backup' and args:
        result = file_manager.backup(args[0])
        console.print(f'[{"green" if result.success else "red"}]{result.message}[/]')
        return True

    elif cmd == 'organize' and args:
        move = '--move' in args
        path = [a for a in args if not a.startswith('-')][0]
        result = file_manager.organize_by_extension(path, move_files=move)
        console.print(f'[{"green" if result.success else "red"}]{result.message}[/]')
        return True

    elif cmd == 'disk':
        path = args[0] if args else '~'
        result = file_manager.get_disk_usage(path)
        console.print(f'[{"green" if result.success else "red"}]{result.message}[/]')
        return True

    elif cmd == 'dupes' and args:
        result = file_manager.find_duplicates(args[0])
        if result.success and result.details:
            dupes = result.details.get('duplicates', {})
            for key, paths in list(dupes.items())[:10]:
                console.print(f'  [yellow]Duplicates:[/]')
                for p in paths:
                    console.print(f'    {p}')
        else:
            console.print(f'[red]{result.message}[/]')
        return True

    return False


def handle_workflow_command(args: list) -> bool:
    """Handle workflow commands. Returns True if handled."""
    if not args:
        console.print('[yellow]Usage: workflow <name> [args...][/]')
        console.print('Available workflows: ' + ', '.join(WORKFLOWS.keys()))
        return True

    wf_name = args[0]
    wf_args = args[1:]

    if wf_name not in WORKFLOWS:
        console.print(f'[red]Unknown workflow: {wf_name}[/]')
        return True

    try:
        workflow = WORKFLOWS[wf_name](wf_args)
        result = workflow.execute()
        status = '[green]Success[/]' if result.success else '[red]Failed[/]'
        console.print(f'\nWorkflow {status}: {result.message}')
        console.print(f'Steps completed: {result.steps_completed}/{result.steps_total}')
    except Exception as e:
        console.print(f'[red]Workflow error: {e}[/]')

    return True

def run(dry_run: bool = False):
    """Main REPL loop."""
    console.print(Panel('Private AI Assistant - Local Mode', style='bold cyan'))
    console.print('Commands: [cyan]tasks[/cyan], [cyan]files[/cyan], [cyan]workflow[/cyan], [cyan]providers[/cyan], [cyan]reload[/cyan], [cyan]quit[/cyan]')
    console.print('Or type a natural language request.\n')

    while True:
        try:
            user_input = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            console.print('\nGoodbye.')
            break

        if not user_input:
            continue

        parts = user_input.split()
        cmd_lower = parts[0].lower() if parts else ''

        # Built-in commands
        if cmd_lower in ('quit', 'exit', 'q'):
            console.print('Goodbye.')
            break

        if cmd_lower == 'tasks':
            show_tasks()
            continue

        if cmd_lower == 'files':
            show_file_commands()
            continue

        if cmd_lower == 'providers':
            show_providers()
            continue

        if cmd_lower == 'reload':
            from modules.permission_guard import reload_config
            reload_config()
            console.print('[green]Config reloaded.[/green]\n')
            continue

        if cmd_lower == 'workflow':
            handle_workflow_command(parts[1:])
            continue

        # Check for direct file commands (mkdir, ls, cp, mv, etc.)
        if handle_file_command(parts):
            console.print()
            continue

        # Check for task shortcut
        task_cmd = get_task_command(user_input)
        if task_cmd:
            console.print(f'[dim]Task shortcut: {task_cmd}[/dim]')
            result = execute(task_cmd, llm_tier='auto', dry_run=dry_run)
            display_result(result)
            continue

        # Query LLM
        console.print('[dim]Querying LLM...[/dim]')
        llm_result = query(user_input)

        action = llm_result.get('action', '')
        explanation = llm_result.get('explanation', '')
        llm_tier = llm_result.get('tier', 'confirm')
        provider = llm_result.get('provider_used', 'unknown')
        fixes = llm_result.get('fixes_applied', [])

        if not action:
            console.print(f'[yellow]No action generated.[/yellow]')
            if explanation:
                console.print(f'[dim]{explanation}[/dim]')
            console.print()
            continue

        # Display LLM response
        tier_style = TIER_STYLES.get(llm_tier, 'white')
        workflow_type = llm_result.get('workflow_type', 'standard')
        model_used = llm_result.get('model_used', '')

        console.print(f'\n[bold]Action:[/bold] {action}')
        console.print(f'[bold]Explanation:[/bold] {explanation}')
        console.print(f'[bold]LLM Tier:[/bold] [{tier_style}]{llm_tier}[/{tier_style}]')
        console.print(f'[dim]Provider: {provider} | Workflow: {workflow_type}[/dim]')
        if model_used:
            console.print(f'[dim]Model: {model_used}[/dim]')
        if fixes:
            console.print(f'[green]Auto-fixes applied:[/green]')
            for fix in fixes:
                console.print(f'  [green]✓[/green] {fix}')

        # Execute with permission guard
        result = execute(action, llm_tier=llm_tier, dry_run=dry_run)
        display_result(result)

def display_result(result: dict):
    """Display execution result with appropriate styling."""
    status = result.get('status', 'unknown')
    tier = result.get('tier', 'unknown')
    tier_style = TIER_STYLES.get(tier, 'white')

    # Status styling
    status_styles = {
        'ok': 'green',
        'error': 'red',
        'blocked': 'red bold',
        'denied': 'yellow',
        'timeout': 'red',
        'dry_run': 'blue'
    }
    status_style = status_styles.get(status, 'white')

    console.print(f'[bold]Guard Tier:[/bold] [{tier_style}]{tier}[/{tier_style}]')
    console.print(f'[bold]Status:[/bold] [{status_style}]{status}[/{status_style}]')

    if result.get('message'):
        console.print(f'[yellow]{result["message"]}[/yellow]')

    if result.get('stdout'):
        console.print(result['stdout'].rstrip())

    if result.get('stderr') and status == 'error':
        console.print(f'[red]{result["stderr"].rstrip()}[/red]')

    console.print()

def main():
    parser = argparse.ArgumentParser(description='Private AI Assistant')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be executed without running')
    args = parser.parse_args()

    run(dry_run=args.dry_run)

if __name__ == '__main__':
    main()
