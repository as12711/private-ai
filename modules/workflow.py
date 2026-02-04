"""
Workflow Module - Multi-step operation handling with permission management.

Allows complex tasks to be broken into steps, each with proper permission
checking, rollback support, and progress tracking.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

from .permission_guard import classify_command, execute, log_action, validate_command
from .file_manager import FileManager, OperationResult


class StepType(Enum):
    """Types of workflow steps."""
    SHELL = 'shell'           # Shell command
    FILE_OP = 'file_op'       # FileManager operation
    PYTHON = 'python'         # Python callable
    WORKFLOW = 'workflow'     # Nested workflow


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    ROLLED_BACK = 'rolled_back'


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    step_type: StepType
    action: Union[str, Dict[str, Any], Callable]
    description: str = ''
    tier_override: Optional[str] = None  # Force a specific permission tier
    rollback_action: Optional[Union[str, Dict[str, Any]]] = None
    continue_on_fail: bool = False  # Continue workflow even if this step fails
    condition: Optional[Callable[[], bool]] = None  # Only run if condition returns True

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'step_type': self.step_type.value,
            'action': str(self.action) if callable(self.action) else self.action,
            'description': self.description,
            'status': self.status.value,
            'result': self.result,
            'started_at': self.started_at,
            'completed_at': self.completed_at
        }


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    success: bool
    workflow_name: str
    steps_completed: int
    steps_total: int
    steps: List[Dict[str, Any]]
    message: str
    started_at: str
    completed_at: str
    rolled_back: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Workflow:
    """
    A multi-step workflow with permission checking and rollback support.

    Example usage:
        workflow = Workflow('setup_project')
        workflow.add_shell_step('create_dir', 'mkdir ~/projects/new_app',
                               description='Create project directory')
        workflow.add_file_step('create_readme', 'create_file',
                              {'path': '~/projects/new_app/README.md', 'content': '# New App'})
        result = workflow.execute()
    """

    def __init__(self, name: str, description: str = '', dry_run: bool = False):
        self.name = name
        self.description = description
        self.dry_run = dry_run
        self.steps: List[WorkflowStep] = []
        self.file_manager = FileManager()
        self._approval_cache: Dict[str, bool] = {}

    def add_shell_step(self, name: str, command: str, description: str = '',
                       tier_override: Optional[str] = None,
                       rollback_command: Optional[str] = None,
                       continue_on_fail: bool = False) -> 'Workflow':
        """Add a shell command step."""
        step = WorkflowStep(
            name=name,
            step_type=StepType.SHELL,
            action=command,
            description=description,
            tier_override=tier_override,
            rollback_action=rollback_command,
            continue_on_fail=continue_on_fail
        )
        self.steps.append(step)
        return self

    def add_file_step(self, name: str, operation: str, params: Dict[str, Any],
                      description: str = '', rollback_params: Optional[Dict[str, Any]] = None,
                      continue_on_fail: bool = False) -> 'Workflow':
        """
        Add a file operation step.

        Args:
            operation: FileManager method name (e.g., 'create_directory', 'copy', 'move')
            params: Parameters to pass to the operation
        """
        step = WorkflowStep(
            name=name,
            step_type=StepType.FILE_OP,
            action={'operation': operation, 'params': params},
            description=description,
            rollback_action={'operation': operation, 'params': rollback_params} if rollback_params else None,
            continue_on_fail=continue_on_fail
        )
        self.steps.append(step)
        return self

    def add_python_step(self, name: str, func: Callable, description: str = '',
                        rollback_func: Optional[Callable] = None,
                        continue_on_fail: bool = False) -> 'Workflow':
        """Add a Python callable step."""
        step = WorkflowStep(
            name=name,
            step_type=StepType.PYTHON,
            action=func,
            description=description,
            rollback_action=rollback_func,
            continue_on_fail=continue_on_fail
        )
        self.steps.append(step)
        return self

    def add_conditional_step(self, name: str, condition: Callable[[], bool],
                             step_type: StepType, action: Any, **kwargs) -> 'Workflow':
        """Add a step that only executes if condition returns True."""
        step = WorkflowStep(
            name=name,
            step_type=step_type,
            action=action,
            condition=condition,
            **kwargs
        )
        self.steps.append(step)
        return self

    def _get_workflow_tier(self) -> str:
        """
        Determine overall workflow permission tier.

        Returns the most restrictive tier among all steps.
        """
        tier_priority = {'blocked': 0, 'confirm': 1, 'standard': 2, 'auto': 3}
        overall_tier = 'auto'

        for step in self.steps:
            if step.step_type == StepType.SHELL:
                if step.tier_override:
                    step_tier = step.tier_override
                else:
                    step_tier = classify_command(step.action)
            elif step.step_type == StepType.FILE_OP:
                # File operations are generally 'standard' tier
                step_tier = 'standard'
            else:
                # Python steps default to 'standard'
                step_tier = 'standard'

            if tier_priority.get(step_tier, 1) < tier_priority.get(overall_tier, 3):
                overall_tier = step_tier

        return overall_tier

    def preview(self) -> Dict[str, Any]:
        """
        Preview the workflow without executing.

        Returns a summary of all steps and their permission requirements.
        """
        preview_data = {
            'name': self.name,
            'description': self.description,
            'overall_tier': self._get_workflow_tier(),
            'steps': []
        }

        for i, step in enumerate(self.steps):
            step_preview = {
                'index': i,
                'name': step.name,
                'type': step.step_type.value,
                'description': step.description,
                'has_rollback': step.rollback_action is not None,
                'continue_on_fail': step.continue_on_fail,
                'conditional': step.condition is not None
            }

            if step.step_type == StepType.SHELL:
                step_preview['command'] = step.action
                step_preview['tier'] = step.tier_override or classify_command(step.action)
            elif step.step_type == StepType.FILE_OP:
                step_preview['operation'] = step.action['operation']
                step_preview['params'] = step.action['params']
                step_preview['tier'] = 'standard'
            elif step.step_type == StepType.PYTHON:
                step_preview['function'] = step.action.__name__ if hasattr(step.action, '__name__') else str(step.action)
                step_preview['tier'] = 'standard'

            preview_data['steps'].append(step_preview)

        return preview_data

    def _request_approval(self, message: str, tier: str) -> bool:
        """Request user approval for an operation."""
        if self.dry_run:
            return True

        # Check approval cache
        cache_key = f'{tier}:{message}'
        if cache_key in self._approval_cache:
            return self._approval_cache[cache_key]

        if tier == 'auto':
            return True

        if tier == 'blocked':
            print(f'\n[BLOCKED] Operation not allowed: {message}')
            return False

        # For confirm and standard tiers
        prompt = f'\n[{tier.upper()}] {message}\nApprove? (yes/no/all): '
        response = input(prompt).strip().lower()

        if response == 'all':
            # Approve all remaining operations of this tier
            self._approval_cache[cache_key] = True
            return True
        elif response == 'yes':
            return True
        else:
            return False

    def _execute_shell_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a shell command step."""
        # Validate command first
        is_valid, error_msg = validate_command(step.action)
        if not is_valid:
            return {
                'success': False,
                'status': 'error',
                'message': f'Validation failed: {error_msg}'
            }

        tier = step.tier_override or classify_command(step.action)

        if tier == 'blocked':
            return {
                'success': False,
                'status': 'blocked',
                'message': f'Command blocked: {step.action}'
            }

        # Request approval for non-auto tiers
        if tier in ('confirm', 'standard'):
            if not self._request_approval(f'Execute: {step.action}', tier):
                return {
                    'success': False,
                    'status': 'denied',
                    'message': 'User denied execution'
                }

        # Execute the command
        result = execute(step.action, llm_tier=tier, dry_run=self.dry_run)
        return {
            'success': result['status'] == 'ok' or result['status'] == 'dry_run',
            'status': result['status'],
            'stdout': result.get('stdout', ''),
            'stderr': result.get('stderr', ''),
            'message': result.get('message', '')
        }

    def _execute_file_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a file operation step."""
        action = step.action
        operation = action['operation']
        params = action['params']

        # Get the FileManager method
        method = getattr(self.file_manager, operation, None)
        if not method:
            return {
                'success': False,
                'status': 'error',
                'message': f'Unknown file operation: {operation}'
            }

        # Request approval
        if not self._request_approval(
            f'File operation: {operation}({params})',
            'standard'
        ):
            return {
                'success': False,
                'status': 'denied',
                'message': 'User denied execution'
            }

        if self.dry_run:
            return {
                'success': True,
                'status': 'dry_run',
                'message': f'Would execute: {operation}({params})'
            }

        # Execute the operation
        try:
            result: OperationResult = method(**params)
            return {
                'success': result.success,
                'status': 'ok' if result.success else 'error',
                'message': result.message,
                'path': result.path,
                'details': result.details
            }
        except Exception as e:
            return {
                'success': False,
                'status': 'error',
                'message': str(e)
            }

    def _execute_python_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a Python callable step."""
        if not self._request_approval(
            f'Execute Python: {step.name}',
            'standard'
        ):
            return {
                'success': False,
                'status': 'denied',
                'message': 'User denied execution'
            }

        if self.dry_run:
            return {
                'success': True,
                'status': 'dry_run',
                'message': f'Would execute: {step.name}'
            }

        try:
            result = step.action()
            return {
                'success': True,
                'status': 'ok',
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'status': 'error',
                'message': str(e)
            }

    def _rollback_step(self, step: WorkflowStep) -> bool:
        """Attempt to rollback a step."""
        if not step.rollback_action:
            return False

        try:
            if step.step_type == StepType.SHELL:
                result = execute(step.rollback_action, dry_run=self.dry_run)
                return result['status'] == 'ok'
            elif step.step_type == StepType.FILE_OP:
                action = step.rollback_action
                method = getattr(self.file_manager, action['operation'], None)
                if method:
                    result = method(**action['params'])
                    return result.success
            elif step.step_type == StepType.PYTHON and callable(step.rollback_action):
                step.rollback_action()
                return True
        except Exception:
            pass

        return False

    def execute(self, auto_approve_all: bool = False) -> WorkflowResult:
        """
        Execute the workflow.

        Args:
            auto_approve_all: Skip all approval prompts (use with caution)
        """
        started_at = datetime.now().isoformat()
        completed_steps: List[WorkflowStep] = []

        # Show preview first
        preview = self.preview()
        print(f'\n=== Workflow: {self.name} ===')
        print(f'Description: {self.description}')
        print(f'Steps: {len(self.steps)}')
        print(f'Overall tier: {preview["overall_tier"]}')

        if preview['overall_tier'] == 'blocked':
            return WorkflowResult(
                success=False,
                workflow_name=self.name,
                steps_completed=0,
                steps_total=len(self.steps),
                steps=[s.to_dict() for s in self.steps],
                message='Workflow contains blocked operations',
                started_at=started_at,
                completed_at=datetime.now().isoformat()
            )

        # Request overall approval for confirm-tier workflows
        if preview['overall_tier'] == 'confirm' and not auto_approve_all:
            print('\nThis workflow requires confirmation.')
            for i, step_info in enumerate(preview['steps']):
                print(f'  {i + 1}. [{step_info["tier"]}] {step_info["name"]}: {step_info.get("description", step_info.get("command", ""))}')

            response = input('\nProceed with workflow? (yes/no): ').strip().lower()
            if response != 'yes':
                return WorkflowResult(
                    success=False,
                    workflow_name=self.name,
                    steps_completed=0,
                    steps_total=len(self.steps),
                    steps=[s.to_dict() for s in self.steps],
                    message='User cancelled workflow',
                    started_at=started_at,
                    completed_at=datetime.now().isoformat()
                )

        if auto_approve_all:
            self._approval_cache = {f'standard:': True, f'confirm:': True}

        # Execute steps
        for i, step in enumerate(self.steps):
            print(f'\n[Step {i + 1}/{len(self.steps)}] {step.name}')

            # Check condition
            if step.condition and not step.condition():
                print(f'  Skipped (condition not met)')
                step.status = StepStatus.SKIPPED
                continue

            step.status = StepStatus.RUNNING
            step.started_at = datetime.now().isoformat()

            # Execute based on type
            if step.step_type == StepType.SHELL:
                result = self._execute_shell_step(step)
            elif step.step_type == StepType.FILE_OP:
                result = self._execute_file_step(step)
            elif step.step_type == StepType.PYTHON:
                result = self._execute_python_step(step)
            else:
                result = {'success': False, 'message': f'Unknown step type: {step.step_type}'}

            step.result = result
            step.completed_at = datetime.now().isoformat()

            if result['success']:
                step.status = StepStatus.COMPLETED
                completed_steps.append(step)
                print(f'  ✓ {result.get("message", "Completed")}')
            else:
                step.status = StepStatus.FAILED
                print(f'  ✗ {result.get("message", "Failed")}')

                if not step.continue_on_fail:
                    # Rollback completed steps
                    print('\nRolling back...')
                    rolled_back = False
                    for completed_step in reversed(completed_steps):
                        if self._rollback_step(completed_step):
                            completed_step.status = StepStatus.ROLLED_BACK
                            print(f'  Rolled back: {completed_step.name}')
                            rolled_back = True

                    return WorkflowResult(
                        success=False,
                        workflow_name=self.name,
                        steps_completed=len(completed_steps),
                        steps_total=len(self.steps),
                        steps=[s.to_dict() for s in self.steps],
                        message=f'Failed at step {i + 1}: {step.name}',
                        started_at=started_at,
                        completed_at=datetime.now().isoformat(),
                        rolled_back=rolled_back
                    )

        return WorkflowResult(
            success=True,
            workflow_name=self.name,
            steps_completed=len(completed_steps),
            steps_total=len(self.steps),
            steps=[s.to_dict() for s in self.steps],
            message='Workflow completed successfully',
            started_at=started_at,
            completed_at=datetime.now().isoformat()
        )


# === Pre-built Workflows ===

def create_project_workflow(project_name: str, base_path: str = '~/projects') -> Workflow:
    """Create a new project directory structure."""
    project_path = f'{base_path}/{project_name}'

    workflow = Workflow(
        name=f'create_project_{project_name}',
        description=f'Create project structure for {project_name}'
    )

    workflow.add_file_step(
        'create_root',
        'create_directory',
        {'path': project_path},
        description='Create project root directory',
        rollback_params={'path': project_path, 'recursive': True}
    )

    for subdir in ['src', 'tests', 'docs', 'config']:
        workflow.add_file_step(
            f'create_{subdir}',
            'create_directory',
            {'path': f'{project_path}/{subdir}'},
            description=f'Create {subdir} directory'
        )

    workflow.add_file_step(
        'create_readme',
        'create_file',
        {'path': f'{project_path}/README.md', 'content': f'# {project_name}\n\nProject description here.\n'},
        description='Create README.md'
    )

    return workflow


def clear_cache_workflow(include_pip: bool = True, include_npm: bool = False) -> Workflow:
    """Create a cache clearing workflow."""
    workflow = Workflow(
        name='clear_caches',
        description='Clear various system caches'
    )

    # Pacman cache (Arch Linux)
    workflow.add_shell_step(
        'pacman_cache_info',
        'du -sh /var/cache/pacman/pkg/',
        description='Check pacman cache size',
        tier_override='auto'
    )

    workflow.add_shell_step(
        'pacman_clear',
        'sudo pacman -Sc --noconfirm',
        description='Clear old pacman cache',
        tier_override='confirm',
        continue_on_fail=True
    )

    if include_pip:
        workflow.add_shell_step(
            'pip_cache',
            'pip cache purge',
            description='Clear pip cache',
            tier_override='standard',
            continue_on_fail=True
        )

    if include_npm:
        workflow.add_shell_step(
            'npm_cache',
            'npm cache clean --force',
            description='Clear npm cache',
            tier_override='standard',
            continue_on_fail=True
        )

    # Thumbnail cache
    workflow.add_file_step(
        'thumbnail_cache',
        'delete_directory',
        {'path': '~/.cache/thumbnails', 'recursive': True},
        description='Clear thumbnail cache',
        continue_on_fail=True
    )

    return workflow


def backup_config_workflow(backup_dir: str = '~/backups/config') -> Workflow:
    """Create a configuration backup workflow."""
    workflow = Workflow(
        name='backup_configs',
        description='Backup important configuration files'
    )

    config_files = [
        '~/.bashrc',
        '~/.zshrc',
        '~/.config/nvim',
        '~/.gitconfig',
    ]

    workflow.add_file_step(
        'create_backup_dir',
        'create_directory',
        {'path': backup_dir},
        description='Create backup directory'
    )

    for config in config_files:
        name = Path(config).name.replace('.', '_')
        workflow.add_file_step(
            f'backup_{name}',
            'backup',
            {'source': config, 'backup_dir': backup_dir, 'timestamp': True},
            description=f'Backup {config}',
            continue_on_fail=True  # Continue even if some configs don't exist
        )

    return workflow
