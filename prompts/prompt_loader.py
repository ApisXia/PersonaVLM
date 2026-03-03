"""
Simple prompt template loader for street crossing decision system.
"""

import os
from typing import Dict, Any


class PromptLoader:
    """Simple loader for prompt templates."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """Initialize with prompts directory path."""
        self.prompts_dir = prompts_dir
        self._templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates from files."""
        template_files = {
            'system': 'simulation/system_prompt.txt',
            'user': 'simulation/user_prompt.txt', 
            'history_format': 'simulation/history_format.txt',
            'questionnaire': 'questionnaire/user_prompt.txt',
            'questionnaire_system': 'questionnaire/system_prompt.txt'
        }
        
        for name, filename in template_files.items():
            filepath = os.path.join(self.prompts_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self._templates[name] = f.read().strip()
            except FileNotFoundError:
                print(f"Warning: Template file {filepath} not found")
                self._templates[name] = ""
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get system prompt with variables filled in."""
        return self._templates['system'].format(**kwargs)
    
    def get_user_prompt(self, **kwargs) -> str:
        """Get user prompt with variables filled in."""
        return self._templates['user'].format(**kwargs)
    
    def get_questionnaire(self, **kwargs) -> str:
        """Get questionnaire user prompt with variables filled in."""
        return self._templates['questionnaire'].format(**kwargs)
    
    def get_questionnaire_system(self, **kwargs) -> str:
        """Get questionnaire system prompt with variables filled in.""" 
        return self._templates['questionnaire_system'].format(**kwargs)
    
    def format_history(self, history) -> str:
        """Format history using template."""
        if not history:
            return "No previous decisions"
        
        history_lines = []
        for decision in history:
            formatted_line = self._templates['history_format'].format(
                time=decision['time'],
                decision=decision['decision'],
                reason=decision['reason']
            )
            history_lines.append(formatted_line)
        
        return chr(10).join(history_lines)


# Global instance for easy importing
prompt_loader = PromptLoader()