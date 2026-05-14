from rich.console import Console
from rich.text import Text

def print_llm_result(prompt, response):
    """
    Print LLM prompt, response and token usage with colored formatting
    """
    console = Console()
    
    # Print prompt
    console.print(Text("USER PROMPT:", style="bold green"))
    console.print(Text(prompt, style="bold blue"), end="\n\n")
    
    # Print response
    console.print(Text("LLM RESPONSE:", style="bold green"))
    console.print(Text(response.content, style="bold blue"), end="\n\n")
    
    # Print token usage (quando disponivel na resposta do provedor).
    usage = response.response_metadata.get("token_usage") or response.response_metadata.get("usage_metadata")
    if usage:
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        console.print(f"[bold white]Input tokens:[/bold white] [bright_black]{prompt_tokens}[/bright_black]")
        console.print(f"[bold white]Output tokens:[/bold white] [bright_black]{completion_tokens}[/bright_black]")
        console.print(f"[bold white]Total tokens:[/bold white] [bright_black]{total_tokens}[/bright_black]")
    else:
        console.print("[yellow]Token usage indisponivel para este provedor.[/yellow]")
    console.print(f"[yellow]{'-'*50} [/yellow]")