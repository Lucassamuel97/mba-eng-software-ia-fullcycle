from rich.console import Console
from rich.text import Text

def print_llm_result(prompt, response):
    console = Console()
    
    # 1. Print do Prompt e Resposta (mantido seu padrão)
    console.print(Text("USER PROMPT:", style="bold green"))
    console.print(Text(prompt, style="bold blue"), end="\n\n")
    
    console.print(Text("LLM RESPONSE:", style="bold green"))
    console.print(Text(response.content, style="bold blue"), end="\n\n")
    
    # 2. Captura de Tokens via usage_metadata (Padrão LangChain atual)
    # O Gemini no LangChain popula esse dicionário automaticamente
    usage = getattr(response, "usage_metadata", None)
    
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        console.print(f"[bold white]Input tokens:[/bold white] [bright_black]{input_tokens}[/bright_black]")
        console.print(f"[bold white]Output tokens:[/bold white] [bright_black]{output_tokens}[/bright_black]")
        console.print(f"[bold white]Total tokens:[/bold white] [bright_black]{total_tokens}[/bright_black]")
    else:
        # Fallback para versões específicas onde pode estar apenas no response_metadata
        usage_fallback = response.response_metadata.get("usage_metadata") or response.response_metadata.get("token_usage")
        if usage_fallback:
            # Aqui as chaves podem variar dependendo da versão (ex: prompt_token_count)
            i = usage_fallback.get("input_tokens") or usage_fallback.get("prompt_token_count", 0)
            o = usage_fallback.get("output_tokens") or usage_fallback.get("candidates_token_count", 0)
            t = usage_fallback.get("total_tokens") or (i + o)
            console.print(f"[bold white]Input tokens:[/bold white] [bright_black]{i}[/bright_black]")
            console.print(f"[bold white]Output tokens:[/bold white] [bright_black]{o}[/bright_black]")
            console.print(f"[bold white]Total tokens:[/bold white] [bright_black]{t}[/bright_black]")
        else:
            console.print("[yellow]Token usage indisponível para este provedor.[/yellow]")
            
    console.print(f"[yellow]{'-'*50} [/yellow]")