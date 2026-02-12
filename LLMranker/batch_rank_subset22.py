"""
Batch ranking script for subset_22 playlists across multiple LLMs.
Runs the ranker on all 22 playlists with different models.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import ranker function
from ranker import rank_playlist, load_prompt

# Available LLMs to test
AVAILABLE_LLMS = {
    "llama3.1": {"description": "Ollama (llama3.1)"},
    "mistral-large": {"description": "Ollama (mistral-large)"},
    "mistral-tiny": {"description": "Ollama (mistral-tiny)"},
    "zephyr-latest": {"description": "Ollama (zephyr:latest)"},
    "gemma-latest": {"description": "Ollama (gemma:latest)"},
    "qwen": {"description": "Ollama (qwen:latest)"},
    "t5": {"description": "Ollama (t5:latest)"},
    "gpt4o": {"description": "OpenAI"},
    "gpt41": {"description": "OpenAI"},
    "gemini": {"description": "Google"},
    "claude-latest": {"description": "Anthropic"},
}

# Map user-friendly names to Ollama model IDs
MODEL_ALIASES = {
    "llama3.1": "llama3.1",
    "mistral-large": "mistral-large",
    "mistral-tiny": "mistral-tiny",
    "zephyr-latest": "zephyr:latest",
    "gemma-latest": "gemma:latest",
    "qwen": "qwen:latest",
    "t5": "t5:latest",
}


# Default LLMs to use
DEFAULT_LLMS = [
    "llama3.1",
    "mistral-large",
    "mistral-tiny",
    "zephyr-latest",
    "gemma-latest",
    "gpt4o",
    "gpt41",
    "gemini",
    "qwen",
    "t5",
]


def load_playlists_yaml(yaml_path: str) -> dict:
    """Load playlists from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['playlists']


def run_ranker(title: str, songs_df, output_csv: str, 
               model: str, prompt_path: str = './LLMranker/prompt_ranker.txt',
               temperature: float = 0.2) -> bool:
    """Call rank_playlist function directly."""
    try:
        prompt_text = load_prompt(prompt_path)
        
        # Call rank_playlist directly
        df_ranked = rank_playlist(
            title=title,
            songs_df=songs_df,
            prompt_text=prompt_text,
            model=model,
            temperature=temperature
        )
        
        # Save to CSV
        df_ranked.to_csv(output_csv, index=False)
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run playlist ranker across multiple LLMs"
    )
    parser.add_argument(
        '--llms', nargs='+', default=DEFAULT_LLMS,
        help=f"LLMs to test. Available: {', '.join(AVAILABLE_LLMS.keys())}"
    )
    parser.add_argument(
        '--yaml', default='./subset_22.yml',
        help="Path to playlists YAML file"
    )
    parser.add_argument(
        '--output-dir', default='./ranking_results',
        help="Base output directory for results"
    )
    parser.add_argument(
        '--temperature', type=float, default=0.2,
        help="LLM temperature parameter"
    )
    parser.add_argument(
        '--list-avail', action='store_true',
        help="List available LLMs and exit"
    )
    
    args = parser.parse_args()
    
    # Show available LLMs
    if args.list_avail:
        print("Available LLMs:")
        for model, info in AVAILABLE_LLMS.items():
            print(f"  - {model}: {info['description']}")
        return 0
    
    # Validate LLMs
    for llm in args.llms:
        if llm not in AVAILABLE_LLMS:
            print(f"Unknown LLM: {llm}")
            print(f"Available: {', '.join(AVAILABLE_LLMS.keys())}")
            return 1
    
    # Load playlists
    if not os.path.exists(args.yaml):
        print(f"Error: {args.yaml} not found")
        return 1
    
    print(f"Loading playlists from {args.yaml}...")
    playlists = load_playlists_yaml(args.yaml)
    print(f"Loaded {len(playlists)} playlists\n")
    
    # Create base output directory
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each LLM
    results_summary = {}
    
    for llm_name in args.llms:
        print(f"\n{'='*60}")
        print(f"Testing LLM: {llm_name}")
        print(f"{'='*60}")
        
        # Create LLM output directory
        llm_output_dir = base_output / llm_name / timestamp
        llm_output_dir.mkdir(parents=True, exist_ok=True)
        
        results_summary[llm_name] = {
            "total": len(playlists),
            "success": 0,
            "failed": 0,
            "failures": []
        }
        
        # Process each playlist
        for idx, playlist in enumerate(playlists, 1):
            pid = playlist['pid']
            title = playlist['playlist_title']
            
            print(f"  [{idx}/{len(playlists)}] {title} (PID: {pid})...", end=' ', flush=True)
            
            try:
                # Get songs for this playlist
                songs = playlist.get('tracks')
                
                if not songs:
                    print(f"✗ No tracks found")
                    results_summary[llm_name]["failed"] += 1
                    results_summary[llm_name]["failures"].append(f"{pid}: no tracks")
                    continue
                
                # Convert songs to DataFrame
                songs_df = pd.DataFrame([{
                    'uri': track['uri'],
                    'song': track['song'],
                    'artist': track['artist']
                } for track in songs])
                
                # Run ranker
                output_csv = llm_output_dir / f"ranked_{pid}.csv"
                model_id = MODEL_ALIASES.get(llm_name, llm_name)
                success = run_ranker(
                    title=title,
                    songs_df=songs_df,
                    output_csv=str(output_csv),
                    model=model_id,
                    temperature=args.temperature
                )
                
                if success:
                    results_summary[llm_name]["success"] += 1
                    print(f"✓ Completed")
                else:
                    results_summary[llm_name]["failed"] += 1
                    results_summary[llm_name]["failures"].append(f"{pid}: ranking failed")
                    
            except Exception as e:
                print(f"✗ Exception: {e}")
                results_summary[llm_name]["failed"] += 1
                results_summary[llm_name]["failures"].append(f"{pid}: {str(e)}")
        
        # Print LLM summary
        success = results_summary[llm_name]["success"]
        failed = results_summary[llm_name]["failed"]
        print(f"\n  Summary: {success}/{len(playlists)} successful")
        if failed > 0 and results_summary[llm_name]["failures"]:
            print(f"  Failed playlists:")
            for failure in results_summary[llm_name]["failures"][:3]:
                print(f"    - {failure}")
            if len(results_summary[llm_name]["failures"]) > 3:
                print(f"    ... and {len(results_summary[llm_name]['failures']) - 3} more")
    
    # Save overall summary
    summary_file = base_output / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for llm_name, stats in results_summary.items():
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{llm_name:20s}: {stats['success']:2d}/{stats['total']} ({success_rate:5.1f}%)")
    
    print(f"\nResults saved to: {base_output}")
    print(f"Summary: {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
