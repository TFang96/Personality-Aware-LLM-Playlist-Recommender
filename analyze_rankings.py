#!/usr/bin/env python3
"""
Analysis script for comparing ranking results across different LLMs.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse


def analyze_rankings(results_dir: str):
    """Analyze and compare ranking results across LLMs."""
    
    base_path = Path(results_dir)
    
    if not base_path.exists():
        print(f"Error: {results_dir} not found")
        return
    
    # Find all LLM directories
    llm_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(llm_dirs)} LLM result directories\n")
    
    comparison_data = {}
    
    for llm_dir in sorted(llm_dirs):
        llm_name = llm_dir.name
        
        # Find the latest timestamp directory
        timestamp_dirs = sorted([d for d in llm_dir.iterdir() if d.is_dir()])
        
        if not timestamp_dirs:
            continue
        
        latest_run = timestamp_dirs[-1]
        
        # Count CSV files
        csv_files = list(latest_run.glob("ranked_*.csv"))
        
        print(f"LLM: {llm_name}")
        print(f"  Latest run: {latest_run.name}")
        print(f"  CSV files: {len(csv_files)}")
        
        # Analyze first playlist as example
        if csv_files:
            first_csv = sorted(csv_files)[0]
            df = pd.read_csv(first_csv)
            print(f"  Sample (first playlist):")
            print(f"    - Columns: {', '.join(df.columns)}")
            print(f"    - Rows: {len(df)}")
            if 'score' in df.columns:
                print(f"    - Score range: {df['score'].min():.2f} - {df['score'].max():.2f}")
        
        comparison_data[llm_name] = {
            'csv_count': len(csv_files),
            'latest_run': latest_run.name,
            'path': str(latest_run)
        }
        print()
    
    return comparison_data


def compare_playlist_rankings(results_dir: str, playlist_id: str):
    """Compare rankings for a specific playlist across LLMs."""
    
    base_path = Path(results_dir)
    
    print(f"\nComparing rankings for playlist {playlist_id}:")
    print(f"{'='*60}\n")
    
    all_rankings = {}
    
    # Find rankings for this playlist across all LLMs
    for llm_dir in sorted(base_path.iterdir()):
        if not llm_dir.is_dir():
            continue
        
        llm_name = llm_dir.name
        timestamp_dirs = sorted([d for d in llm_dir.iterdir() if d.is_dir()])
        
        if not timestamp_dirs:
            continue
        
        latest_run = timestamp_dirs[-1]
        csv_path = latest_run / f"ranked_{playlist_id}.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_rankings[llm_name] = df
            
            print(f"{llm_name}:")
            print(f"  Top 5 songs:")
            for idx, row in df.head(5).iterrows():
                score = row.get('score', 'N/A')
                print(f"    {idx+1}. {row['song']} - {row['artist']} (score: {score})")
            print()
    
    # Show common top songs
    if len(all_rankings) > 1:
        print("\nConsensus (songs in top-5 of multiple LLMs):")
        song_counts = defaultdict(int)
        for df in all_rankings.values():
            for song, artist in zip(df.head(5)['song'], df.head(5)['artist']):
                song_counts[(song, artist)] += 1
        
        for (song, artist), count in sorted(song_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:
                print(f"  • {song} - {artist}: {count} LLMs")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ranking results across LLMs"
    )
    parser.add_argument(
        '--results-dir', default='./ranking_results',
        help="Results directory from batch_rank_subset22.py"
    )
    parser.add_argument(
        '--playlist', help="Specific playlist ID to compare"
    )
    
    args = parser.parse_args()
    
    comparison = analyze_rankings(args.results_dir)
    
    if args.playlist:
        compare_playlist_rankings(args.results_dir, args.playlist)
    
    print(f"\nTo compare a specific playlist, use:")
    print(f"  python analyze_rankings.py --results-dir {args.results_dir} --playlist <PID>")


if __name__ == "__main__":
    main()
