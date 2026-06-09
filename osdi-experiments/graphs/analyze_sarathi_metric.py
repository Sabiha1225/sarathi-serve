#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse


## Creating an Analysis Script

# Create `analyze_sarathi_metrics.py`:

# ```python

# Print summary statistics
#python analyze_sarathi_metrics.py sequence_metrics.csv

# With visualizations
#python analyze_sarathi_metrics.py sequence_metrics.csv --visualize

# Export to JSON
#python analyze_sarathi_metrics.py sequence_metrics.csv --json

# All together
#python analyze_sarathi_metrics.py sequence_metrics.csv --visualize --json

#python analyze_sarathi_metrics.py sequence_metrics.csv --total-time 625.5

#python analyze_sarathi_metrics.py sequence_metrics.csv --total-time 625.5 --visualize --json

sns.set_style("whitegrid")


class SarathiMetricsAnalyzer:
    """Analyze Sarathi metrics CSV file."""
    
    def __init__(self, csv_file: str, total_time: float = None):
        """Load and prepare metrics.
        
        Args:
            csv_file: Path to sequence_metrics.csv
            total_time: Total time for throughput calculation (in seconds).
                       If None, calculated from data.
        """
        self.df = pd.read_csv(csv_file)
        self.output_dir = Path(csv_file).parent
        self.user_provided_total_time = total_time
        
        # Clean data: remove NaN values for calculations
        self.df = self.df.dropna(subset=['request_e2e_time', 'request_num_decode_tokens'])
        
        print(f"Loaded {len(self.df)} completed requests")
        print(f"CSV columns available: {len(self.df.columns)}")
        if self.user_provided_total_time:
            print(f"✅ Using user-provided total time: {self.user_provided_total_time:.2f}s")
    
    def get_total_time(self) -> float:
        """Get total time - either from user or calculated from data."""
        if self.user_provided_total_time is not None:
            return self.user_provided_total_time
        
        # Calculate from data: last completion - first arrival
        first_arrival = self.df['request_scheduling_delay'].min()
        last_completion = self.df['request_e2e_time'].max()
        calculated_time = last_completion + first_arrival
        
        return calculated_time
    
    def calculate_metrics(self) -> dict:
        """Calculate all required metrics."""
        
        metrics = {}
        total_time = self.get_total_time()
        
        # ===== 1. TPOT (Time Per Output Token) =====
        tpot_values = []
        for idx, row in self.df.iterrows():
            output_tokens = row['request_num_decode_tokens']
            
            if pd.notna(row['prefill_e2e_time']) and output_tokens > 1:
                decode_time = row['request_e2e_time'] - row['prefill_e2e_time']
                tpot = decode_time / (output_tokens - 1) if output_tokens > 1 else None
                if tpot and tpot > 0:
                    tpot_values.append(tpot)
        
        if tpot_values:
            metrics['tpot'] = {
                'min': min(tpot_values),
                'max': max(tpot_values),
                'mean': np.mean(tpot_values),
                'median': np.median(tpot_values),
                'p50': np.percentile(tpot_values, 50),
                'p95': np.percentile(tpot_values, 95),
                'p99': np.percentile(tpot_values, 99),
            }
        
        # ===== 2. TTFT (Time To First Token) =====
        ttft_values = self.df['prefill_e2e_time'].dropna()
        if len(ttft_values) > 0:
            metrics['ttft'] = {
                'min': ttft_values.min(),
                'max': ttft_values.max(),
                'mean': ttft_values.mean(),
                'median': ttft_values.median(),
                'p50': ttft_values.quantile(0.50),
                'p95': ttft_values.quantile(0.95),
                'p99': ttft_values.quantile(0.99),
            }
        
        # ===== 3. ITL (Internal Token Latency) =====
        if 'tpot' in metrics:
            metrics['itl'] = metrics['tpot']
        
        # ===== 4. Request Latency (E2E) =====
        latency_values = self.df['request_e2e_time'].dropna()
        if len(latency_values) > 0:
            metrics['request_latency'] = {
                'min': latency_values.min(),
                'max': latency_values.max(),
                'mean': latency_values.mean(),
                'median': latency_values.median(),
                'p50': latency_values.quantile(0.50),
                'p95': latency_values.quantile(0.95),
                'p99': latency_values.quantile(0.99),
            }
        
        # ===== 5. Throughput Metrics =====
        total_requests = len(self.df)
        total_tokens = self.df['request_num_tokens'].sum()
        total_output_tokens = self.df['request_num_decode_tokens'].sum()
        
        metrics['throughput'] = {
            'total_time_s': total_time,
            'total_requests': total_requests,
            'total_tokens': int(total_tokens),
            'total_output_tokens': int(total_output_tokens),
            'request_throughput_req_per_s': total_requests / total_time if total_time > 0 else 0,
            'token_throughput_tokens_per_s': total_tokens / total_time if total_time > 0 else 0,
            'output_token_throughput_tokens_per_s': total_output_tokens / total_time if total_time > 0 else 0,
        }
        
        # ===== 6. Scheduling Metrics =====
        scheduling_delay = self.df['request_scheduling_delay'].dropna()
        if len(scheduling_delay) > 0:
            metrics['scheduling'] = {
                'min': scheduling_delay.min(),
                'max': scheduling_delay.max(),
                'mean': scheduling_delay.mean(),
                'median': scheduling_delay.median(),
                'p50': scheduling_delay.quantile(0.50),
                'p95': scheduling_delay.quantile(0.95),
                'p99': scheduling_delay.quantile(0.99),
            }
        
        # ===== 7. Token Statistics =====
        metrics['token_stats'] = {
            'avg_prompt_tokens': self.df['request_num_prefill_tokens'].mean(),
            'avg_output_tokens': self.df['request_num_decode_tokens'].mean(),
            'avg_total_tokens': self.df['request_num_tokens'].mean(),
            'min_prompt_tokens': int(self.df['request_num_prefill_tokens'].min()),
            'max_prompt_tokens': int(self.df['request_num_prefill_tokens'].max()),
            'min_output_tokens': int(self.df['request_num_decode_tokens'].min()),
            'max_output_tokens': int(self.df['request_num_decode_tokens'].max()),
        }
        
        return metrics
    
    def print_summary(self, metrics: dict):
        """Print comprehensive summary."""
        print("\n" + "="*80)
        print("SARATHI-SERVE COMPREHENSIVE METRICS ANALYSIS")
        print("="*80)
        
        # Throughput
        print("\n📊 THROUGHPUT METRICS")
        print("-" * 80)
        tp = metrics['throughput']
        print(f"  Total Time:                    {tp['total_time_s']:.2f}s")
        print(f"  Total Requests:                {tp['total_requests']}")
        print(f"  Total Tokens:                  {tp['total_tokens']}")
        print(f"  Total Output Tokens:           {tp['total_output_tokens']}")
        print(f"  Request Throughput:            {tp['request_throughput_req_per_s']:.2f} req/s")
        print(f"  Token Throughput:              {tp['token_throughput_tokens_per_s']:.2f} tokens/s")
        print(f"  Output Token Throughput:       {tp['output_token_throughput_tokens_per_s']:.2f} output-tokens/s")
        
        # TTFT
        if 'ttft' in metrics:
            print("\n⏱️  TTFT (Time To First Token)")
            print("-" * 80)
            ttft = metrics['ttft']
            print(f"  Min:       {ttft['min']:.4f}s")
            print(f"  Max:       {ttft['max']:.4f}s")
            print(f"  Mean:      {ttft['mean']:.4f}s")
            print(f"  Median:    {ttft['median']:.4f}s")
            print(f"  P50:       {ttft['p50']:.4f}s")
            print(f"  P95:       {ttft['p95']:.4f}s")
            print(f"  P99:       {ttft['p99']:.4f}s")
        
        # TPOT / ITL
        if 'tpot' in metrics:
            print("\n⏱️  TPOT (Time Per Output Token) / ITL (Internal Token Latency)")
            print("-" * 80)
            tpot = metrics['tpot']
            print(f"  Min:       {tpot['min']:.4f}s")
            print(f"  Max:       {tpot['max']:.4f}s")
            print(f"  Mean:      {tpot['mean']:.4f}s")
            print(f"  Median:    {tpot['median']:.4f}s")
            print(f"  P50:       {tpot['p50']:.4f}s")
            print(f"  P95:       {tpot['p95']:.4f}s")
            print(f"  P99:       {tpot['p99']:.4f}s")
        
        # Request Latency
        if 'request_latency' in metrics:
            print("\n⏱️  REQUEST LATENCY (E2E)")
            print("-" * 80)
            lat = metrics['request_latency']
            print(f"  Min:       {lat['min']:.4f}s")
            print(f"  Max:       {lat['max']:.4f}s")
            print(f"  Mean:      {lat['mean']:.4f}s")
            print(f"  Median:    {lat['median']:.4f}s")
            print(f"  P50:       {lat['p50']:.4f}s")
            print(f"  P95:       {lat['p95']:.4f}s")
            print(f"  P99:       {lat['p99']:.4f}s")
        
        # Scheduling Delay
        if 'scheduling' in metrics:
            print("\n⏱️  SCHEDULING DELAY")
            print("-" * 80)
            sched = metrics['scheduling']
            print(f"  Min:       {sched['min']:.4f}s")
            print(f"  Max:       {sched['max']:.4f}s")
            print(f"  Mean:      {sched['mean']:.4f}s")
            print(f"  Median:    {sched['median']:.4f}s")
        
        # Token Stats
        print("\n📝 TOKEN STATISTICS")
        print("-" * 80)
        tokens = metrics['token_stats']
        print(f"  Avg Prompt Tokens:             {tokens['avg_prompt_tokens']:.1f}")
        print(f"  Avg Output Tokens:             {tokens['avg_output_tokens']:.1f}")
        print(f"  Avg Total Tokens:              {tokens['avg_total_tokens']:.1f}")
        print(f"  Prompt Tokens Range:           {tokens['min_prompt_tokens']} - {tokens['max_prompt_tokens']}")
        print(f"  Output Tokens Range:           {tokens['min_output_tokens']} - {tokens['max_output_tokens']}")
        
        print("\n" + "="*80 + "\n")
    
    def create_visualizations(self):
        """Create comprehensive plots."""
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. TTFT Distribution
        ax1 = plt.subplot(2, 3, 1)
        ttft_data = self.df['prefill_e2e_time'].dropna()
        ax1.hist(ttft_data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(ttft_data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {ttft_data.mean():.4f}s')
        ax1.set_xlabel('TTFT (seconds)')
        ax1.set_ylabel('Count')
        ax1.set_title('TTFT Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Request Latency CDF
        ax2 = plt.subplot(2, 3, 2)
        latency_data = sorted(self.df['request_e2e_time'].dropna())
        cdf = np.arange(1, len(latency_data) + 1) / len(latency_data)
        ax2.plot(latency_data, cdf, linewidth=2, color='darkgreen')
        ax2.fill_between(latency_data, cdf, alpha=0.3, color='lightgreen')
        ax2.set_xlabel('Request Latency (seconds)')
        ax2.set_ylabel('CDF')
        ax2.set_title('Request Latency CDF')
        ax2.grid(alpha=0.3)
        
        # 3. TPOT Distribution
        ax3 = plt.subplot(2, 3, 3)
        tpot_data = []
        for idx, row in self.df.iterrows():
            output_tokens = row['request_num_decode_tokens']
            if pd.notna(row['prefill_e2e_time']) and output_tokens > 1:
                decode_time = row['request_e2e_time'] - row['prefill_e2e_time']
                tpot = decode_time / (output_tokens - 1)
                if tpot > 0:
                    tpot_data.append(tpot)
        
        if tpot_data:
            ax3.hist(tpot_data, bins=50, color='purple', edgecolor='black', alpha=0.7)
            ax3.axvline(np.mean(tpot_data), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(tpot_data):.4f}s')
            ax3.set_xlabel('TPOT (seconds)')
            ax3.set_ylabel('Count')
            ax3.set_title('TPOT Distribution')
            ax3.legend()
            ax3.grid(alpha=0.3)
        
        # 4. Scheduling Delay Distribution
        ax4 = plt.subplot(2, 3, 4)
        sched_data = self.df['request_scheduling_delay'].dropna()
        ax4.hist(sched_data, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax4.axvline(sched_data.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {sched_data.mean():.4f}s')
        ax4.set_xlabel('Scheduling Delay (seconds)')
        ax4.set_ylabel('Count')
        ax4.set_title('Scheduling Delay Distribution')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Request Count
        ax5 = plt.subplot(2, 3, 5)
        total_requests = len(self.df)
        ax5.text(0.5, 0.5, f'Total Requests\n{total_requests}', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        # 6. Prompt vs Output Token Distribution
        ax6 = plt.subplot(2, 3, 6)
        scatter = ax6.scatter(self.df['request_num_prefill_tokens'], 
                             self.df['request_num_decode_tokens'], 
                             alpha=0.5, s=50, c=self.df['request_num_restarts'],
                             cmap='RdYlGn_r')
        ax6.set_xlabel('Prefill Tokens')
        ax6.set_ylabel('Decode Tokens')
        ax6.set_title('Prompt vs Output Tokens')
        ax6.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('# Preemptions')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization to {self.output_dir / 'metrics_analysis.png'}")
        plt.close()
    
    def export_to_json(self, metrics: dict):
        """Export metrics to JSON."""
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        metrics_serializable = convert_to_serializable(metrics)
        
        with open(self.output_dir / 'metrics_summary.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"✅ Saved JSON summary to {self.output_dir / 'metrics_summary.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Sarathi metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_sarathi_metrics.py sequence_metrics.csv
  python analyze_sarathi_metrics.py sequence_metrics.csv --total-time 625.5
  python analyze_sarathi_metrics.py sequence_metrics.csv --total-time 625.5 --visualize
  python analyze_sarathi_metrics.py sequence_metrics.csv --total-time 625.5 --json
  python analyze_sarathi_metrics.py sequence_metrics.csv --total-time 625.5 --visualize --json
        """
    )
    
    parser.add_argument(
        'csv_file',
        help='Path to sequence_metrics.csv file'
    )
    parser.add_argument(
        '--total-time',
        type=float,
        default=None,
        help='Total time for throughput calculation in seconds'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization plots'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Export summary to JSON file'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {args.csv_file}")
        return
    
    print(f"\n📊 Analyzing: {args.csv_file}")
    if args.total_time:
        print(f"📍 Total time (provided): {args.total_time:.2f}s")
    
    analyzer = SarathiMetricsAnalyzer(args.csv_file, total_time=args.total_time)
    metrics = analyzer.calculate_metrics()
    
    analyzer.print_summary(metrics)
    
    if args.visualize:
        print("📈 Creating visualizations...")
        analyzer.create_visualizations()
    
    if args.json:
        print("📄 Exporting to JSON...")
        analyzer.export_to_json(metrics)
    
    print("✅ Analysis complete!")


if __name__ == '__main__':
    main()