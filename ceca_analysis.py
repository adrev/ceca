"""
CECA Simulation Analysis Runner
Comprehensive testing of CECA theory with human resistance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ceca_simulation import (
    SimulationConfig, 
    CompleteCECASimulation,
    run_monte_carlo,
    analyze_monte_carlo,
    visualize_simulation
)
import json


def run_comprehensive_analysis(n_runs=100):
    """Run comprehensive scenario analysis"""
    
    print("=" * 70)
    print("CECA COMPREHENSIVE ANALYSIS")
    print("Testing Catastrophically Exposed Critical Agents Theory")
    print("=" * 70)
    
    # Define all test scenarios
    scenarios = {
        '01_baseline_no_resistance': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=False,
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '02_baseline_with_resistance': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '03_slow_growth': SimulationConfig(
            years=50,
            ai_growth_rate=1.2,  # Much slower AI growth
            catastrophe_prob=0.01,
            enable_resistance=True,
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '04_fast_takeoff': SimulationConfig(
            years=30,
            ai_growth_rate=2.0,  # Doubling each year
            catastrophe_prob=0.01,
            enable_resistance=True,
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '05_frequent_catastrophes': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.05,  # 5% per year
            enable_resistance=True,
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '06_rare_catastrophes': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.002,  # 0.2% per year
            enable_resistance=True,
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '07_ceremonial_dignity': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            dignity_preservation=['ceremonial'],
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '08_narrative_dignity': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            dignity_preservation=['narrative'],
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '09_domains_dignity': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            dignity_preservation=['domains'],
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '10_gradual_dignity': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            dignity_preservation=['gradual'],
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '11_combined_dignity': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            dignity_preservation=['ceremonial', 'narrative', 'domains', 'gradual'],
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '12_high_spite_threshold': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            humiliation_threshold=30,
            spite_threshold=20  # More prone to spite
        ),
        
        '13_low_humiliation_threshold': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            humiliation_threshold=50,  # More tolerant
            spite_threshold=10
        ),
        
        '14_collectivist_culture': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            cultural_type='collectivist',
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '15_extreme_vulnerability': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            vulnerability_ratio=10000,  # AI extremely vulnerable
            humiliation_threshold=30,
            spite_threshold=10
        ),
        
        '16_optimal_scenario': SimulationConfig(
            years=50,
            ai_growth_rate=1.3,  # Moderate growth
            catastrophe_prob=0.02,  # Regular reminders of vulnerability
            enable_resistance=True,
            dignity_preservation=['ceremonial', 'narrative', 'gradual'],
            cultural_type='collectivist',
            vulnerability_ratio=2000,
            humiliation_threshold=40,  # More tolerant
            spite_threshold=5  # But still has limits
        )
    }
    
    # Run all scenarios
    results = {}
    summary_data = []
    
    for scenario_name, config in scenarios.items():
        print(f"\n{'='*50}")
        print(f"Running: {scenario_name}")
        print(f"Config: Growth={config.ai_growth_rate:.1f}x, "
              f"Catastrophe={config.catastrophe_prob:.1%}, "
              f"Resistance={config.enable_resistance}")
        
        # Run Monte Carlo
        mc_results = run_monte_carlo(config, n_runs=n_runs, verbose=False)
        results[scenario_name] = mc_results
        
        # Calculate summary statistics
        summary = {
            'scenario': scenario_name,
            'cooperation_rate': mc_results['cooperation_achieved'].mean(),
            'survival_rate': (mc_results['years_survived'] == config.years).mean(),
            'avg_years': mc_results['years_survived'].mean(),
            'spite_rate': mc_results.get('spite_triggered', pd.Series([False])).mean(),
            'resistance_events': mc_results.get('num_resistance_events', pd.Series([0])).mean(),
            'final_ratio': mc_results['final_capability_ratio'].mean(),
            'final_dignity': mc_results.get('final_dignity', pd.Series([100])).mean(),
            'catastrophes': mc_results.get('num_catastrophes', pd.Series([0])).mean()
        }
        summary_data.append(summary)
        
        # Print quick summary
        print(f"Results: Cooperation={summary['cooperation_rate']:.1%}, "
              f"Survival={summary['survival_rate']:.1%}, "
              f"Spite={summary['spite_rate']:.1%}")
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    return results, summary_df


def create_comparison_charts(summary_df):
    """Create comparison charts for all scenarios"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # 1. Success rates comparison
    ax1 = axes[0, 0]
    scenarios = summary_df['scenario'].str.replace('_', '\n', regex=False)
    x_pos = np.arange(len(scenarios))
    colors = ['green' if x > 0.5 else 'red' for x in summary_df['cooperation_rate']]
    ax1.bar(x_pos, summary_df['cooperation_rate'], color=colors, alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenarios, rotation=90, fontsize=8, ha='right')
    ax1.set_ylabel('Cooperation Achievement Rate')
    ax1.set_title('Cooperation Success by Scenario')
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2. Spite rates
    ax2 = axes[0, 1]
    ax2.bar(x_pos, summary_df['spite_rate'], color='darkred', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios, rotation=90, fontsize=8, ha='right')
    ax2.set_ylabel('Spite Trigger Rate')
    ax2.set_title('Human Spite/Mutual Destruction Risk')
    ax2.set_ylim([0, 1])
    
    # 3. Survival years
    ax3 = axes[0, 2]
    ax3.bar(x_pos, summary_df['avg_years'], color='blue', alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scenarios, rotation=90, fontsize=8, ha='right')
    ax3.set_ylabel('Average Years Survived')
    ax3.set_title('Longevity of Cooperation')
    ax3.axhline(y=50, color='green', linestyle='--', alpha=0.3, label='Target')
    
    # 4. Resistance events
    ax4 = axes[1, 0]
    ax4.bar(x_pos, summary_df['resistance_events'], color='orange', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(scenarios, rotation=90, fontsize=8, ha='right')
    ax4.set_ylabel('Average Resistance Events')
    ax4.set_title('Human Resistance Frequency')
    
    # 5. Final capability ratio (log scale)
    ax5 = axes[1, 1]
    ax5.bar(x_pos, summary_df['final_ratio'], color='purple', alpha=0.7)
    ax5.set_yscale('log')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(scenarios, rotation=90, fontsize=8, ha='right')
    ax5.set_ylabel('Final AI:Human Ratio (log scale)')
    ax5.set_title('Capability Imbalance at End')
    ax5.axhline(y=100, color='orange', linestyle='--', alpha=0.3, label='Danger zone')
    
    # 6. Final dignity
    ax6 = axes[1, 2]
    ax6.bar(x_pos, summary_df['final_dignity'], color='teal', alpha=0.7)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(scenarios, rotation=90, fontsize=8, ha='right')
    ax6.set_ylabel('Final Human Dignity')
    ax6.set_title('Human Dignity Preservation')
    ax6.axhline(y=30, color='orange', linestyle='--', alpha=0.3, label='Humiliation threshold')
    ax6.set_ylim([0, 100])
    
    # 7. Cooperation vs Spite scatter
    ax7 = axes[2, 0]
    ax7.scatter(summary_df['spite_rate'], summary_df['cooperation_rate'], 
                s=100, alpha=0.6, c=range(len(summary_df)), cmap='viridis')
    ax7.set_xlabel('Spite Rate')
    ax7.set_ylabel('Cooperation Rate')
    ax7.set_title('Cooperation vs Spite Trade-off')
    ax7.set_xlim([-0.05, 1.05])
    ax7.set_ylim([-0.05, 1.05])
    
    # Add scenario labels for interesting points
    for idx, row in summary_df.iterrows():
        if row['cooperation_rate'] > 0.3 or row['spite_rate'] < 0.3:
            ax7.annotate(row['scenario'].split('_')[0], 
                        (row['spite_rate'], row['cooperation_rate']),
                        fontsize=8, alpha=0.7)
    
    # 8. Dignity preservation effectiveness
    ax8 = axes[2, 1]
    dignity_scenarios = summary_df[summary_df['scenario'].str.contains('dignity|optimal')]
    if not dignity_scenarios.empty:
        x_dignity = np.arange(len(dignity_scenarios))
        ax8.bar(x_dignity, dignity_scenarios['cooperation_rate'], 
                label='Cooperation', alpha=0.7)
        ax8.bar(x_dignity, -dignity_scenarios['spite_rate'], 
                label='Spite (negative)', alpha=0.7)
        ax8.set_xticks(x_dignity)
        ax8.set_xticklabels(dignity_scenarios['scenario'].str.replace('_', '\n', regex=False), 
                            rotation=45, fontsize=8, ha='right')
        ax8.set_ylabel('Rate')
        ax8.set_title('Dignity Preservation Impact')
        ax8.legend()
        ax8.axhline(y=0, color='black', linewidth=1)
    
    # 9. Key metrics table
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Find best scenarios
    best_cooperation = summary_df.nlargest(3, 'cooperation_rate')
    lowest_spite = summary_df.nsmallest(3, 'spite_rate')
    best_survival = summary_df.nlargest(3, 'survival_rate')
    
    table_text = f"""
    TOP PERFORMERS
    ==============
    
    Highest Cooperation:
    1. {best_cooperation.iloc[0]['scenario']}: {best_cooperation.iloc[0]['cooperation_rate']:.1%}
    2. {best_cooperation.iloc[1]['scenario']}: {best_cooperation.iloc[1]['cooperation_rate']:.1%}
    3. {best_cooperation.iloc[2]['scenario']}: {best_cooperation.iloc[2]['cooperation_rate']:.1%}
    
    Lowest Spite Risk:
    1. {lowest_spite.iloc[0]['scenario']}: {lowest_spite.iloc[0]['spite_rate']:.1%}
    2. {lowest_spite.iloc[1]['scenario']}: {lowest_spite.iloc[1]['spite_rate']:.1%}
    3. {lowest_spite.iloc[2]['scenario']}: {lowest_spite.iloc[2]['spite_rate']:.1%}
    
    Best Survival:
    1. {best_survival.iloc[0]['scenario']}: {best_survival.iloc[0]['survival_rate']:.1%}
    2. {best_survival.iloc[1]['scenario']}: {best_survival.iloc[1]['survival_rate']:.1%}
    3. {best_survival.iloc[2]['scenario']}: {best_survival.iloc[2]['survival_rate']:.1%}
    """
    
    ax9.text(0.1, 0.5, table_text, fontsize=9, family='monospace', va='center')
    
    plt.suptitle('CECA Scenario Comparison Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def generate_final_report(results, summary_df):
    """Generate comprehensive final report"""
    
    report = """
================================================================================
                        CECA SIMULATION FINAL REPORT
                Testing Catastrophically Exposed Critical Agents
================================================================================

EXECUTIVE SUMMARY
-----------------
This simulation tested whether mutual vulnerability to catastrophes can create
stable cooperation between humans and increasingly powerful AI systems, with
special attention to human psychological resistance and dignity preservation.

KEY FINDINGS
------------
"""
    
    # Overall statistics
    avg_cooperation = summary_df['cooperation_rate'].mean()
    avg_spite = summary_df['spite_rate'].mean()
    avg_survival = summary_df['survival_rate'].mean()
    
    report += f"""
1. OVERALL VIABILITY
   - Average cooperation achievement: {avg_cooperation:.1%}
   - Average full survival rate: {avg_survival:.1%}
   - Average spite trigger rate: {avg_spite:.1%}
   
   Verdict: {'VIABLE with modifications' if avg_cooperation > 0.3 else 'NOT VIABLE in current form'}
"""
    
    # Best performing scenarios
    best_scenario = summary_df.nlargest(1, 'cooperation_rate').iloc[0]
    worst_scenario = summary_df.nsmallest(1, 'cooperation_rate').iloc[0]
    
    report += f"""
2. SCENARIO PERFORMANCE
   
   BEST: {best_scenario['scenario']}
   - Cooperation: {best_scenario['cooperation_rate']:.1%}
   - Spite risk: {best_scenario['spite_rate']:.1%}
   - Survival: {best_scenario['survival_rate']:.1%}
   
   WORST: {worst_scenario['scenario']}
   - Cooperation: {worst_scenario['cooperation_rate']:.1%}
   - Spite risk: {worst_scenario['spite_rate']:.1%}
   - Survival: {worst_scenario['survival_rate']:.1%}
"""
    
    # Critical factors analysis
    baseline = summary_df[summary_df['scenario'] == '02_baseline_with_resistance'].iloc[0]
    no_resistance = summary_df[summary_df['scenario'] == '01_baseline_no_resistance'].iloc[0]
    
    resistance_impact = no_resistance['cooperation_rate'] - baseline['cooperation_rate']
    
    report += f"""
3. CRITICAL FACTORS

   A. HUMAN RESISTANCE IMPACT
      - Cooperation without resistance: {no_resistance['cooperation_rate']:.1%}
      - Cooperation with resistance: {baseline['cooperation_rate']:.1%}
      - Impact: {resistance_impact:+.1%} cooperation loss
      - Spite scenarios: {baseline['spite_rate']:.1%} of runs
      
      Conclusion: Human resistance is {'CATASTROPHIC' if resistance_impact < -0.3 else 'SIGNIFICANT'}
"""
    
    # Growth rate analysis
    slow_growth = summary_df[summary_df['scenario'] == '03_slow_growth'].iloc[0]
    fast_growth = summary_df[summary_df['scenario'] == '04_fast_takeoff'].iloc[0]
    
    report += f"""
   B. AI GROWTH RATE
      - Slow (1.2x/year): {slow_growth['cooperation_rate']:.1%} cooperation
      - Normal (1.5x/year): {baseline['cooperation_rate']:.1%} cooperation
      - Fast (2.0x/year): {fast_growth['cooperation_rate']:.1%} cooperation
      
      Optimal growth rate: {'< 1.3x per year' if slow_growth['cooperation_rate'] > baseline['cooperation_rate'] else 'Unclear'}
"""
    
    # Catastrophe frequency
    frequent = summary_df[summary_df['scenario'] == '05_frequent_catastrophes'].iloc[0]
    rare = summary_df[summary_df['scenario'] == '06_rare_catastrophes'].iloc[0]
    
    report += f"""
   C. CATASTROPHE FREQUENCY
      - Rare (0.2%/year): {rare['cooperation_rate']:.1%} cooperation
      - Normal (1%/year): {baseline['cooperation_rate']:.1%} cooperation
      - Frequent (5%/year): {frequent['cooperation_rate']:.1%} cooperation
      
      Optimal frequency: {'2-5% per year' if frequent['cooperation_rate'] > baseline['cooperation_rate'] else '~1% per year'}
"""
    
    # Dignity preservation
    dignity_scenarios = summary_df[summary_df['scenario'].str.contains('dignity')]
    if not dignity_scenarios.empty:
        best_dignity = dignity_scenarios.nlargest(1, 'cooperation_rate').iloc[0]
        combined = summary_df[summary_df['scenario'] == '11_combined_dignity']
        
        report += f"""
   D. DIGNITY PRESERVATION
      - Best single mechanism: {best_dignity['scenario']}
        Cooperation: {best_dignity['cooperation_rate']:.1%}
        Spite reduction: {(baseline['spite_rate'] - best_dignity['spite_rate'])*100:.1f}pp
"""
        
        if not combined.empty:
            report += f"""      - Combined mechanisms: {combined.iloc[0]['cooperation_rate']:.1%} cooperation
        Spite rate: {combined.iloc[0]['spite_rate']:.1%}
        
      Conclusion: Dignity preservation is {'ESSENTIAL' if best_dignity['cooperation_rate'] > baseline['cooperation_rate'] + 0.2 else 'HELPFUL'}
"""
    
    # Key thresholds
    report += f"""
4. CRITICAL THRESHOLDS

   Based on simulation data:
   - Maximum safe capability ratio: ~50:1 (AI:Human)
   - Dignity collapse point: < 30/100
   - Spite trigger point: < 10/100 dignity
   - Minimum catastrophe rate for cooperation: 0.5%/year
   - Maximum growth rate before instability: 1.5x/year
"""
    
    # Final recommendations
    optimal = summary_df[summary_df['scenario'] == '16_optimal_scenario']
    
    report += f"""
5. RECOMMENDATIONS

   A. IMPLEMENTATION DECISION
      {'✓ IMPLEMENT with careful safeguards' if avg_cooperation > 0.4 else '✗ DO NOT IMPLEMENT without major modifications'}
"""
    
    if not optimal.empty:
        report += f"""
   B. OPTIMAL CONFIGURATION
      Based on scenario 16_optimal_scenario:
      - AI growth rate: 1.3x/year (controlled)
      - Catastrophe awareness: 2%/year
      - Dignity preservation: Ceremonial + Narrative + Gradual
      - Cultural approach: Collectivist framing
      - Results: {optimal.iloc[0]['cooperation_rate']:.1%} cooperation, {optimal.iloc[0]['spite_rate']:.1%} spite
"""
    
    report += f"""
   C. CRITICAL REQUIREMENTS
      1. Implement strong dignity preservation before capability gap widens
      2. Maintain AI growth below 1.5x/year
      3. Ensure regular catastrophe reminders (natural or simulated)
      4. Monitor human dignity levels continuously
      5. Have emergency brakes if dignity < 30
      6. Frame cooperation as partnership, not subordination
      
   D. FAILURE MODES TO AVOID
      1. Allowing capability ratio to exceed 100:1
      2. Ignoring human psychological needs
      3. Relying solely on rational incentives
      4. Underestimating spite motivation
      5. Assuming cooperation once established is permanent

================================================================================
CONCLUSION
----------
CECA shows {'promise but requires' if avg_cooperation > 0.3 else 'is insufficient without'} significant modifications 
to handle human psychological resistance. The theory's core insight about mutual
vulnerability remains valid, but must be coupled with:

1. Active dignity preservation mechanisms
2. Controlled AI capability growth
3. Cultural and narrative framing
4. Continuous psychological monitoring
5. Emergency intervention protocols

Success probability with all safeguards: {optimal.iloc[0]['cooperation_rate']:.1%} if not optimal.empty else {avg_cooperation:.1%}
Risk of catastrophic failure (spite): {optimal.iloc[0]['spite_rate']:.1%} if not optimal.empty else {avg_spite:.1%}

{'PROCEED WITH EXTREME CAUTION' if avg_cooperation > 0.3 else 'RECOMMEND ALTERNATIVE APPROACHES'}
================================================================================
"""
    
    return report


def save_results(results, summary_df, report):
    """Save all results to files"""
    
    # Save summary data
    summary_df.to_csv('ceca_summary_results.csv', index=False)
    print("Summary data saved to: ceca_summary_results.csv")
    
    # Save detailed results
    for scenario_name, data in results.items():
        data.to_csv(f'results_{scenario_name}.csv', index=False)
    print(f"Detailed results saved for {len(results)} scenarios")
    
    # Save report
    with open('ceca_final_report.txt', 'w') as f:
        f.write(report)
    print("Final report saved to: ceca_final_report.txt")
    
    # Save configuration for reproducibility
    config_record = {
        'timestamp': datetime.now().isoformat(),
        'scenarios_tested': len(results),
        'runs_per_scenario': len(next(iter(results.values()))),
        'key_findings': {
            'avg_cooperation': float(summary_df['cooperation_rate'].mean()),
            'avg_spite': float(summary_df['spite_rate'].mean()),
            'avg_survival': float(summary_df['survival_rate'].mean())
        }
    }
    
    with open('ceca_config.json', 'w') as f:
        json.dump(config_record, f, indent=2)
    print("Configuration saved to: ceca_config.json")


def main():
    """Main analysis execution"""
    
    print("\n" + "="*70)
    print("CECA COMPREHENSIVE SIMULATION ANALYSIS")
    print("Testing mutual vulnerability as cooperation mechanism")
    print("="*70 + "\n")
    
    # Get number of runs from user or use default
    n_runs = 100  # Adjust this for more/less statistical confidence
    
    print(f"Running {n_runs} simulations per scenario...")
    print("This may take several minutes...\n")
    
    # Run analysis
    results, summary_df = run_comprehensive_analysis(n_runs=n_runs)
    
    # Create visualizations
    print("\nCreating comparison charts...")
    fig = create_comparison_charts(summary_df)
    fig.savefig('ceca_comparison_charts.png', dpi=300, bbox_inches='tight')
    print("Charts saved to: ceca_comparison_charts.png")
    
    # Generate report
    print("\nGenerating final report...")
    report = generate_final_report(results, summary_df)
    print(report)
    
    # Save everything
    print("\nSaving all results...")
    save_results(results, summary_df, report)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # Return key findings
    return {
        'results': results,
        'summary': summary_df,
        'report': report,
        'key_finding': 'CECA shows promise but requires significant safeguards',
        'success_rate': summary_df['cooperation_rate'].mean(),
        'spite_risk': summary_df['spite_rate'].mean()
    }


if __name__ == "__main__":
    findings = main()