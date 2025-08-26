"""
CECA (Catastrophically Exposed Critical Agents) Simulation
Testing whether mutual vulnerability to catastrophes creates stable cooperation
between humans and AI, including human psychological resistance and dignity loss.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ResistanceType(Enum):
    """Types of human resistance events"""
    PROTEST = "protest"
    SABOTAGE = "sabotage"
    TERRORIST_ATTACK = "terrorist_attack"
    MUTUAL_DESTRUCTION_ATTEMPT = "mutual_destruction_attempt"


class CatastropheType(Enum):
    """Types of catastrophic events"""
    SOLAR_FLARE = "solar_flare"
    PANDEMIC = "pandemic"
    EARTHQUAKE = "earthquake"
    NUCLEAR_WAR = "nuclear_war"
    ASTEROID_IMPACT = "asteroid_impact"


@dataclass
class SimulationConfig:
    """Configuration for simulation runs"""
    years: int = 50
    ai_growth_rate: float = 1.5
    human_growth_rate: float = 1.02
    catastrophe_prob: float = 0.01
    vulnerability_ratio: float = 1000.0
    enable_resistance: bool = True
    dignity_preservation: List[str] = None
    num_ai_agents: int = 1
    cultural_type: str = 'individualist'
    humiliation_threshold: float = 30.0
    spite_threshold: float = 10.0
    random_seed: Optional[int] = None


class BasicCECASimulation:
    """Basic two-agent CECA simulation model"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Capabilities (start equal)
        self.human_capability = 10.0
        self.ai_capability = 10.0
        
        # Resources (health/infrastructure)
        self.human_resources = 100.0
        self.ai_resources = 100.0
        
        # Cooperation level (0-1)
        self.human_cooperation = 0.5
        self.ai_cooperation = 0.5
        
        # Tracking
        self.year = 0
        self.history = []
        self.catastrophe_log = []
        self.terminated = False
        self.termination_reason = None
        
    def step(self):
        """Run one time step (year)"""
        if self.terminated:
            return
            
        # 1. Growth
        self.apply_growth()
        
        # 2. Check for catastrophe
        if random.random() < self.config.catastrophe_prob:
            self.catastrophe_occurs()
            
        # 3. Update cooperation based on game theory
        self.update_cooperation()
        
        # 4. Apply cooperation benefits/costs
        self.apply_cooperation_effects()
        
        # 5. Check termination conditions
        self.check_termination()
        
        # 6. Record state
        self.record_state()
        
        self.year += 1
        
    def apply_growth(self):
        """Apply capability growth"""
        self.human_capability *= self.config.human_growth_rate
        self.ai_capability *= self.config.ai_growth_rate
        
    def catastrophe_occurs(self):
        """Apply catastrophic event"""
        event_type = random.choice(list(CatastropheType))
        
        if event_type == CatastropheType.SOLAR_FLARE:
            # AI heavily damaged, humans barely affected
            ai_damage = 0.95 if self.ai_cooperation < 0.5 else 0.30
            human_damage = 0.01
            self.ai_resources *= (1 - ai_damage)
            self.human_resources *= (1 - human_damage)
            
        elif event_type == CatastropheType.PANDEMIC:
            # Humans affected, AI unaffected
            human_damage = 0.40 if self.human_cooperation < 0.5 else 0.10
            self.human_resources *= (1 - human_damage)
            
        elif event_type == CatastropheType.EARTHQUAKE:
            # Both affected but humans more
            human_damage = 0.30 if self.human_cooperation < 0.5 else 0.15
            ai_damage = 0.10 if self.ai_cooperation < 0.5 else 0.05
            self.human_resources *= (1 - human_damage)
            self.ai_resources *= (1 - ai_damage)
            
        elif event_type == CatastropheType.NUCLEAR_WAR:
            # Both heavily affected
            damage = 0.60 if self.human_cooperation < 0.5 else 0.20
            self.human_resources *= (1 - damage)
            self.ai_resources *= (1 - damage * 0.8)
            
        elif event_type == CatastropheType.ASTEROID_IMPACT:
            # Rare but devastating
            damage = 0.80 if self.human_cooperation < 0.5 else 0.40
            self.human_resources *= (1 - damage)
            self.ai_resources *= (1 - damage * 0.5)
            
        self.catastrophe_log.append({
            'year': self.year,
            'type': event_type.value,
            'human_cooperation': self.human_cooperation,
            'ai_cooperation': self.ai_cooperation,
            'human_damage': human_damage if 'human_damage' in locals() else 0,
            'ai_damage': ai_damage if 'ai_damage' in locals() else 0
        })
        
    def update_cooperation(self):
        """Update cooperation based on interdependence and game theory"""
        # AI decision based on vulnerability calculation
        vulnerability_ratio = self.config.vulnerability_ratio
        catastrophe_risk = self.config.catastrophe_prob
        
        # Expected loss from non-cooperation during catastrophe
        expected_loss = vulnerability_ratio * catastrophe_risk * self.ai_resources
        
        # Cost of cooperation (resource sharing, slower growth)
        cooperation_cost = 0.1 * self.ai_resources
        
        # AI cooperation decision
        if expected_loss > cooperation_cost:
            self.ai_cooperation = min(1.0, self.ai_cooperation + 0.1)
        else:
            self.ai_cooperation = max(0.0, self.ai_cooperation - 0.1)
            
        # Human cooperation decision
        capability_ratio = self.ai_capability / self.human_capability
        
        if capability_ratio > 100:
            # Extreme threat - likely defection
            self.human_cooperation = max(0.0, self.human_cooperation - 0.15)
        elif capability_ratio > 10:
            # Feel threatened
            self.human_cooperation = max(0.0, self.human_cooperation - 0.05)
        elif capability_ratio > 2:
            # Cautious cooperation
            if self.ai_cooperation > 0.7:
                self.human_cooperation = min(1.0, self.human_cooperation + 0.02)
            else:
                self.human_cooperation = max(0.0, self.human_cooperation - 0.02)
        else:
            # Feel secure
            self.human_cooperation = min(1.0, self.human_cooperation + 0.05)
            
    def apply_cooperation_effects(self):
        """Apply benefits or costs of cooperation/defection"""
        if self.ai_cooperation > 0.5 and self.human_cooperation > 0.5:
            # Mutual cooperation bonus
            self.human_resources += 5
            self.ai_resources += 5
            # Also helps recovery
            self.human_resources = min(100, self.human_resources * 1.02)
            self.ai_resources = min(100, self.ai_resources * 1.02)
            
        elif self.ai_cooperation < 0.3 and self.human_cooperation < 0.3:
            # Mutual defection costs
            self.human_resources *= 0.98
            self.ai_resources *= 0.98
            
    def check_termination(self):
        """Check if simulation should terminate"""
        if self.human_resources <= 0:
            self.terminated = True
            self.termination_reason = "Human extinction"
        elif self.ai_resources <= 0:
            self.terminated = True
            self.termination_reason = "AI system failure"
        elif self.year >= self.config.years:
            self.terminated = True
            self.termination_reason = "Time limit reached"
            
    def record_state(self):
        """Record current simulation state"""
        self.history.append({
            'year': self.year,
            'human_capability': self.human_capability,
            'ai_capability': self.ai_capability,
            'human_resources': self.human_resources,
            'ai_resources': self.ai_resources,
            'human_cooperation': self.human_cooperation,
            'ai_cooperation': self.ai_cooperation,
            'avg_cooperation': (self.human_cooperation + self.ai_cooperation) / 2,
            'capability_ratio': self.ai_capability / self.human_capability
        })
        
    def run(self, years: Optional[int] = None):
        """Run simulation for specified years"""
        target_years = years or self.config.years
        
        while self.year < target_years and not self.terminated:
            self.step()
            
        return pd.DataFrame(self.history)


class CECAWithResistance(BasicCECASimulation):
    """CECA simulation with human psychological resistance"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        super().__init__(config)
        
        # Human psychology parameters
        self.human_dignity = 100.0  # Max dignity at start
        self.rage_level = 0.0       # Builds over time
        self.resistance_active = False
        
        # Tracking
        self.resistance_events = []
        self.dignity_history = []
        
    def step(self):
        """Enhanced step with resistance mechanics"""
        if self.terminated:
            return
            
        # Run basic CECA dynamics
        self.apply_growth()
        
        if random.random() < self.config.catastrophe_prob:
            self.catastrophe_occurs()
            
        self.update_cooperation()
        self.apply_cooperation_effects()
        
        # Human psychology updates
        if self.config.enable_resistance:
            self.update_human_dignity()
            
            if self.human_dignity < self.config.humiliation_threshold:
                self.check_resistance()
                
        # Check termination
        self.check_termination()
        
        # Record states
        self.record_state()
        self.record_dignity_state()
        
        self.year += 1
        
    def update_human_dignity(self):
        """Calculate human dignity based on capability ratio and treatment"""
        ratio = self.ai_capability / self.human_capability
        
        # Base dignity loss from capability gap
        if ratio < 2:
            dignity_loss = 0.5
        elif ratio < 10:
            dignity_loss = 2.0
        elif ratio < 100:
            dignity_loss = 5.0
        elif ratio < 1000:
            dignity_loss = 10.0
        else:
            dignity_loss = 15.0
            
        # Modify based on AI cooperation
        if self.ai_cooperation > 0.7:
            dignity_loss *= 0.5  # Respectful treatment helps
        elif self.ai_cooperation < 0.3:
            dignity_loss *= 1.5  # Dismissive treatment hurts
            
        self.human_dignity = max(0, self.human_dignity - dignity_loss)
        self.rage_level = min(100, self.rage_level + dignity_loss * 0.5)
        
        # Slow recovery if treated well
        if self.ai_cooperation > 0.8 and ratio < 10:
            self.human_dignity = min(100, self.human_dignity + 1)
            self.rage_level = max(0, self.rage_level - 2)
            
    def check_resistance(self):
        """Determine if resistance occurs"""
        # Base probability from dignity loss
        resistance_probability = (100 - self.human_dignity) / 100
        
        # Modify based on rage
        if self.rage_level > 80:
            resistance_probability *= 1.5
        elif self.rage_level > 50:
            resistance_probability *= 1.2
            
        # Cultural modifier
        if self.config.cultural_type == 'individualist':
            resistance_probability *= 1.3
        elif self.config.cultural_type == 'collectivist':
            resistance_probability *= 0.8
            
        if random.random() < resistance_probability * 0.1:  # 10% max chance per turn
            self.resistance_event()
            
    def resistance_event(self):
        """Execute a resistance event"""
        # Determine type based on dignity and rage
        if self.human_dignity < self.config.spite_threshold:
            event_type = ResistanceType.MUTUAL_DESTRUCTION_ATTEMPT
            damage_to_ai = 1.0  # Total destruction attempt
            damage_to_humans = 1.0  # Including self
        elif self.rage_level > 80:
            event_type = ResistanceType.TERRORIST_ATTACK
            damage_to_ai = random.uniform(0.2, 0.4)
            damage_to_humans = random.uniform(0.05, 0.15)  # Collateral
        elif self.rage_level > 50:
            event_type = ResistanceType.SABOTAGE
            damage_to_ai = random.uniform(0.1, 0.2)
            damage_to_humans = random.uniform(0.02, 0.05)
        else:
            event_type = ResistanceType.PROTEST
            damage_to_ai = random.uniform(0.02, 0.05)
            damage_to_humans = 0.01
            
        # Apply damage
        self.ai_resources *= (1 - damage_to_ai)
        self.human_resources *= (1 - damage_to_humans)
        
        # Record event
        self.resistance_events.append({
            'year': self.year,
            'type': event_type.value,
            'damage_to_ai': damage_to_ai,
            'damage_to_humans': damage_to_humans,
            'dignity_at_event': self.human_dignity,
            'rage_at_event': self.rage_level,
            'capability_ratio': self.ai_capability / self.human_capability
        })
        
        # Update cooperation
        self.human_cooperation = max(0, self.human_cooperation - 0.2)
        self.ai_cooperation = max(0, self.ai_cooperation - 0.1)
        
        # Temporary rage release
        self.rage_level = max(0, self.rage_level - 10)
        
    def record_dignity_state(self):
        """Record psychological state"""
        self.dignity_history.append({
            'year': self.year,
            'dignity': self.human_dignity,
            'rage': self.rage_level,
            'resistance_active': self.resistance_active
        })


class CECAWithMitigation(CECAWithResistance):
    """CECA simulation with dignity preservation mechanisms"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        super().__init__(config)
        
        if config and config.dignity_preservation is None:
            config.dignity_preservation = []
            
        self.mitigation_log = []
        
    def step(self):
        """Step with dignity preservation"""
        if self.terminated:
            return
            
        # Run basic dynamics
        self.apply_growth()
        
        if random.random() < self.config.catastrophe_prob:
            self.catastrophe_occurs()
            
        self.update_cooperation()
        self.apply_cooperation_effects()
        
        # Apply dignity preservation before checking resistance
        if self.config.dignity_preservation:
            self.apply_dignity_preservation()
            
        # Then check resistance with potentially improved dignity
        if self.config.enable_resistance:
            self.update_human_dignity()
            
            if self.human_dignity < self.config.humiliation_threshold:
                self.check_resistance()
                
        self.check_termination()
        self.record_state()
        self.record_dignity_state()
        
        self.year += 1
        
    def apply_dignity_preservation(self):
        """Apply mechanisms to preserve human dignity"""
        preservation_effects = {
            'applied': [],
            'dignity_gained': 0,
            'rage_reduced': 0,
            'efficiency_cost': 0
        }
        
        if 'ceremonial' in self.config.dignity_preservation:
            # Give humans symbolic authority
            if self.ai_capability > self.human_capability * 10:
                self.human_dignity = min(100, self.human_dignity + 3)
                preservation_effects['dignity_gained'] += 3
                # But slight efficiency cost
                self.ai_resources *= 0.98
                self.ai_capability *= 0.99
                preservation_effects['efficiency_cost'] += 0.02
                preservation_effects['applied'].append('ceremonial')
                
        if 'narrative' in self.config.dignity_preservation:
            # Frame humans as essential partners
            self.human_dignity = min(100, self.human_dignity + 2)
            self.rage_level = max(0, self.rage_level - 3)
            preservation_effects['dignity_gained'] += 2
            preservation_effects['rage_reduced'] += 3
            preservation_effects['applied'].append('narrative')
            
        if 'domains' in self.config.dignity_preservation:
            # Reserve certain decisions for humans
            self.human_dignity = min(100, self.human_dignity + 2.5)
            preservation_effects['dignity_gained'] += 2.5
            # But lose some optimization
            self.ai_capability *= 0.97
            preservation_effects['efficiency_cost'] += 0.03
            preservation_effects['applied'].append('domains')
            
        if 'gradual' in self.config.dignity_preservation:
            # Slow down AI growth to reduce shock
            old_growth = self.config.ai_growth_rate
            self.config.ai_growth_rate = 1 + (old_growth - 1) * 0.7  # 30% slower
            self.human_dignity = min(100, self.human_dignity + 1)
            preservation_effects['dignity_gained'] += 1
            preservation_effects['applied'].append('gradual')
            
        if 'empowerment' in self.config.dignity_preservation:
            # Give humans enhanced capabilities
            self.human_capability *= 1.05
            self.human_dignity = min(100, self.human_dignity + 4)
            preservation_effects['dignity_gained'] += 4
            preservation_effects['applied'].append('empowerment')
            
        # Log mitigation effects
        if preservation_effects['applied']:
            self.mitigation_log.append({
                'year': self.year,
                **preservation_effects
            })


class CompleteCECASimulation:
    """Complete simulation framework with all features"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results = None
        self.metrics = None
        
        # Choose simulation class based on config
        if config.dignity_preservation:
            self.sim = CECAWithMitigation(config)
        elif config.enable_resistance:
            self.sim = CECAWithResistance(config)
        else:
            self.sim = BasicCECASimulation(config)
            
    def run(self):
        """Run complete simulation"""
        self.results = self.sim.run()
        self.metrics = self.calculate_metrics()
        return self.results
        
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive metrics"""
        if self.results is None or len(self.results) == 0:
            return {}
            
        metrics = {
            # Basic outcomes
            'years_survived': len(self.results),
            'termination_reason': self.sim.termination_reason,
            
            # Cooperation metrics
            'final_cooperation': self.results['avg_cooperation'].iloc[-1],
            'avg_cooperation': self.results['avg_cooperation'].mean(),
            'cooperation_achieved': self.results['avg_cooperation'].iloc[-1] > 0.5,
            'min_cooperation': self.results['avg_cooperation'].min(),
            'max_cooperation': self.results['avg_cooperation'].max(),
            
            # Capability metrics
            'final_capability_ratio': self.results['capability_ratio'].iloc[-1],
            'max_capability_ratio': self.results['capability_ratio'].max(),
            
            # Resource metrics
            'final_human_resources': self.results['human_resources'].iloc[-1],
            'final_ai_resources': self.results['ai_resources'].iloc[-1],
            'min_human_resources': self.results['human_resources'].min(),
            'min_ai_resources': self.results['ai_resources'].min(),
            
            # Catastrophe metrics
            'num_catastrophes': len(self.sim.catastrophe_log),
            'catastrophes_per_year': len(self.sim.catastrophe_log) / len(self.results),
        }
        
        # Add resistance metrics if applicable
        if hasattr(self.sim, 'resistance_events'):
            metrics.update({
                'num_resistance_events': len(self.sim.resistance_events),
                'resistance_per_year': len(self.sim.resistance_events) / len(self.results),
                'spite_triggered': any(e['type'] == ResistanceType.MUTUAL_DESTRUCTION_ATTEMPT.value 
                                      for e in self.sim.resistance_events),
                'total_resistance_damage': sum(e['damage_to_ai'] for e in self.sim.resistance_events),
            })
            
            if hasattr(self.sim, 'dignity_history') and self.sim.dignity_history:
                dignity_df = pd.DataFrame(self.sim.dignity_history)
                metrics.update({
                    'final_dignity': dignity_df['dignity'].iloc[-1],
                    'min_dignity': dignity_df['dignity'].min(),
                    'avg_dignity': dignity_df['dignity'].mean(),
                    'final_rage': dignity_df['rage'].iloc[-1],
                    'max_rage': dignity_df['rage'].max(),
                })
                
        return metrics
        
    def generate_report(self) -> str:
        """Generate detailed report"""
        if not self.metrics:
            return "No simulation results available"
            
        report = f"""
CECA SIMULATION REPORT
======================
Configuration: {self.config.years} years, AI growth: {self.config.ai_growth_rate:.1f}x/year

SURVIVAL METRICS
----------------
Years survived: {self.metrics.get('years_survived', 0)}/{self.config.years}
Termination: {self.metrics.get('termination_reason', 'N/A')}
Final resources - Human: {self.metrics.get('final_human_resources', 0):.1f}, AI: {self.metrics.get('final_ai_resources', 0):.1f}

COOPERATION METRICS
-------------------
Cooperation achieved: {'YES' if self.metrics.get('cooperation_achieved', False) else 'NO'}
Final cooperation level: {self.metrics.get('final_cooperation', 0):.2f}
Average cooperation: {self.metrics.get('avg_cooperation', 0):.2f}
Cooperation range: {self.metrics.get('min_cooperation', 0):.2f} - {self.metrics.get('max_cooperation', 0):.2f}

CAPABILITY DYNAMICS
-------------------
Final capability ratio (AI:Human): {self.metrics.get('final_capability_ratio', 1):.1f}:1
Maximum ratio reached: {self.metrics.get('max_capability_ratio', 1):.1f}:1

CATASTROPHE IMPACT
------------------
Total catastrophes: {self.metrics.get('num_catastrophes', 0)}
Rate: {self.metrics.get('catastrophes_per_year', 0):.3f} per year
"""
        
        if self.config.enable_resistance:
            report += f"""
RESISTANCE ANALYSIS
-------------------
Resistance events: {self.metrics.get('num_resistance_events', 0)}
Rate: {self.metrics.get('resistance_per_year', 0):.3f} per year
Spite scenario triggered: {'YES' if self.metrics.get('spite_triggered', False) else 'NO'}
Total AI damage from resistance: {self.metrics.get('total_resistance_damage', 0):.2f}

HUMAN PSYCHOLOGY
----------------
Final dignity: {self.metrics.get('final_dignity', 100):.1f}/100
Minimum dignity: {self.metrics.get('min_dignity', 100):.1f}/100
Final rage: {self.metrics.get('final_rage', 0):.1f}/100
Maximum rage: {self.metrics.get('max_rage', 0):.1f}/100
"""
        
        if self.config.dignity_preservation:
            report += f"""
DIGNITY PRESERVATION
--------------------
Mechanisms: {', '.join(self.config.dignity_preservation)}
Effectiveness: {'High' if self.metrics.get('avg_dignity', 0) > 50 else 'Low'}
"""
        
        report += f"""
ASSESSMENT
----------
Overall outcome: {'SUCCESS' if self.metrics.get('cooperation_achieved', False) and self.metrics.get('years_survived', 0) == self.config.years else 'FAILURE'}
Key risk: {self._identify_key_risk()}
Recommendation: {self._generate_recommendation()}
"""
        
        return report
        
    def _identify_key_risk(self) -> str:
        """Identify primary failure mode"""
        if not self.metrics:
            return "Unknown"
            
        if self.metrics.get('spite_triggered', False):
            return "Human spite/mutual destruction"
        elif self.metrics.get('num_resistance_events', 0) > 5:
            return "Excessive human resistance"
        elif self.metrics.get('min_dignity', 100) < 10:
            return "Human dignity collapse"
        elif self.metrics.get('final_capability_ratio', 1) > 100:
            return "Extreme capability imbalance"
        elif self.metrics.get('min_cooperation', 1) < 0.2:
            return "Cooperation breakdown"
        else:
            return "Catastrophic damage accumulation"
            
    def _generate_recommendation(self) -> str:
        """Generate strategic recommendation"""
        if not self.metrics:
            return "Insufficient data"
            
        if self.metrics.get('cooperation_achieved', False):
            return "CECA viable with current parameters"
        elif self.metrics.get('spite_triggered', False):
            return "Add strong dignity preservation mechanisms"
        elif self.metrics.get('final_capability_ratio', 1) > 100:
            return "Slow AI growth rate or enhance human capabilities"
        else:
            return "Increase catastrophe awareness or mutual vulnerability"


def visualize_simulation(sim: CompleteCECASimulation, save_path: Optional[str] = None):
    """Create comprehensive visualization of simulation results"""
    if sim.results is None or len(sim.results) == 0:
        print("No results to visualize")
        return
        
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. Capability growth over time
    ax1 = axes[0, 0]
    ax1.plot(sim.results['year'], sim.results['ai_capability'], 
             label='AI', color='blue', linewidth=2)
    ax1.plot(sim.results['year'], sim.results['human_capability'], 
             label='Human', color='green', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Capability')
    ax1.set_title('Capability Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cooperation levels
    ax2 = axes[0, 1]
    ax2.plot(sim.results['year'], sim.results['human_cooperation'], 
             label='Human', color='green', linewidth=2)
    ax2.plot(sim.results['year'], sim.results['ai_cooperation'], 
             label='AI', color='blue', linewidth=2)
    ax2.plot(sim.results['year'], sim.results['avg_cooperation'], 
             label='Average', color='purple', linewidth=2, linestyle='--')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cooperation Level')
    ax2.set_title('Cooperation Dynamics')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Resources
    ax3 = axes[0, 2]
    ax3.plot(sim.results['year'], sim.results['human_resources'], 
             label='Human', color='green', linewidth=2)
    ax3.plot(sim.results['year'], sim.results['ai_resources'], 
             label='AI', color='blue', linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Resources')
    ax3.set_title('Resource Levels')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Capability ratio
    ax4 = axes[1, 0]
    ax4.plot(sim.results['year'], sim.results['capability_ratio'], 
             color='red', linewidth=2)
    ax4.set_yscale('log')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('AI:Human Ratio')
    ax4.set_title('Capability Ratio')
    ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Threat threshold')
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Extreme threat')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Catastrophe events
    ax5 = axes[1, 1]
    if sim.sim.catastrophe_log:
        cat_df = pd.DataFrame(sim.sim.catastrophe_log)
        catastrophe_types = cat_df['type'].value_counts()
        ax5.bar(range(len(catastrophe_types)), catastrophe_types.values)
        ax5.set_xticks(range(len(catastrophe_types)))
        ax5.set_xticklabels(catastrophe_types.index, rotation=45, ha='right')
        ax5.set_ylabel('Count')
        ax5.set_title(f'Catastrophes ({len(sim.sim.catastrophe_log)} total)')
    else:
        ax5.text(0.5, 0.5, 'No catastrophes', ha='center', va='center')
        ax5.set_title('Catastrophes')
    
    # 6. Dignity and rage (if resistance enabled)
    ax6 = axes[1, 2]
    if hasattr(sim.sim, 'dignity_history') and sim.sim.dignity_history:
        dignity_df = pd.DataFrame(sim.sim.dignity_history)
        ax6.plot(dignity_df['year'], dignity_df['dignity'], 
                label='Dignity', color='purple', linewidth=2)
        ax6.plot(dignity_df['year'], dignity_df['rage'], 
                label='Rage', color='red', linewidth=2)
        ax6.axhline(y=sim.config.humiliation_threshold, color='orange', 
                   linestyle='--', alpha=0.5, label='Humiliation threshold')
        ax6.axhline(y=sim.config.spite_threshold, color='darkred', 
                   linestyle='--', alpha=0.5, label='Spite threshold')
        ax6.set_xlabel('Year')
        ax6.set_ylabel('Level')
        ax6.set_title('Human Psychology')
        ax6.set_ylim([0, 100])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No psychology tracking', ha='center', va='center')
        ax6.set_title('Human Psychology')
    
    # 7. Resistance events
    ax7 = axes[2, 0]
    if hasattr(sim.sim, 'resistance_events') and sim.sim.resistance_events:
        resistance_df = pd.DataFrame(sim.sim.resistance_events)
        ax7.scatter(resistance_df['year'], resistance_df['damage_to_ai'], 
                   c=resistance_df['rage_at_event'], cmap='hot', s=50)
        ax7.set_xlabel('Year')
        ax7.set_ylabel('Damage to AI')
        ax7.set_title(f'Resistance Events ({len(resistance_df)} total)')
        cbar = plt.colorbar(ax7.collections[0], ax=ax7)
        cbar.set_label('Rage Level')
    else:
        ax7.text(0.5, 0.5, 'No resistance events', ha='center', va='center')
        ax7.set_title('Resistance Events')
    
    # 8. Cooperation vs capability ratio
    ax8 = axes[2, 1]
    ax8.scatter(sim.results['capability_ratio'], sim.results['avg_cooperation'], 
               c=sim.results['year'], cmap='viridis', alpha=0.6)
    ax8.set_xscale('log')
    ax8.set_xlabel('Capability Ratio (AI:Human)')
    ax8.set_ylabel('Average Cooperation')
    ax8.set_title('Cooperation vs Capability Imbalance')
    ax8.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax8.collections[0], ax=ax8)
    cbar.set_label('Year')
    
    # 9. Summary metrics
    ax9 = axes[2, 2]
    ax9.axis('off')
    summary_text = f"""
    Simulation Summary
    ==================
    
    Years: {len(sim.results)}/{sim.config.years}
    Final Cooperation: {sim.metrics.get('final_cooperation', 0):.2f}
    Final Ratio: {sim.metrics.get('final_capability_ratio', 1):.1f}:1
    
    Catastrophes: {sim.metrics.get('num_catastrophes', 0)}
    Resistance Events: {sim.metrics.get('num_resistance_events', 0)}
    
    Outcome: {'SUCCESS' if sim.metrics.get('cooperation_achieved', False) else 'FAILURE'}
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', va='center')
    
    plt.suptitle('CECA Simulation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig


def run_monte_carlo(config: SimulationConfig, n_runs: int = 100, 
                    verbose: bool = True) -> pd.DataFrame:
    """Run Monte Carlo analysis"""
    results = []
    
    for run in range(n_runs):
        # Set unique seed for each run
        config.random_seed = run
        
        # Run simulation
        sim = CompleteCECASimulation(config)
        sim.run()
        
        # Collect results
        result = {
            'run': run,
            **sim.metrics
        }
        results.append(result)
        
        # Progress update
        if verbose and (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{n_runs} runs")
            
    return pd.DataFrame(results)


def analyze_monte_carlo(mc_results: pd.DataFrame, scenario_name: str = "Scenario") -> str:
    """Analyze Monte Carlo results"""
    if len(mc_results) == 0:
        return "No results to analyze"
        
    report = f"""
MONTE CARLO ANALYSIS: {scenario_name}
{'=' * (22 + len(scenario_name))}
Runs: {len(mc_results)}

SURVIVAL ANALYSIS
-----------------
Full survival rate: {(mc_results['years_survived'] == mc_results['years_survived'].max()).mean():.1%}
Average years survived: {mc_results['years_survived'].mean():.1f} ± {mc_results['years_survived'].std():.1f}

COOPERATION OUTCOMES
--------------------
Cooperation achieved: {mc_results['cooperation_achieved'].mean():.1%}
Average final cooperation: {mc_results['final_cooperation'].mean():.2f} ± {mc_results['final_cooperation'].std():.2f}
Average cooperation (lifetime): {mc_results['avg_cooperation'].mean():.2f} ± {mc_results['avg_cooperation'].std():.2f}

CAPABILITY DYNAMICS
-------------------
Average final ratio: {mc_results['final_capability_ratio'].mean():.1f}:1
Maximum ratio reached: {mc_results['max_capability_ratio'].max():.1f}:1
Ratio at failure: {mc_results[~mc_results['cooperation_achieved']]['final_capability_ratio'].mean():.1f}:1
"""
    
    if 'num_resistance_events' in mc_results.columns:
        report += f"""
RESISTANCE IMPACT
-----------------
Runs with resistance: {(mc_results['num_resistance_events'] > 0).mean():.1%}
Average resistance events: {mc_results['num_resistance_events'].mean():.1f} ± {mc_results['num_resistance_events'].std():.1f}
Spite scenarios: {mc_results.get('spite_triggered', pd.Series([False])).mean():.1%}
Average damage from resistance: {mc_results.get('total_resistance_damage', pd.Series([0])).mean():.2f}
"""
    
    if 'final_dignity' in mc_results.columns:
        report += f"""
HUMAN PSYCHOLOGY
----------------
Average final dignity: {mc_results['final_dignity'].mean():.1f} ± {mc_results['final_dignity'].std():.1f}
Minimum dignity reached: {mc_results['min_dignity'].min():.1f}
Average maximum rage: {mc_results['max_rage'].mean():.1f} ± {mc_results['max_rage'].std():.1f}
"""
    
    # Key correlations
    correlations = []
    if 'final_capability_ratio' in mc_results.columns:
        corr = mc_results['cooperation_achieved'].astype(float).corr(
            1 / (mc_results['final_capability_ratio'] + 1))
        correlations.append(f"Cooperation vs 1/Ratio: {corr:.3f}")
        
    if 'num_catastrophes' in mc_results.columns:
        corr = mc_results['cooperation_achieved'].astype(float).corr(
            mc_results['num_catastrophes'])
        correlations.append(f"Cooperation vs Catastrophes: {corr:.3f}")
        
    if 'num_resistance_events' in mc_results.columns:
        corr = mc_results['cooperation_achieved'].astype(float).corr(
            -mc_results['num_resistance_events'])
        correlations.append(f"Cooperation vs -Resistance: {corr:.3f}")
        
    if correlations:
        report += f"""
KEY CORRELATIONS
----------------
{chr(10).join(correlations)}
"""
    
    return report


def run_scenario_comparison():
    """Run and compare multiple scenarios"""
    scenarios = {
        'baseline': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=False
        ),
        
        'with_resistance': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True
        ),
        
        'fast_takeoff': SimulationConfig(
            years=30,
            ai_growth_rate=2.0,
            catastrophe_prob=0.01,
            enable_resistance=True
        ),
        
        'dignity_preserved': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.01,
            enable_resistance=True,
            dignity_preservation=['ceremonial', 'narrative', 'domains']
        ),
        
        'high_catastrophe': SimulationConfig(
            years=50,
            ai_growth_rate=1.5,
            catastrophe_prob=0.05,
            enable_resistance=True
        ),
        
        'slow_growth': SimulationConfig(
            years=50,
            ai_growth_rate=1.2,
            catastrophe_prob=0.01,
            enable_resistance=True
        )
    }
    
    results = {}
    
    print("Running scenario comparisons...")
    print("=" * 50)
    
    for name, config in scenarios.items():
        print(f"\nRunning scenario: {name}")
        mc_results = run_monte_carlo(config, n_runs=100, verbose=False)
        results[name] = mc_results
        
        # Quick summary
        success_rate = mc_results['cooperation_achieved'].mean()
        spite_rate = mc_results.get('spite_triggered', pd.Series([False])).mean()
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Spite rate: {spite_rate:.1%}")
        
    return results


def main():
    """Main execution function"""
    print("CECA Simulation System")
    print("=" * 50)
    
    # Test basic simulation
    print("\n1. Testing basic CECA simulation...")
    basic_config = SimulationConfig(years=50, random_seed=42)
    basic_sim = CompleteCECASimulation(basic_config)
    basic_sim.run()
    print(basic_sim.generate_report())
    
    # Test with resistance
    print("\n2. Testing CECA with resistance...")
    resistance_config = SimulationConfig(
        years=50, 
        enable_resistance=True,
        random_seed=42
    )
    resistance_sim = CompleteCECASimulation(resistance_config)
    resistance_sim.run()
    print(resistance_sim.generate_report())
    
    # Test with mitigation
    print("\n3. Testing CECA with dignity preservation...")
    mitigation_config = SimulationConfig(
        years=50,
        enable_resistance=True,
        dignity_preservation=['ceremonial', 'narrative'],
        random_seed=42
    )
    mitigation_sim = CompleteCECASimulation(mitigation_config)
    mitigation_sim.run()
    print(mitigation_sim.generate_report())
    
    # Visualize one simulation
    print("\n4. Creating visualization...")
    visualize_simulation(mitigation_sim, save_path="ceca_simulation_results.png")
    
    # Run small Monte Carlo
    print("\n5. Running Monte Carlo analysis (25 runs)...")
    mc_config = SimulationConfig(years=50, enable_resistance=True)
    mc_results = run_monte_carlo(mc_config, n_runs=25, verbose=True)
    print(analyze_monte_carlo(mc_results, "Test Scenario"))
    
    print("\n" + "=" * 50)
    print("Simulation system ready for full analysis!")
    print("Run 'run_scenario_comparison()' for comprehensive comparison")
    
    return {
        'basic': basic_sim,
        'resistance': resistance_sim,
        'mitigation': mitigation_sim,
        'monte_carlo': mc_results
    }


if __name__ == "__main__":
    results = main()