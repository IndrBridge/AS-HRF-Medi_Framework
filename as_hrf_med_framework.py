import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicinalHarmonicResonanceFramework:
    """
    Simplified Proof of Concept implementation of the Harmonic Resonance Framework
    for integrative medicine, combining Ayurveda, Unani, Allopathy, and Homeopathy
    into a unified computational model.
    """
    
    def __init__(self):
        # Configuration parameters
        self.harmonic_modes = 8  # Simplified from full implementation
        self.phase_coherence_factor = 0.85
        self.enable_logging = True
        
        # Medical system weights (dynamically adjusted by AI)
        self.system_weights = {
            'allopathy': 0.35,    # Fast acting, symptom targeting
            'ayurveda': 0.30,     # Root cause, holistic healing
            'unani': 0.20,        # Temperament, organ function
            'homeopathy': 0.15,   # Energy-based, micro-dosing
        }
        
        # Medicinal compound representations cache
        self.compound_cache = {}
        
        # Current session values
        self.target_disease = None
        self.patient_profile = None
        self.simulated_steps = 30  # Time steps for drug effectiveness simulation
    
    def analyze_compound_combination(self, 
                                     compounds: List[Dict],
                                     target_disease: str,
                                     patient_profile: Optional[Dict] = None,
                                     simulation_steps: int = 30) -> Dict:
        """
        Main analysis function using the Harmonic Resonance principles to evaluate
        a combination of medicinal compounds across different medical systems.
        
        Parameters:
        -----------
        compounds : List[Dict]
            List of compound dictionaries with their properties and medical system
        target_disease : str
            The target disease or condition for treatment
        patient_profile : Dict, optional
            Patient-specific data for personalization
        simulation_steps : int
            Number of temporal steps for efficacy simulation
            
        Returns:
        --------
        Dict : Results including efficacy, safety, and recommendations
        """
        start_time = time.time()
        
        # Store session parameters
        self.target_disease = target_disease
        self.patient_profile = patient_profile or {}
        self.simulated_steps = simulation_steps
        
        # Log simulation parameters
        logger.info(f"Starting integrative medicine analysis with {self.harmonic_modes} harmonic modes")
        logger.info(f"Target disease: {target_disease}, Compounds: {len(compounds)}")
        
        # 1. Transform compounds into phase-state representation
        compound_phases = self._transform_compounds_to_phase_space(compounds)
        
        # 2. Create computation grids for medicinal properties 
        grid_data = self._prepare_medicinal_property_grid(compound_phases)
        
        # 3. Run the harmonic simulation for treatment efficacy
        efficacy_progression = self._run_harmonic_treatment_simulation(
            grid_data,
            num_steps=simulation_steps
        )
        
        # 4. Calculate safety profile based on phase interactions
        safety_profile = self._calculate_safety_profile(grid_data, compound_phases)
        
        # 5. Generate recommendations for optimization
        recommendations = self._generate_optimization_recommendations(
            grid_data, 
            efficacy_progression, 
            safety_profile,
            compounds
        )
        
        # Calculate combined metrics
        max_efficacy = np.max(efficacy_progression['overall'])
        onset_speed = self._calculate_onset_speed(efficacy_progression['overall'])
        sustained_effect = self._calculate_sustained_effect(efficacy_progression['overall'])
        
        # Log simulation results
        simulation_time = time.time() - start_time
        logger.info(f"Harmonic Resonance analysis completed in {simulation_time:.2f} seconds")
        logger.info(f"Max efficacy: {max_efficacy:.2f}, Onset speed: {onset_speed:.2f}, Sustainability: {sustained_effect:.2f}")
        
        # Return comprehensive results
        return {
            'efficacy': {
                'maximum': max_efficacy,
                'onset_speed': onset_speed,
                'sustainability': sustained_effect,
                'progression': efficacy_progression,
            },
            'safety': safety_profile,
            'recommendations': recommendations,
            'analysis_details': {
                'harmonic_modes_used': self.harmonic_modes,
                'simulation_steps': simulation_steps,
                'phase_coherence': self._calculate_phase_coherence(grid_data),
                'system_contributions': self._calculate_system_contributions(efficacy_progression)
            }
        }
    
    def _transform_compounds_to_phase_space(self, compounds: List[Dict]) -> Dict:
        """
        Transform medicinal compounds into phase-state representation.
        This is analogous to extracting elevation data in the original HRF.
        """
        phase_data = {}
        
        # Process each compound
        for compound in compounds:
            compound_id = compound.get('id', compound.get('name', ''))
            
            # Check if we already have this compound in cache
            if compound_id in self.compound_cache:
                phase_data[compound_id] = self.compound_cache[compound_id]
                continue
            
            # Handle different medical systems with their specific properties
            medical_system = compound.get('system', 'allopathy').lower()
            
            # Generate system-specific phase representation
            if medical_system == 'allopathy':
                # Phase representation for allopathic drugs - focuses on molecular interactions
                phase_representation = self._generate_allopathic_phases(compound)
            
            elif medical_system == 'ayurveda':
                # Phase representation for Ayurvedic herbs - focuses on dosha balancing
                phase_representation = self._generate_ayurvedic_phases(compound)
            
            elif medical_system == 'unani':
                # Phase representation for Unani medicines - focuses on temperament
                phase_representation = self._generate_unani_phases(compound)
            
            elif medical_system == 'homeopathy':
                # Phase representation for homeopathic remedies - focuses on energy patterns
                phase_representation = self._generate_homeopathic_phases(compound)
            
            else:
                # Default handling for unknown systems
                logger.warning(f"Unknown medical system: {medical_system}, using generic phase representation")
                phase_representation = self._generate_generic_phases(compound)
            
            # Store the phase representation
            phase_data[compound_id] = {
                'phase_angles': phase_representation['phase_angles'],
                'amplitudes': phase_representation['amplitudes'],
                'system': medical_system,
                'properties': compound.get('properties', {}),
                'original': compound
            }
            
            # Cache for future use
            self.compound_cache[compound_id] = phase_data[compound_id]
        
        logger.info(f"Transformed {len(compounds)} compounds to phase space representation")
        return phase_data
    
    def _generate_allopathic_phases(self, compound: Dict) -> Dict:
        """
        Generate phase representation for allopathic drugs.
        This focuses on receptor binding, enzyme inhibition, and other biochemical mechanisms.
        """
        # Number of phase dimensions to represent
        dimensions = 8
        
        # Initialize phase angles and amplitudes
        phase_angles = np.zeros(dimensions)
        amplitudes = np.zeros(dimensions)
        
        properties = compound.get('properties', {})
        
        # Half-life - affects duration of action
        half_life = properties.get('half_life_hours', 6)
        phase_angles[0] = np.mod(half_life / 6, 2*np.pi)
        amplitudes[0] = min(1.0, max(0.1, half_life / 24.0))
        
        # Receptor specificity - affects targeting precision
        specificity = properties.get('receptor_specificity', 0.5)
        phase_angles[1] = specificity * np.pi  # 0 = low specificity, π = high specificity
        amplitudes[1] = specificity
        
        # Side effect profile - affects safety
        side_effect_severity = properties.get('side_effect_severity', 0.3)
        phase_angles[2] = side_effect_severity * np.pi
        amplitudes[2] = 1.0 - side_effect_severity  # Invert so higher amplitude = safer
        
        # Speed of onset - how quickly it works
        onset_hours = properties.get('onset_hours', 1.0)
        phase_angles[3] = np.mod(onset_hours / 2, 2*np.pi)
        amplitudes[3] = 1.0 - min(1.0, onset_hours / 12.0)  # Faster onset = higher amplitude
        
        # Primary mechanism type
        mechanism_map = {
            'enzyme_inhibition': 0,
            'receptor_binding': 1,
            'channel_blocking': 2,
            'hormone_like': 3,
            'antibiotic': 4
        }
        mechanism = properties.get('mechanism', 'receptor_binding')
        mech_idx = mechanism_map.get(mechanism, 1)
        phase_angles[4] = mech_idx * np.pi / 3
        amplitudes[4] = 0.8
        
        # Overall potency
        potency = properties.get('potency', 0.7)
        phase_angles[5] = potency * np.pi
        amplitudes[5] = potency
        
        # Fill remaining dimensions
        for i in range(6, dimensions):
            phase_angles[i] = np.random.uniform(0, 2*np.pi)
            amplitudes[i] = np.random.uniform(0.3, 0.7)
        
        return {
            'phase_angles': phase_angles,
            'amplitudes': amplitudes
        }
    
    def _generate_ayurvedic_phases(self, compound: Dict) -> Dict:
        """
        Generate phase representation for Ayurvedic medicines.
        This focuses on dosha effects, rasa (taste), virya (potency), and other Ayurvedic concepts.
        """
        # Number of phase dimensions to represent
        dimensions = 8
        
        # Initialize phase angles and amplitudes
        phase_angles = np.zeros(dimensions)
        amplitudes = np.zeros(dimensions)
        
        properties = compound.get('properties', {})
        
        # Map Ayurvedic doshas to phase dimensions
        # Vata effect (0 = decreases, π = increases)
        vata_effect = properties.get('vata_effect', 0)  # -1 to 1 scale
        phase_angles[0] = (vata_effect + 1) * np.pi / 2  # Map to 0-π
        amplitudes[0] = abs(vata_effect)
        
        # Pitta effect
        pitta_effect = properties.get('pitta_effect', 0)
        phase_angles[1] = (pitta_effect + 1) * np.pi / 2
        amplitudes[1] = abs(pitta_effect)
        
        # Kapha effect
        kapha_effect = properties.get('kapha_effect', 0)
        phase_angles[2] = (kapha_effect + 1) * np.pi / 2
        amplitudes[2] = abs(kapha_effect)
        
        # Rasa (taste) - affects physiological response
        rasa_map = {
            'sweet': 0,
            'sour': 1,
            'salty': 2,
            'pungent': 3,
            'bitter': 4,
            'astringent': 5
        }
        
        primary_rasa = properties.get('primary_rasa', 'bitter')
        rasa_idx = rasa_map.get(primary_rasa, 4)  # Default to bitter
        phase_angles[3] = rasa_idx * np.pi / 3  # Divide circle into 6 tastes
        amplitudes[3] = 0.8  # Standard amplitude for taste
        
        # Virya (potency - hot/cold)
        virya = properties.get('virya', 0)  # -1 (cold) to 1 (hot)
        phase_angles[4] = (virya + 1) * np.pi / 2
        amplitudes[4] = 0.7
        
        # Overall potency
        potency = properties.get('potency', 0.6)
        phase_angles[5] = potency * np.pi
        amplitudes[5] = potency
        
        # Fill remaining dimensions
        for i in range(6, dimensions):
            phase_angles[i] = np.random.uniform(0, 2*np.pi)
            amplitudes[i] = np.random.uniform(0.3, 0.7)
        
        return {
            'phase_angles': phase_angles,
            'amplitudes': amplitudes
        }
    
    def _generate_unani_phases(self, compound: Dict) -> Dict:
        """
        Generate phase representation for Unani medicines.
        This focuses on temperament (mizaj), humors, and potency.
        """
        # Number of phase dimensions to represent
        dimensions = 8
        
        # Initialize phase angles and amplitudes
        phase_angles = np.zeros(dimensions)
        amplitudes = np.zeros(dimensions)
        
        properties = compound.get('properties', {})
        
        # Map Unani temperament (mizaj) to phase dimensions
        # Hot-Cold scale
        temperament_hot = properties.get('temperament_hot', 0)  # 0 to 4 scale
        phase_angles[0] = temperament_hot * np.pi / 4
        amplitudes[0] = temperament_hot / 4
        
        # Dry-Wet scale
        temperament_dry = properties.get('temperament_dry', 0)  # 0 to 4 scale
        phase_angles[1] = temperament_dry * np.pi / 4
        amplitudes[1] = temperament_dry / 4
        
        # Humor effects
        # Blood (Dam)
        blood_effect = properties.get('blood_effect', 0)  # -1 to 1 scale
        phase_angles[2] = (blood_effect + 1) * np.pi / 2
        amplitudes[2] = abs(blood_effect)
        
        # Phlegm (Balgham)
        phlegm_effect = properties.get('phlegm_effect', 0)
        phase_angles[3] = (phlegm_effect + 1) * np.pi / 2
        amplitudes[3] = abs(phlegm_effect)
        
        # Overall potency
        potency = properties.get('potency', 0.6)
        phase_angles[4] = potency * np.pi
        amplitudes[4] = potency
        
        # Fill remaining dimensions
        for i in range(5, dimensions):
            phase_angles[i] = np.random.uniform(0, 2*np.pi)
            amplitudes[i] = np.random.uniform(0.3, 0.7)
        
        return {
            'phase_angles': phase_angles,
            'amplitudes': amplitudes
        }
    
    def _generate_homeopathic_phases(self, compound: Dict) -> Dict:
        """
        Generate phase representation for homeopathic remedies.
        This focuses on energy patterns, dilution levels, and symptom matching.
        """
        # Number of phase dimensions to represent
        dimensions = 8
        
        # Initialize phase angles and amplitudes
        phase_angles = np.zeros(dimensions)
        amplitudes = np.zeros(dimensions)
        
        properties = compound.get('properties', {})
        
        # Map homeopathic properties to phase dimensions
        # Dilution potency (e.g., 6X, 30C, 200C)
        potency_scale = properties.get('potency_scale', 'C')
        potency_value = properties.get('potency_value', 30)
        
        # Convert to standardized potency factor
        if potency_scale == 'X':
            std_potency = potency_value / 10
        elif potency_scale == 'C':
            std_potency = potency_value
        elif potency_scale == 'M':
            std_potency = potency_value * 1000
        else:
            std_potency = potency_value
        
        # Map potency to phase angle and amplitude
        # Higher potencies are represented by greater phase shifts
        phase_angles[0] = min(std_potency, 200) * np.pi / 200
        # Higher potencies have higher amplitude in homeopathic theory
        amplitudes[0] = min(1.0, std_potency / 100)
        
        # Vital force impact (energy level effect)
        vital_force = properties.get('vital_force_impact', 0.5)  # 0 to 1
        phase_angles[1] = vital_force * np.pi
        amplitudes[1] = vital_force
        
        # Overall potency effect
        potency = properties.get('potency_effect', 0.4)
        phase_angles[2] = potency * np.pi
        amplitudes[2] = potency
        
        # Fill remaining dimensions
        for i in range(3, dimensions):
            phase_angles[i] = np.random.uniform(0, 2*np.pi)
            amplitudes[i] = np.random.uniform(0.3, 0.7)
        
        return {
            'phase_angles': phase_angles,
            'amplitudes': amplitudes
        }
    
    def _generate_generic_phases(self, compound: Dict) -> Dict:
        """
        Generate a generic phase representation for compounds without a specific system.
        Used as fallback when the medical system is unknown.
        """
        # Number of phase dimensions to represent
        dimensions = 8
        
        # Initialize phase angles and amplitudes
        phase_angles = np.zeros(dimensions)
        amplitudes = np.zeros(dimensions)
        
        properties = compound.get('properties', {})
        
        # Generate based on general medicinal properties
        # Efficacy (claimed or known)
        efficacy = properties.get('efficacy', 0.5)  # 0 to 1
        phase_angles[0] = efficacy * np.pi
        amplitudes[0] = efficacy
        
        # Onset speed
        onset_speed = properties.get('onset_speed', 0.5)  # 0 to 1
        phase_angles[1] = onset_speed * np.pi
        amplitudes[1] = onset_speed
        
        # Fill remaining dimensions with random but consistent values
        # Use compound name or ID to seed the random generator for consistency
        compound_id = compound.get('id', compound.get('name', ''))
        seed = sum(ord(c) for c in compound_id) if compound_id else 42
        rng = np.random.RandomState(seed)
        
        for i in range(2, dimensions):
            phase_angles[i] = rng.uniform(0, 2*np.pi)
            amplitudes[i] = rng.uniform(0.3, 0.7)
        
        return {
            'phase_angles': phase_angles,
            'amplitudes': amplitudes
        }
    
    def _prepare_medicinal_property_grid(self, compound_phases: Dict) -> Dict:
        """
        Create computational grids for medicinal properties.
        """
        # Define grid dimensions
        # Use a 2D grid to represent the therapeutic property space
        grid_size = 16  # Size of grid (16x16 for PoC)
        
        # Initialize grid arrays for different medicinal properties
        efficacy_grid = np.zeros((grid_size, grid_size))
        toxicity_grid = np.zeros((grid_size, grid_size))
        onset_grid = np.zeros((grid_size, grid_size))
        duration_grid = np.zeros((grid_size, grid_size))
        
        # System contribution grids
        system_grids = {
            'allopathy': np.zeros((grid_size, grid_size)),
            'ayurveda': np.zeros((grid_size, grid_size)),
            'unani': np.zeros((grid_size, grid_size)),
            'homeopathy': np.zeros((grid_size, grid_size))
        }
        
        # Create mapping between compound and grid position
        compound_positions = {}
        
        # Layout compounds in the grid based on their phase relationships
        for idx, (compound_id, phase_data) in enumerate(compound_phases.items()):
            # Use phase angles to determine position in grid
            # Use first two phase dimensions for x,y positioning
            phase_x = phase_data['phase_angles'][0] / (2*np.pi)
            phase_y = phase_data['phase_angles'][1] / (2*np.pi)
            
            # Convert to grid coordinates
            grid_x = int(phase_x * (grid_size-1))
            grid_y = int(phase_y * (grid_size-1))
            
            # Store the grid position
            compound_positions[compound_id] = (grid_x, grid_y)
            
            # Calculate influence radius based on amplitude
            avg_amplitude = np.mean(phase_data['amplitudes'])
            influence_radius = int(max(2, avg_amplitude * grid_size * 0.2))
            
            # Apply compound influence to grids
            for x in range(max(0, grid_x-influence_radius), min(grid_size, grid_x+influence_radius+1)):
                for y in range(max(0, grid_y-influence_radius), min(grid_size, grid_y+influence_radius+1)):
                    # Calculate distance-based falloff
                    distance = np.sqrt((x - grid_x)**2 + (y - grid_y)**2)
                    if distance > influence_radius:
                        continue
                    
                    # Calculate influence factor (decreases with distance)
                    influence = (1 - distance/influence_radius) * avg_amplitude
                    
                    # Apply to appropriate property grids
                    system = phase_data['system']
                    properties = phase_data['original'].get('properties', {})
                    
                    # Extract property values with defaults
                    efficacy_val = properties.get('efficacy', 0.7) * influence
                    toxicity_val = properties.get('toxicity', 0.3) * influence
                    onset_val = properties.get('onset_speed', 0.5) * influence
                    duration_val = properties.get('duration_factor', 0.6) * influence
                    
                    # Update property grids
                    efficacy_grid[x, y] += efficacy_val
                    toxicity_grid[x, y] += toxicity_val
                    onset_grid[x, y] += onset_val
                    duration_grid[x, y] += duration_val
                    
                    # Update system contribution grid
                    if system in system_grids:
                        system_grids[system][x, y] += influence * self.system_weights.get(system, 0.25)
        
        # Normalize grids
        max_efficacy = np.max(efficacy_grid) if np.max(efficacy_grid) > 0 else 1.0
        max_toxicity = np.max(toxicity_grid) if np.max(toxicity_grid) > 0 else 1.0
        max_onset = np.max(onset_grid) if np.max(onset_grid) > 0 else 1.0
        max_duration = np.max(duration_grid) if np.max(duration_grid) > 0 else 1.0
        
        efficacy_grid = efficacy_grid / max_efficacy
        toxicity_grid = toxicity_grid / max_toxicity
        onset_grid = onset_grid / max_onset
        duration_grid = duration_grid / max_duration
        
        # Normalize system grids
        for system in system_grids:
            max_val = np.max(system_grids[system]) if np.max(system_grids[system]) > 0 else 1.0
            system_grids[system] = system_grids[system] / max_val
        
        # Calculate combined therapeutic index (efficacy/toxicity ratio)
        therapeutic_index = np.zeros((grid_size, grid_size))
        for x in range(grid_size):
            for y in range(grid_size):
                if toxicity_grid[x, y] > 0:
                    therapeutic_index[x, y] = efficacy_grid[x, y] / max(0.1, toxicity_grid[x, y])
                else:
                    therapeutic_index[x, y] = efficacy_grid[x, y] * 10  # Arbitrary high value
        
        # Cap the therapeutic index for numerical stability
        therapeutic_index = np.minimum(therapeutic_index, 10.0)
        
        return {
            'efficacy': efficacy_grid,
            'toxicity': toxicity_grid,
            'onset': onset_grid,
            'duration': duration_grid,
            'therapeutic_index': therapeutic_index,
            'system_grids': system_grids,
            'compound_positions': compound_positions,
            'shape': (grid_size, grid_size)
        }
    
    def _create_medicinal_basis_functions(self, rows: int, cols: int) -> List[np.ndarray]:
        """
        Create harmonic basis functions for medicinal property representation.
        """
        basis = []
        
        # 1. Uniform function (constant efficacy level everywhere)
        uniform = np.ones((rows, cols))
        basis.append(uniform / np.sqrt(np.sum(uniform**2) + 1e-10))
        
        # 2. Gradient basis functions (representing directional efficacy changes)
        # West-East gradient
        we_gradient = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                we_gradient[r, c] = c / (cols - 1)
        basis.append(we_gradient / np.sqrt(np.sum(we_gradient**2) + 1e-10))
        
        # North-South gradient
        ns_gradient = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                ns_gradient[r, c] = r / (rows - 1)
        basis.append(ns_gradient / np.sqrt(np.sum(ns_gradient**2) + 1e-10))
        
        # 3. Radial patterns (representing central or localized effects)
        center_pattern = np.zeros((rows, cols))
        center_r, center_c = rows // 2, cols // 2
        for r in range(rows):
            for c in range(cols):
                dist_r = (r - center_r) / max(1, rows // 2)
                dist_c = (c - center_c) / max(1, cols // 2)
                dist = np.sqrt(dist_r**2 + dist_c**2)
                center_pattern[r, c] = 1.0 - min(1.0, dist * 1.5)
        basis.append(center_pattern / np.sqrt(np.sum(center_pattern**2) + 1e-10))
        
        # 4. Sinusoidal patterns for more complex efficacy distribution
        for k in range(1, min(2, self.harmonic_modes // 2)):
            # Sine pattern in X direction
            sin_x = np.zeros((rows, cols))
            for r in range(rows):
                for c in range(cols):
                    sin_x[r, c] = np.sin(np.pi * k * c / (cols - 1))
            basis.append(sin_x / np.sqrt(np.sum(sin_x**2) + 1e-10))
            
            # Sine pattern in Y direction
            sin_y = np.zeros((rows, cols))
            for r in range(rows):
                for c in range(cols):
                    sin_y[r, c] = np.sin(np.pi * k * r / (rows - 1))
            basis.append(sin_y / np.sqrt(np.sum(sin_y**2) + 1e-10))
        
        # Limit to the number of modes we want
        return basis[:self.harmonic_modes]
    
    def _project_to_basis(self, field: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
        """
        Project the efficacy field onto the harmonic basis functions.
        """
        # Initialize coefficient array [amplitude, phase]
        coeffs = np.zeros((len(basis), 2))
        
        # For each basis function, calculate projection
        for i, basis_func in enumerate(basis):
            # Calculate inner product (projection)
            projection = np.sum(field * basis_func)
            
            # Store as amplitude (magnitude of projection)
            coeffs[i, 0] = projection
            
            # Initial phase is 0
            coeffs[i, 1] = 0.0
            
        return coeffs
    
    def _run_harmonic_treatment_simulation(self, grid_data: Dict, num_steps: int = 30) -> Dict:
        """
        Run a harmonic simulation of treatment efficacy over time.
        """
        grid_size = grid_data['shape'][0]
        
        # Initialize time series arrays for tracking efficacy progression
        time_series = {
            'overall': np.zeros(num_steps),
            'allopathy': np.zeros(num_steps),
            'ayurveda': np.zeros(num_steps),
            'unani': np.zeros(num_steps),
            'homeopathy': np.zeros(num_steps)
        }
        
        # Create basis functions for the simulation
        basis_functions = self._create_medicinal_basis_functions(grid_size, grid_size)
        
        # Project the initial state to the basis
        initial_state = grid_data['efficacy'].copy()
        coefficients = self._project_to_basis(initial_state, basis_functions)
        
        # Get system contribution matrices
        system_grids = grid_data['system_grids']
        
        # Simulate over time steps
        for step in range(num_steps):
            # Evolve the coefficients based on medical properties
            coefficients = self._evolve_medicinal_coefficients(
                coefficients,
                grid_data['therapeutic_index'],
                grid_data['onset'],
                grid_data['duration'],
                step,
                num_steps
            )
            
            # Reconstruct the efficacy state
            efficacy_state = self._reconstruct_from_basis(coefficients, basis_functions)
            
            # Ensure non-negative values
            efficacy_state = np.maximum(0.0, efficacy_state)
            
            # Calculate overall efficacy for this time step
            overall_efficacy = np.mean(efficacy_state)
            time_series['overall'][step] = overall_efficacy
            
            # Calculate contribution from each medical system
            for system in time_series.keys():
                if system != 'overall' and system in system_grids:
                    # Calculate system contribution to efficacy
                    system_contribution = np.mean(efficacy_state * system_grids[system])
                    time_series[system][step] = system_contribution
            
            # Log progress at intervals
            if step % max(1, num_steps // 5) == 0:
                logger.info(f"Simulation step {step+1}/{num_steps}: "
                          f"Efficacy={overall_efficacy:.4f}")
        
        return time_series
    
    def _evolve_medicinal_coefficients(self, coeffs: np.ndarray, therapeutic_index: np.ndarray,
                                   onset_grid: np.ndarray, duration_grid: np.ndarray,
                                   current_step: int, total_steps: int) -> np.ndarray:
        """
        Evolve the coefficients over time to simulate treatment efficacy changes.
        """
        # Create a copy for evolved coefficients
        new_coeffs = coeffs.copy()
        
        # Calculate time-dependent factors
        time_progress = current_step / max(1, total_steps - 1)  # 0 to 1
        
        # Calculate mean therapeutic properties
        mean_index = np.mean(therapeutic_index)
        mean_onset = np.mean(onset_grid)
        mean_duration = np.mean(duration_grid)
        
        # Calculate system-specific evolution factors
        # Fast-acting (allopathic) factor - peaks early
        allopathic_factor = self.system_weights['allopathy'] * (1.0 - time_progress)**2
        
        # Slow-building (ayurvedic) factor - builds up gradually
        ayurvedic_factor = self.system_weights['ayurveda'] * (1.0 - np.exp(-3 * time_progress))
        
        # Medium (unani) factor - peaks in the middle
        unani_factor = self.system_weights['unani'] * 4 * time_progress * (1.0 - time_progress)
        
        # Subtle (homeopathic) factor - consistent but gentle
        homeopathic_factor = self.system_weights['homeopathy'] * 0.5
        
        # Combined evolution factor
        combined_factor = allopathic_factor + ayurvedic_factor + unani_factor + homeopathic_factor
        
        # Safety factor to prevent numerical instability
        stability_factor = min(0.5, combined_factor)
        
        # Apply evolution to each mode
        for i in range(len(coeffs)):
            # Mode-specific evolution factors
            # Lower modes (fundamental patterns) evolve differently than higher modes (details)
            mode_factor = stability_factor * (1.0 if i < 3 else (1.0 - 0.1 * i))
            
            # Phase evolution - represents therapy progression
            phase_change = min(0.3, mode_factor * (1.0 + 0.1 * i))
            new_coeffs[i, 1] = coeffs[i, 1] + phase_change
            
            # Amplitude evolution - depends on treatment phase
            if time_progress < 0.3:  # Early phase - onset dominates
                # Onset-dependent growth
                growth = mean_onset * (0.3 - time_progress) * 2.0
                damping = np.exp(-0.1 * i * time_progress)  # Mode-specific damping
                new_coeffs[i, 0] = coeffs[i, 0] * (1.0 + growth) * damping
                
            elif time_progress < 0.7:  # Middle phase - therapeutic index dominates
                # Efficacy-dependent evolution
                efficacy_factor = 1.0 + (mean_index - 1.0) * 0.1
                new_coeffs[i, 0] = coeffs[i, 0] * efficacy_factor
                
            else:  # Late phase - duration dominates
                # Duration-dependent decay or persistence
                decay = 1.0 - (1.0 - mean_duration) * 0.2 * (time_progress - 0.7) / 0.3
                new_coeffs[i, 0] = coeffs[i, 0] * decay
            
            # Apply stability constraints - limit growth per step
            max_change_factor = 1.3  # Maximum 30% increase per step
            new_coeffs[i, 0] = min(new_coeffs[i, 0], coeffs[i, 0] * max_change_factor)
            
            # Ensure non-negative amplitude
            new_coeffs[i, 0] = max(0.0, new_coeffs[i, 0])
        
        return new_coeffs
    
    def _reconstruct_from_basis(self, coeffs: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct the efficacy field from harmonic coefficients.
        """
        # Get the shape from the first basis function
        result = np.zeros_like(basis[0])
        
        # Add each mode's contribution
        for i, basis_func in enumerate(basis):
            if i < len(coeffs):
                amplitude = coeffs[i, 0]
                phase = coeffs[i, 1]
                
                # Apply phase shift
                shifted_mode = basis_func * np.cos(phase)
                
                # Add to result
                result += amplitude * shifted_mode
        
        # Ensure non-negative values
        return np.maximum(0.0, result)
    
    def _calculate_safety_profile(self, grid_data: Dict, compound_phases: Dict) -> Dict:
        """
        Calculate the safety profile of the compound combination.
        """
        # Extract relevant grids
        efficacy_grid = grid_data['efficacy']
        toxicity_grid = grid_data['toxicity']
        therapeutic_index = grid_data['therapeutic_index']
        
        # Calculate overall safety metrics
        mean_toxicity = np.mean(toxicity_grid)
        max_toxicity = np.max(toxicity_grid)
        therapeutic_ratio = np.mean(therapeutic_index)
        
        # Calculate interaction risk
        interaction_risk = self._calculate_compound_interactions(compound_phases)
        
        # Calculate system-specific safety profiles
        system_safety = {}
        for system, grid in grid_data['system_grids'].items():
            # Calculate weighted toxicity for this system
            system_toxicity = np.sum(toxicity_grid * grid) / np.sum(grid) if np.sum(grid) > 0 else 0
            
            # Calculate relative safety percentage (0-100%)
            if max_toxicity > 0:
                relative_safety = 100 * (1 - system_toxicity / max_toxicity)
            else:
                relative_safety = 100
                
            system_safety[system] = {
                'toxicity_level': system_toxicity,
                'relative_safety': relative_safety,
                'risk_level': 'Low' if relative_safety > 80 else 
                              'Medium' if relative_safety > 60 else 'High'
            }
        
        # Identify main risk factors
        risk_factors = []
        
        if mean_toxicity > 0.5:
            risk_factors.append("High overall toxicity")
        
        if therapeutic_ratio < 2.0:
            risk_factors.append("Low therapeutic index")
            
        if interaction_risk > 0.3:
            risk_factors.append("Significant compound interaction risks")
        
        # Create detailed safety profile
        return {
            'overall_safety_score': 100 * (1 - mean_toxicity),
            'therapeutic_index': therapeutic_ratio,
            'interaction_risk': interaction_risk * 100,  # Convert to percentage
            'risk_factors': risk_factors,
            'system_safety': system_safety,
            'risk_level': 'Low' if mean_toxicity < 0.3 else 
                         'Medium' if mean_toxicity < 0.6 else 'High'
        }
    
    def _calculate_compound_interactions(self, compound_phases: Dict) -> float:
        """
        Calculate the risk of adverse interactions between compounds.
        """
        if len(compound_phases) < 2:
            return 0.0
            
        # Calculate pairwise phase interactions
        interaction_values = []
        
        compounds = list(compound_phases.keys())
        
        for i in range(len(compounds)):
            for j in range(i+1, len(compounds)):
                comp1 = compounds[i]
                comp2 = compounds[j]
                
                # Get phase data
                phase1 = compound_phases[comp1]['phase_angles']
                phase2 = compound_phases[comp2]['phase_angles']
                
                amp1 = compound_phases[comp1]['amplitudes']
                amp2 = compound_phases[comp2]['amplitudes']
                
                # Calculate phase differences (in radians)
                phase_diffs = np.abs(np.mod(phase1 - phase2 + np.pi, 2*np.pi) - np.pi)
                
                # Calculate weighted interaction risk
                # Phase differences near π indicate opposing actions (higher risk)
                # Phase differences near 0 indicate synergistic actions (lower risk)
                weighted_diffs = phase_diffs * amp1 * amp2
                
                # Normalize to 0-1 range (0 = no interaction, 1 = maximum interaction)
                avg_diff = np.sum(weighted_diffs) / np.sum(amp1 * amp2) if np.sum(amp1 * amp2) > 0 else 0
                normalized_risk = avg_diff / np.pi
                
                # Add system-specific risk factors
                system1 = compound_phases[comp1]['system']
                system2 = compound_phases[comp2]['system']
                
                # Higher risk for combinations across different systems
                if system1 != system2:
                    # Calculate cross-system risk factor
                    # Allopathy + traditional medicine often has higher interaction risk
                    if 'allopathy' in (system1, system2):
                        normalized_risk *= 1.3  # 30% risk increase
                
                interaction_values.append(normalized_risk)
        
        # Calculate overall interaction risk
        if interaction_values:
            return sum(interaction_values) / len(interaction_values)
        else:
            return 0.0
    
    def _calculate_onset_speed(self, efficacy_progression: np.ndarray) -> float:
        """
        Calculate the onset speed from efficacy progression.
        """
        if len(efficacy_progression) < 2:
            return 0.0
            
        # Find the time to reach 50% of maximum efficacy
        max_efficacy = np.max(efficacy_progression)
        if max_efficacy <= 0:
            return 0.0
            
        half_max = max_efficacy * 0.5
        
        # Find first time step that exceeds half-max
        for i, eff in enumerate(efficacy_progression):
            if eff >= half_max:
                time_to_half_max = i / (len(efficacy_progression) - 1)
                # Convert to onset speed (0-1 scale, higher = faster onset)
                return 1.0 - time_to_half_max
                
        return 0.0  # Never reaches half-max
    
    def _calculate_sustained_effect(self, efficacy_progression: np.ndarray) -> float:
        """
        Calculate how well the efficacy is sustained over time.
        """
        if len(efficacy_progression) < 3:
            return 0.0
            
        # Focus on the latter half of the curve
        midpoint = len(efficacy_progression) // 2
        latter_half = efficacy_progression[midpoint:]
        
        # Calculate the maximum efficacy
        max_efficacy = np.max(efficacy_progression)
        if max_efficacy <= 0:
            return 0.0
            
        # Calculate the average value in latter half relative to maximum
        avg_latter = np.mean(latter_half)
        sustainability = avg_latter / max_efficacy
        
        return sustainability
    
    def _calculate_phase_coherence(self, grid_data: Dict) -> float:
        """
        Calculate the phase coherence, representing how well the different
        compounds work together in harmony.
        """
        # Use efficacy and toxicity grids to calculate coherence
        efficacy = grid_data['efficacy']
        toxicity = grid_data['toxicity']
        
        # Calculate the variance of the therapeutic index
        # Lower variance = higher coherence
        therapeutic_index = grid_data['therapeutic_index']
        variance = np.var(therapeutic_index)
        
        # Calculate normalized coherence (0-1 scale)
        max_expected_variance = 10.0  # Based on typical therapeutic index ranges
        coherence = 1.0 - min(1.0, variance / max_expected_variance)
        
        # Adjust for phase coherence factor
        return coherence * self.phase_coherence_factor
    
    def _calculate_system_contributions(self, efficacy_progression: Dict) -> Dict:
        """
        Calculate the relative contribution of each medical system.
        """
        # Extract final efficacy values
        final_values = {
            system: progression[-1] 
            for system, progression in efficacy_progression.items()
            if system != 'overall'
        }
        
        # Calculate total contribution
        total = sum(final_values.values())
        
        # Calculate percentages
        if total > 0:
            return {system: (value * 100 / total) for system, value in final_values.items()}
        else:
            # Equal distribution if no contribution
            systems = [s for s in final_values.keys()]
            return {system: 100 / len(systems) for system in systems}
    
    def _generate_optimization_recommendations(self, grid_data: Dict, 
                                           efficacy_progression: Dict,
                                           safety_profile: Dict,
                                           compounds: List[Dict]) -> List[Dict]:
        """
        Generate recommendations for optimizing the compound combination.
        """
        recommendations = []
        
        # 1. Analyze overall efficacy trend
        overall_prog = efficacy_progression['overall']
        max_efficacy = np.max(overall_prog)
        final_efficacy = overall_prog[-1]
        
        # Check for efficacy drop-off
        if final_efficacy < max_efficacy * 0.7:
            recommendations.append({
                'type': 'dosage_adjustment',
                'priority': 'high',
                'issue': 'Efficacy drop-off over time',
                'recommendation': 'Consider adding a time-release component from Ayurvedic tradition',
                'details': f'Efficacy drops to {final_efficacy/max_efficacy*100:.1f}% of maximum by the end of treatment'
            })
        
        # 2. Analyze system contributions
        system_contributions = self._calculate_system_contributions(efficacy_progression)
        
        # Find dominant and underrepresented systems
        max_contrib = max(system_contributions.values())
        min_contrib = min(system_contributions.values())
        
        dominant_system = [s for s, v in system_contributions.items() if v == max_contrib][0]
        underrep_system = [s for s, v in system_contributions.items() if v == min_contrib][0]
        
        # Check for imbalance
        if max_contrib > 50 and min_contrib < 10:
            recommendations.append({
                'type': 'system_balance',
                'priority': 'medium',
                'issue': f'Imbalance between medical systems',
                'recommendation': f'Increase {underrep_system} components to improve balance',
                'details': f'{dominant_system} currently dominates ({max_contrib:.1f}%), while {underrep_system} contributes only {min_contrib:.1f}%'
            })
        
        # 3. Safety-based recommendations
        if safety_profile['interaction_risk'] > 40:
            recommendations.append({
                'type': 'interaction_mitigation',
                'priority': 'high',
                'issue': 'High compound interaction risk',
                'recommendation': 'Introduce buffer compounds or adjust dosages',
                'details': f'Interaction risk of {safety_profile["interaction_risk"]:.1f}% exceeds recommended threshold'
            })
        
        return recommendations
    
    def visualize_efficacy_progression(self, efficacy_progression: Dict, 
                                     title: str = "Treatment Efficacy Over Time"):
        """
        Generate a visualization of efficacy progression over time.
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot overall efficacy
        time_steps = range(len(efficacy_progression['overall']))
        plt.plot(time_steps, efficacy_progression['overall'], 'k-', linewidth=3, label='Overall')
        
        # Plot system contributions
        colors = {
            'allopathy': 'blue',
            'ayurveda': 'green',
            'unani': 'purple',
            'homeopathy': 'orange'
        }
        
        for system, progression in efficacy_progression.items():
            if system != 'overall' and system in colors:
                plt.plot(time_steps, progression, color=colors[system], linewidth=2, label=system.capitalize())
        
        # Add annotations for key points
        max_idx = np.argmax(efficacy_progression['overall'])
        max_val = efficacy_progression['overall'][max_idx]
        plt.scatter(max_idx, max_val, color='red', s=100, zorder=5)
        plt.annotate(f'Max: {max_val:.2f}', (max_idx, max_val), 
                    textcoords="offset points", xytext=(5,10), ha='center')
        
        # Add labels and legend
        plt.xlabel('Time Steps')
        plt.ylabel('Efficacy')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt

# Example usage - Proof of Concept

def run_poc():
    """
    Run a Proof of Concept for the Medicinal Harmonic Resonance Framework.
    """
    print("Initializing Medicinal Harmonic Resonance Framework...")
    
    # Create framework instance
    hrf = MedicinalHarmonicResonanceFramework()
    
    # Sample data for Diabetes treatment
    # 1. Sample compounds from different medical systems
    
    # Allopathic drug
    metformin = {
        'name': 'Metformin',
        'system': 'allopathy',
        'properties': {
            'half_life_hours': 6.2,
            'receptor_specificity': 0.8,
            'side_effect_severity': 0.3,
            'onset_hours': 2.0,
            'mechanism': 'enzyme_inhibition',
            'potency': 0.75,
            'efficacy': 0.82,
            'toxicity': 0.25,
            'duration_factor': 0.7
        }
    }
    
    # Ayurvedic herb
    gymnema = {
        'name': 'Gymnema Sylvestre',
        'system': 'ayurveda',
        'properties': {
            'vata_effect': -0.2,
            'pitta_effect': -0.4,
            'kapha_effect': -0.7,
            'primary_rasa': 'bitter',
            'virya': -0.2,  # Slightly cooling
            'potency': 0.65,
            'efficacy': 0.68,
            'toxicity': 0.1,
            'onset_speed': 0.3,
            'duration_factor': 0.85
        }
    }
    
    # Unani herb
    fenugreek = {
        'name': 'Fenugreek',
        'system': 'unani',
        'properties': {
            'temperament_hot': 1,
            'temperament_dry': 1,
            'blood_effect': 0.3,
            'phlegm_effect': -0.4,
            'potency': 0.6,
            'efficacy': 0.55,
            'toxicity': 0.08,
            'onset_speed': 0.4,
            'duration_factor': 0.75
        }
    }
    
    # Homeopathic remedy
    uranium_nitricum = {
        'name': 'Uranium Nitricum',
        'system': 'homeopathy',
        'properties': {
            'potency_scale': 'C',
            'potency_value': 30,
            'vital_force_impact': 0.6,
            'potency_effect': 0.45,
            'efficacy': 0.4,
            'toxicity': 0.05,
            'onset_speed': 0.2,
            'duration_factor': 0.9
        }
    }
    
    # Define the compound combination to analyze
    compounds = [metformin, gymnema, fenugreek, uranium_nitricum]
    
    # Sample patient profile
    patient = {
        'age': 55,
        'gender': 'female',
        'weight_kg': 72,
        'chronic_conditions': ['type2_diabetes', 'hypertension'],
        'genetic_markers': {
            'MTHFR': 'heterozygous',
            'APOE': 'E3/E4'
        }
    }
    
    print(f"Analyzing combination of {len(compounds)} compounds for diabetes treatment...")
    
    # Run the analysis
    results = hrf.analyze_compound_combination(
        compounds=compounds,
        target_disease='type2_diabetes',
        patient_profile=patient,
        simulation_steps=30
    )
    
    # Print key results
    print("\n--- ANALYSIS RESULTS ---")
    print(f"Maximum Efficacy: {results['efficacy']['maximum']:.2f}")
    print(f"Onset Speed: {results['efficacy']['onset_speed']:.2f} (higher is faster)")
    print(f"Sustainability: {results['efficacy']['sustainability']:.2f} (higher is more sustained)")
    print(f"Overall Safety Score: {results['safety']['overall_safety_score']:.1f}%")
    print(f"Therapeutic Index: {results['safety']['therapeutic_index']:.2f}")
    print(f"Interaction Risk: {results['safety']['interaction_risk']:.1f}%")
    print(f"Risk Level: {results['safety']['risk_level']}")
    
    print("\nSystem Contributions:")
    for system, contribution in results['analysis_details']['system_contributions'].items():
        print(f"  - {system.capitalize()}: {contribution:.1f}%")
    
    print("\nRecommendations:")
    for i, rec in enumerate(results['recommendations']):
        print(f"{i+1}. [{rec['priority'].upper()}] {rec['issue']}")
        print(f"   Recommendation: {rec['recommendation']}")
        print(f"   Details: {rec['details']}")
    
    # Create visualizations
    plt.figure(1)
    hrf.visualize_efficacy_progression(results['efficacy']['progression'], 
                                     "Diabetes Treatment Efficacy Over Time")
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    run_poc()