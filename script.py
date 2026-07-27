import csv
import random

num_samples = 125000
filename = "ultimate_biomechanics_data.csv"

# Comprehensive feature set
fields = [
    'hip_shoulder_sep_deg',      # Torsional power
    'lead_knee_flexion_deg',     # Stability
    'drive_knee_flexion_deg',    # Power generation
    'torso_lean_deg',            # Axis maintenance
    'shoulder_elbow_angle_deg',  # Arm slot accuracy
    'wrist_snap_range_deg',      # Final release velocity
    'stance_width_ratio',        # Base stability
    'release_height_ratio',      # Release consistency
    'label'                      # 1 = Good, 0 = Poor
]

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(fields)
    
    for _ in range(num_samples):
        is_good = random.random() > 0.35 # 65% of data is "good"
        
        if is_good:
            # Elite form: stable, coiled, and efficient
            row = [
                random.uniform(30, 50),   # Strong separation
                random.uniform(20, 35),   # Balanced lead knee
                random.uniform(40, 65),   # Deep drive knee
                random.uniform(-5, 5),    # Vertical axis
                random.uniform(85, 100),  # Consistent arm slot
                random.uniform(15, 30),   # Good snap
                random.uniform(1.3, 1.6), # Wide stance
                random.uniform(0.4, 0.6), # Mid-height release
                1
            ]
        else:
            # Poor form: stiff, leaning, or imbalanced
            row = [
                random.uniform(0, 15),    # No separation (arm-only throw)
                random.uniform(0, 10),    # Stiff lead leg
                random.uniform(0, 15),    # No drive power
                random.uniform(15, 45),   # Extreme lean (unstable)
                random.uniform(120, 170), # Locked/straight arm
                random.uniform(0, 10),    # No snap
                random.uniform(0.5, 1.0), # Too narrow (tipsy)
                random.uniform(0.7, 0.9), # High/uncontrolled release
                0
            ]
        writer.writerow(row)

print(f"Generated {num_samples} samples with {len(fields)-1} features in {filename}.")