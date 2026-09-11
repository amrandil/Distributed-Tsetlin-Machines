"""
Unit test for a single Tsetlin Automaton to verify penalty behavior.
"""

from cakit.automata import TsetlinAutomaton

# Create a TA with 4 states per arm (total 8 states)
# Arm 0: states {1, 2, 3, 4}
# Arm 1: states {5, 6, 7, 8}
ta = TsetlinAutomaton(n_states=4, initial_state=4)  # Start deep in arm 0

print("=" * 60)
print("Testing Tsetlin Automaton Penalty Behavior")
print("=" * 60)
print(f"\nConfiguration: {ta.n_states} states per arm")
print(f"Arm 0: states {{1, 2, 3, 4}}")
print(f"Arm 1: states {{5, 6, 7, 8}}")
print(f"\nStarting at state {ta.get_state()} (deep in arm {ta.get_arm()})")
print("\nApplying continuous penalties...\n")

# Apply penalties and observe
for i in range(10):
    state = ta.get_state()
    arm = ta.get_arm()
    position = ta.get_position()
    output = ta.get_output()
    
    print(f"Step {i}: state={state}, arm={arm}, position={position}, output={output}")
    
    # Apply penalty
    ta.penalize()
    
    new_state = ta.get_state()
    if new_state == state:
        print(f"         → Penalty had NO EFFECT (stuck at state {state})")
        if arm == 0:
            print(f"         → ⚠️  STUCK AT BOUNDARY - CANNOT SWITCH TO ARM 1!")
        break
    else:
        print(f"         → Penalty: state {state} → {new_state}")

print("\n" + "=" * 60)
print("Expected behavior: Should cross from state 1 → into arm 1")
print("=" * 60)
