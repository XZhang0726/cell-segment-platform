"""
Test Dempster-Shafer evidence fusion

This script checks the core Dempster-Shafer fusion functionality:
1. DST module imports
2. Mass function calculation
3. Dempster's combination rule
4. Fusion result validation
"""

import numpy as np
from src.core.fusion import DempsterShaferFusion, FusionResult

def test_dst_import():
    """Test 1: Verify DST modules can be imported successfully"""
    print("=" * 60)
    print("Test 1: DST module imports")
    print("=" * 60)

    try:
        from src.core.fusion import (
            DempsterShaferFusion,
            FusionResult,
            handle_conflict,
            generate_conflict_map
        )
        print("[PASS] DST module import succeeded")
        return True
    except Exception as e:
        print(f"[FAIL] DST module import failed: {e}")
        return False

def test_mass_function():
    """Test 2: Verify the mass function calculation"""
    print("\n" + "=" * 60)
    print("Test 2: Mass function calculation")
    print("=" * 60)

    try:
        # Initialize the DST fusion engine
        fusion_engine = DempsterShaferFusion({
            'cellpose': 0.9,
            'cellvit': 0.85,
            'cellsam': 0.8
        })

        # Test mass function calculation
        mass = fusion_engine.compute_mass_from_confidence('cellpose', 0.8)

        print(f"Model: cellpose, confidence: 0.8, reliability: 0.9")
        print(f"Mass function: {mass}")

        # Verify that the mass function sums to 1
        total = sum(mass.values())
        print(f"Mass function sum: {total:.6f}")

        if abs(total - 1.0) < 1e-6:
            print("[PASS] Mass function calculation verified")
            return True
        else:
            print(f"[FAIL] The mass function sum is not 1: {total}")
            return False

    except Exception as e:
        print(f"[FAIL] Mass function calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dempster_combination():
    """Test 3: Verify Dempster's combination rule"""
    print("\n" + "=" * 60)
    print("Test 3: Dempster's combination rule")
    print("=" * 60)

    try:
        fusion_engine = DempsterShaferFusion({
            'model1': 0.9,
            'model2': 0.85
        })

        # Define two mass functions
        m1 = {'Cell': 0.7, 'Background': 0.2, 'Cell|Background': 0.1}
        m2 = {'Cell': 0.6, 'Background': 0.3, 'Cell|Background': 0.1}

        print(f"Mass function 1: {m1}")
        print(f"Mass function 2: {m2}")

        # Combine the mass functions
        combined, conflict = fusion_engine.dempster_combine(m1, m2)

        print(f"\nCombination result: {combined}")
        print(f"Conflict coefficient: {conflict:.3f}")

        # Verify that the combined mass sums to 1
        total = sum(combined.values())
        print(f"Sum of the combined mass function: {total:.6f}")

        if abs(total - 1.0) < 1e-6 and 0 <= conflict < 1:
            print("[PASS] Dempster's combination rule verified")
            return True
        else:
            print(f"[FAIL] Unexpected combination result")
            return False

    except Exception as e:
        print(f"[FAIL] Dempster combination failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instance_fusion():
    """Test 4: Verify instance fusion"""
    print("\n" + "=" * 60)
    print("Test 4: Instance fusion")
    print("=" * 60)

    try:
        fusion_engine = DempsterShaferFusion({
            'cellpose': 0.9,
            'cellvit': 0.85,
            'cellsam': 0.8
        })

        # Simulate predictions from three models
        matched_group = [
            ('cellpose', 0.8),
            ('cellvit', 0.6),
            ('cellsam', 0.9)
        ]

        print(f"Matched group: {matched_group}")

        # Run fusion
        result = fusion_engine.fuse_instances(matched_group)

        print(f"\nFusion result:")
        print(f"  Decision: {result.decision}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Conflict: {result.conflict:.3f}")
        print(f"  Uncertainty: {result.uncertainty:.3f}")
        print(f"  Belief-plausibility interval: [{result.belief_cell:.3f}, {result.plausibility_cell:.3f}]")

        # Validate the result
        if (result.decision in ['Cell', 'Background', 'Cell|Background'] and
            0 <= result.confidence <= 1 and
            0 <= result.conflict < 1 and
            0 <= result.uncertainty <= 1):
            print("[PASS] Instance fusion succeeded")
            return True
        else:
            print("[FAIL] Unexpected fusion result")
            return False

    except Exception as e:
        print(f"[FAIL] Instance fusion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conflict_scenarios():
    """Test 5: Verify conflict handling"""
    print("\n" + "=" * 60)
    print("Test 5: Handling conflicting predictions")
    print("=" * 60)

    try:
        fusion_engine = DempsterShaferFusion({
            'model1': 0.9,
            'model2': 0.9
        })

        # Scenario 1: High agreement (low conflict)
        print("\nScenario 1: High agreement")
        group1 = [('model1', 0.9), ('model2', 0.85)]
        result1 = fusion_engine.fuse_instances(group1)
        print(f"  Conflict: {result1.conflict:.3f} (expected: <0.3)")

        # Scenario 2: Moderate conflict
        print("\nScenario 2: Moderate conflict")
        group2 = [('model1', 0.8), ('model2', 0.3)]
        result2 = fusion_engine.fuse_instances(group2)
        print(f"  Conflict: {result2.conflict:.3f} (expected: 0.3-0.6)")

        # Scenario 3: High conflict
        print("\nScenario 3: High conflict")
        group3 = [('model1', 0.9), ('model2', 0.1)]
        result3 = fusion_engine.fuse_instances(group3)
        print(f"  Conflict: {result3.conflict:.3f} (expected: >0.6)")

        if result1.conflict < result2.conflict < result3.conflict:
            print("\n[PASS] Conflicting predictions handled correctly")
            return True
        else:
            print("\n[FAIL] Unexpected ordering of conflict values")
            return False

    except Exception as e:
        print(f"[FAIL] Conflict scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Dempster-Shafer evidence fusion tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("DST module imports", test_dst_import()))
    results.append(("Mass function calculation", test_mass_function()))
    results.append(("Dempster's combination rule", test_dempster_combination()))
    results.append(("Instance fusion", test_instance_fusion()))
    results.append(("Handling conflicting predictions", test_conflict_scenarios()))

    # Summarize the results
    print("\n" + "=" * 60)
    print("Test results")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS] passed" if result else "[FAIL] failed"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All DST fusion tests passed.")
        return True
    else:
        print(f"\n[WARN] {total - passed} tests failed; review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
