import sys
import os
import numpy as np

# Add parent directory to path to import from app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only the core classes, avoiding UI dependencies
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __iter__(self):
        return iter([])
    def __len__(self):
        return 0
    def __call__(self, *args, **kwargs):
        # Special handling for st.columns()
        if args and len(args) > 0 and isinstance(args[0], (int, list)):
            # For st.columns(n), return a tuple of n MockModule objects
            if isinstance(args[0], int):
                return tuple([MockModule() for _ in range(args[0])])
            # For st.columns([...]), return a tuple of MockModule objects
            elif isinstance(args[0], list):
                return tuple([MockModule() for _ in range(len(args[0]))])
        return MockModule()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MockPIL:
    class Image:
        pass
    PngImagePlugin = MockModule()

# Create mock modules
sys.modules['streamlit'] = MockModule()
sys.modules['plotly'] = MockModule()
sys.modules['plotly.graph_objects'] = MockModule()
sys.modules['plotly.graph_objects'].Figure = MockModule()
sys.modules['matplotlib'] = MockModule()
sys.modules['matplotlib.pyplot'] = MockModule()
sys.modules['matplotlib.patches'] = MockModule()
sys.modules['PIL'] = MockPIL

from app import Cube


def test_activity_cases():
    """Test all activity cases"""
    print("Testing activity cases...")
    
    # Define activity cases
    activity_cases = [
        {
            "name": "Proving Non-Commutativity",
            "sequence": "R F' F' R",
            "description": "Test non-commutativity of cube moves"
        },
        {
            "name": "The Commutator",
            "sequence": "R F' R' F",
            "description": "Test commutator sequence"
        },
        {
            "name": "Order of an Element",
            "sequence": "R F' R' F R F' R' F R F' R' F R F' R' F R F' R' F R F' R' F",
            "description": "Test order of commutator element"
        },
        {
            "name": "Conjugation",
            "sequence": "U R F' R' F U'",
            "description": "Test conjugation sequence"
        }
    ]
    
    for case in activity_cases:
        print(f"\nTesting case: {case['name']}")
        print(f"Description: {case['description']}")
        print(f"Sequence: {case['sequence']}")
        
        # Test that the sequence can be applied
        cube = Cube()
        cube.reset()
        
        try:
            cube.apply_sequence(case['sequence'])
            print(f"✓ Sequence applied successfully")
            
            # Test that cube state changed (except for Order of an Element case, which should return to initial state)
            final_state = cube.get_current_state()
            cube.reset()
            initial_state = cube.get_current_state()
            
            if case['name'] == "Order of an Element":
                # For Order of an Element case, we expect it to return to initial state
                assert np.array_equal(final_state, initial_state), "Cube state should return to initial state for Order of an Element"
                print(f"✓ Cube state returned to initial state as expected (order of commutator is 6)")
            else:
                # For other cases, we expect state to change
                assert not np.array_equal(final_state, initial_state), "Cube state should change after sequence"
                print(f"✓ Cube state changed as expected")
            
        except Exception as e:
            print(f"✗ Error applying sequence: {e}")
            raise
    
    print("\nAll activity case tests passed!\n")


def test_sequence_parsing():
    """Test sequence parsing functionality"""
    print("Testing sequence parsing...")
    
    cube = Cube()
    cube.reset()
    
    # Test various sequence formats
    test_sequences = [
        "R U R' U'",  # Basic sequence with apostrophes
        "F2 B2 L2 R2 U2 D2",  # Double moves (though our implementation treats F2 as F F)
        "R L F B U D",  # All faces
        "R' L' F' B' U' D'",  # All faces counterclockwise
        "",  # Empty sequence
    ]
    
    for sequence in test_sequences:
        try:
            cube.reset()
            cube.apply_sequence(sequence)
            print(f"✓ Sequence parsed successfully: '{sequence}'")
        except Exception as e:
            print(f"✗ Error parsing sequence '{sequence}': {e}")
            raise
    
    print("All sequence parsing tests passed!\n")


if __name__ == "__main__":
    print("Running activity case tests for app.py\n")
    
    test_activity_cases()
    test_sequence_parsing()
    
    print("All activity case tests passed successfully!")
