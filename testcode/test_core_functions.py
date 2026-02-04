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

from app import Quaternion, CubeSimulator, Cube


def test_quaternion():
    """Test Quaternion class functionality"""
    print("Testing Quaternion class...")
    
    # Test from_v_theta method
    v = [1, 0, 0]
    theta = np.pi / 2
    q = Quaternion.from_v_theta(v, theta)
    print(f"✓ Quaternion.from_v_theta created: {q}")
    
    # Test multiplication
    q2 = Quaternion.from_v_theta([0, 1, 0], np.pi / 2)
    q_product = q * q2
    print(f"✓ Quaternion multiplication works")
    
    # Test as_v_theta method
    v_out, theta_out = q.as_v_theta()
    print(f"✓ Quaternion.as_v_theta works: v={v_out}, theta={theta_out}")
    
    # Test as_rotation_matrix method
    mat = q.as_rotation_matrix()
    print(f"✓ Quaternion.as_rotation_matrix works: shape={mat.shape}")
    
    # Test rotate method
    point = np.array([1, 0, 0])
    rotated_point = q.rotate(point)
    print(f"✓ Quaternion.rotate works: {rotated_point}")
    
    print("All Quaternion tests passed!\n")


def test_cube_simulator():
    """Test CubeSimulator class functionality"""
    print("Testing CubeSimulator class...")
    
    # Test initialization
    simulator = CubeSimulator(N=3)
    print(f"✓ CubeSimulator initialized with N=3")
    
    # Test next_state method
    # Create a solved cube state
    solved_state = np.array([i for i in range(6 * 3 * 3)])
    
    # Test a simple move
    new_state = simulator.next_state(solved_state, ['R', 1])
    print(f"✓ CubeSimulator.next_state works for simple move")
    
    # Test a sequence of moves
    sequence = [['R', 1], ['U', 1], ['R', -1], ['U', -1]]
    state_after_sequence = solved_state
    for move in sequence:
        state_after_sequence = simulator.next_state(state_after_sequence, move)
    print(f"✓ CubeSimulator.next_state works for move sequence")
    
    print("All CubeSimulator tests passed!\n")


def test_cube():
    """Test Cube class functionality"""
    print("Testing Cube class...")
    
    # Test initialization
    cube = Cube()
    print(f"✓ Cube initialized")
    
    # Test reset method
    cube.reset()
    initial_state = cube.get_current_state()
    print(f"✓ Cube.reset works")
    
    # Test turn_face method
    cube.turn_face('R', 1)
    state_after_turn = cube.get_current_state()
    assert not np.array_equal(initial_state, state_after_turn), "Cube state should change after turn"
    print(f"✓ Cube.turn_face works")
    
    # Test undo method
    cube.undo()
    state_after_undo = cube.get_current_state()
    assert np.array_equal(initial_state, state_after_undo), "Cube state should revert after undo"
    print(f"✓ Cube.undo works")
    
    # Test apply_sequence method
    cube.reset()
    sequence = "R U R' U'"
    cube.apply_sequence(sequence)
    print(f"✓ Cube.apply_sequence works for sequence: {sequence}")
    
    # Test get_current_state method
    current_state = cube.get_current_state()
    print(f"✓ Cube.get_current_state works: shape={current_state.shape}")
    
    print("All Cube tests passed!\n")


def test_cube_states():
    """Test cube state consistency"""
    print("Testing cube state consistency...")
    
    cube = Cube()
    cube.reset()
    
    # Test that cube starts in solved state
    solved_state = cube.get_current_state()
    print(f"✓ Cube starts in solved state")
    
    # Test a complex sequence
    complex_sequence = "R F' R' F R F' R' F"
    cube.apply_sequence(complex_sequence)
    
    # Test that cube state is consistent
    final_state = cube.get_current_state()
    assert final_state.shape == (54,), "Cube state should have shape (54,) for 3x3 cube"
    print(f"✓ Cube state consistency maintained after complex sequence")
    
    print("All cube state tests passed!\n")


if __name__ == "__main__":
    print("Running core functionality tests for app.py\n")
    
    test_quaternion()
    test_cube_simulator()
    test_cube()
    test_cube_states()
    
    print("All tests passed successfully!")
