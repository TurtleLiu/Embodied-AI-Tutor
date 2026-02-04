import sys
import os
import numpy as np

# Add parent directory to path to import from app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import draw_cube_3d, Cube


def test_draw_cube_3d():
    """Test draw_cube_3d function"""
    print("Testing draw_cube_3d function...")
    
    # Create a cube and get its state
    cube = Cube()
    cube.reset()
    cube_state = cube.get_current_state()
    
    try:
        # Test that the function returns a valid figure
        fig = draw_cube_3d(cube_state)
        assert fig is not None, "draw_cube_3d should return a figure"
        print(f"✓ draw_cube_3d returns a valid figure")
        
        # Test that the figure has the expected number of traces
        # For a 3x3 cube, each face has 9 stickers, 6 faces total = 54 traces
        assert len(fig.data) == 54, f"Expected 54 traces, got {len(fig.data)}"
        print(f"✓ draw_cube_3d creates correct number of traces (54)")
        
        # Test with a scrambled cube state
        cube.apply_sequence("R U R' U'")
        scrambled_state = cube.get_current_state()
        fig_scrambled = draw_cube_3d(scrambled_state)
        assert fig_scrambled is not None, "draw_cube_3d should work with scrambled state"
        print(f"✓ draw_cube_3d works with scrambled cube state")
        
    except Exception as e:
        print(f"✗ Error in draw_cube_3d: {e}")
        raise
    
    print("All draw_cube_3d tests passed!\n")


def test_color_mapping():
    """Test color mapping consistency"""
    print("Testing color mapping...")
    
    # Import COLORS from app.py
    from app import COLORS
    
    expected_colors = {
        'W': '#ffffff',  # White - Up
        'Y': '#ffff00',  # Yellow - Down
        'R': '#ff0000',  # Red - Right
        'O': '#ff8c00',  # Orange - Left
        'B': '#0000ff',  # Blue - Front
        'G': '#00ff00'   # Green - Back
    }
    
    assert COLORS == expected_colors, "COLORS dictionary should match expected values"
    print(f"✓ COLORS dictionary is correct")
    
    print("All color mapping tests passed!\n")


if __name__ == "__main__":
    print("Running UI functionality tests for app.py\n")
    
    test_draw_cube_3d()
    test_color_mapping()
    
    print("All UI tests passed successfully!")
