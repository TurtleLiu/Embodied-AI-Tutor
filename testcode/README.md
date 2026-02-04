# 🧪 Test Suite for Rubik's Cube Teaching Robot

This directory contains a comprehensive test suite for validating the correctness of `app.py`, which implements the core functionality of the Rubik's Cube Teaching Robot.

## 📋 Test Design

The test suite is structured to verify different aspects of the application:

### 1. 🧠 Core Functionality Tests (`test_core_functions.py`)
- **Quaternion Class Tests**:
  - `from_v_theta`:
    - **Functionality**: Create quaternions from vectors and angles
    - **Example Input**: `v = [1, 0, 0]`, `theta = π/2`
    - **Expected Output**: Quaternion representing 90° rotation around x-axis
    - **Test**: Verify the quaternion is correctly created and can be used for rotations

  - `__mul__`:
    - **Functionality**: Multiply two quaternions
    - **Example Input**: Two quaternions representing different rotations
    - **Expected Output**: New quaternion representing combined rotation
    - **Test**: Verify multiplication order and result correctness

  - `as_v_theta`:
    - **Functionality**: Convert quaternions back to vectors and angles
    - **Example Input**: Quaternion created from `from_v_theta`
    - **Expected Output**: Original vector and angle (up to numerical precision)
    - **Test**: Verify round-trip conversion works correctly

  - `as_rotation_matrix`:
    - **Functionality**: Convert quaternions to rotation matrices
    - **Example Input**: Quaternion representing a rotation
    - **Expected Output**: 3x3 rotation matrix
    - **Test**: Verify matrix dimensions and correctness

  - `rotate`:
    - **Functionality**: Rotate 3D points
    - **Example Input**: Point `[1, 0, 0]` and quaternion for 90° rotation
    - **Expected Output**: Rotated point `[1, 0, 0]` (unchanged for x-axis rotation)
    - **Test**: Verify points are correctly rotated

- **CubeSimulator Class Tests**:
  - **Initialization**:
    - **Functionality**: Initialize cube simulator with specified size
    - **Example Input**: `N=3` (3x3 cube)
    - **Expected Output**: Initialized simulator with correct move mappings
    - **Test**: Verify simulator is properly set up

  - **Basic Move Execution**:
    - **Functionality**: Execute a single move on the cube
    - **Example Input**: Solved cube state, move `['R', 1]`
    - **Expected Output**: Cube state after right face clockwise rotation
    - **Test**: Verify move changes cube state correctly

  - **Sequence Move Execution**:
    - **Functionality**: Execute a sequence of moves
    - **Example Input**: Solved cube state, sequence `[['R', 1], ['U', 1]]`
    - **Expected Output**: Cube state after both moves
    - **Test**: Verify sequence execution order and result

- **Cube Class Tests**:
  - **Initialization and Reset**:
    - **Functionality**: Initialize and reset cube to solved state
    - **Example Input**: `Cube()` or `cube.reset()`
    - **Expected Output**: Solved cube state
    - **Test**: Verify cube starts in solved state

  - **Single Face Rotation**:
    - **Functionality**: Rotate a single face of the cube
    - **Example Input**: `cube.turn_face('R', 1)`
    - **Expected Output**: Cube state after right face rotation
    - **Test**: Verify face rotation changes state

  - **Undo Operations**:
    - **Functionality**: Undo the last move
    - **Example Input**: After rotating a face, call `cube.undo()`
    - **Expected Output**: Cube state before the last move
    - **Test**: Verify undo returns to previous state

  - **Move Sequence Application**:
    - **Functionality**: Apply a sequence of moves from a string
    - **Example Input**: `cube.apply_sequence("R U R' U'")`
    - **Expected Output**: Cube state after executing all moves
    - **Test**: Verify sequence parsing and execution

  - **Current State Retrieval**:
    - **Functionality**: Get the current cube state
    - **Example Input**: `cube.get_current_state()`
    - **Expected Output**: Array representing current cube state
    - **Test**: Verify state retrieval returns correct data

### 2. 🎨 UI Functionality Tests (`test_ui_functions.py`)
- **3D Cube Visualization**:
  - **Functionality**: Draw 3D representation of the cube
  - **Example Input**: Cube state array
  - **Expected Output**: Plotly figure with 3D cube visualization
  - **Test**: Verify figure creation and correct number of elements (54 stickers)

- **Color Mapping**:
  - **Functionality**: Map cube state to colors
  - **Example Input**: Cube state with color indices
  - **Expected Output**: Correctly colored cube faces
  - **Test**: Verify color mapping consistency

### 3. 📚 Activity Case Tests (`test_activity_cases.py`)
- **Predefined Learning Activities**:
  - **Proving Non-Commutativity**:
    - **Functionality**: Demonstrate that cube moves don't commute
    - **Example Input**: Sequence `"R F' F' R"`
    - **Expected Output**: Cube state showing non-commutativity
    - **Test**: Verify sequence execution changes cube state

  - **The Commutator**:
    - **Functionality**: Demonstrate commutator operation
    - **Example Input**: Sequence `"R F' R' F"`
    - **Expected Output**: Cube state with specific corner and edge swaps
    - **Test**: Verify sequence execution produces expected pattern

  - **Order of an Element**:
    - **Functionality**: Demonstrate the order of a commutator (6)
    - **Example Input**: Sequence `"R F' R' F"` repeated 6 times
    - **Expected Output**: Return to solved state
    - **Test**: Verify cube returns to initial state after 6 repetitions

  - **Conjugation**:
    - **Functionality**: Demonstrate conjugation operation
    - **Example Input**: Sequence `"U R F' R' F U'"`
    - **Expected Output**: Cube state with commutator effect transported to upper face
    - **Test**: Verify sequence execution produces expected pattern

- **Sequence Parsing**:
  - **Functionality**: Parse various move sequence formats
  - **Example Inputs**:
    - `"R U R' U'"` (basic sequence)
    - `"F2 B2 L2 R2 U2 D2"` (double moves)
    - `""` (empty sequence)
  - **Expected Output**: Correctly parsed moves
  - **Test**: Verify different sequence formats are handled correctly

## 🎯 Evaluation Goals

The test suite aims to evaluate the following aspects of the application:

1. **🔢 Mathematical Correctness**:
   - ✅ Accurate quaternion-based 3D rotations
   - ✅ Correct cube state transitions
   - ✅ Proper implementation of group theory concepts

2. **✅ Functional Completeness**:
   - ✅ All core operations work as expected
   - ✅ UI components render correctly
   - ✅ Learning activities execute properly

3. **🛡️ Reliability**:
   - ✅ Consistent behavior across different inputs
   - ✅ Graceful handling of edge cases
   - ✅ Stable state management

4. **📖 Educational Value**:
   - ✅ Proper demonstration of group theory concepts
   - ✅ Accurate visualization of cube states
   - ✅ Clear representation of move sequences

## 🚀 How to Run Tests

### 📋 Prerequisites
- Python 3.7+
- Required dependencies (install via `pip install -r requirements.txt`):
  - streamlit>=1.28.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - Pillow>=9.5.0
  - plotly>=5.17.0

### 🏃‍♂️ Running All Tests

```bash
python run_all_tests.py
```

This will execute the complete test suite and provide detailed output for each test case.

### 🧪 Running Individual Test Files

```bash
# Test core functionality
python test_core_functions.py

# Test UI functionality
python test_ui_functions.py

# Test activity cases
python test_activity_cases.py
```

## 📊 Test Results Interpretation

- **✅** - Test passed successfully
- **❌** - Test failed with error

The test output will include detailed information about any failures, including error messages and stack traces to help diagnose issues.

## 💡 Notes

- 🤖 The test suite uses mock implementations for UI dependencies to allow testing without a full Streamlit environment.
- 🧩 Tests are designed to be independent and can be run in any order.
- 🔄 For the "Order of an Element" test case, it's expected that the cube returns to its initial state after executing the sequence, demonstrating the mathematical property that the commutator has order 6.

## 🤝 Contribution Guidelines

When adding new functionality to the application, please:
1. 📝 Add corresponding tests to the appropriate test file
2. ✅ Ensure all existing tests pass
3. 🎨 Follow the existing test naming and structure conventions

This will help maintain the reliability and correctness of the Rubik's Cube Teaching Robot application. 🚀
