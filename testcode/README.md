# Test Suite for Rubik's Cube Teaching Robot

This directory contains a comprehensive test suite for validating the correctness of `app.py`, which implements the core functionality of the Rubik's Cube Teaching Robot.

## Test Design

The test suite is structured to verify different aspects of the application:

### 1. Core Functionality Tests (`test_core_functions.py`)
- **Quaternion Class Tests**:
  - `from_v_theta`: Create quaternions from vectors and angles
  - `__mul__`: Quaternion multiplication
  - `as_v_theta`: Convert quaternions back to vectors and angles
  - `as_rotation_matrix`: Convert quaternions to rotation matrices
  - `rotate`: Rotate 3D points

- **CubeSimulator Class Tests**:
  - Initialization with different cube sizes
  - Basic move execution
  - Sequence move execution

- **Cube Class Tests**:
  - Initialization and reset functionality
  - Single face rotation
  - Undo operations
  - Move sequence application
  - Current state retrieval

### 2. UI Functionality Tests (`test_ui_functions.py`)
- **3D Cube Visualization**:
  - `draw_cube_3d` function output validation
  - Correct number of visual elements
  - Support for both solved and scrambled states

- **Color Mapping**:
  - Consistency of color definitions
  - Proper mapping to cube faces

### 3. Activity Case Tests (`test_activity_cases.py`)
- **Predefined Learning Activities**:
  - Proving Non-Commutativity
  - The Commutator
  - Order of an Element
  - Conjugation

- **Sequence Parsing**:
  - Various sequence formats
  - Empty sequences
  - Complex move combinations

## Evaluation Goals

The test suite aims to evaluate the following aspects of the application:

1. **Mathematical Correctness**:
   - Accurate quaternion-based 3D rotations
   - Correct cube state transitions
   - Proper implementation of group theory concepts

2. **Functional Completeness**:
   - All core operations work as expected
   - UI components render correctly
   - Learning activities execute properly

3. **Reliability**:
   - Consistent behavior across different inputs
   - Graceful handling of edge cases
   - Stable state management

4. **Educational Value**:
   - Proper demonstration of group theory concepts
   - Accurate visualization of cube states
   - Clear representation of move sequences

## How to Run Tests

### Prerequisites
- Python 3.7+
- Required dependencies (install via `pip install -r requirements.txt`):
  - streamlit>=1.28.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - Pillow>=9.5.0
  - plotly>=5.17.0

### Running All Tests

```bash
python run_all_tests.py
```

This will execute the complete test suite and provide detailed output for each test case.

### Running Individual Test Files

```bash
# Test core functionality
python test_core_functions.py

# Test UI functionality
python test_ui_functions.py

# Test activity cases
python test_activity_cases.py
```

## Test Results Interpretation

- **✓** - Test passed successfully
- **✗** - Test failed with error

The test output will include detailed information about any failures, including error messages and stack traces to help diagnose issues.

## Notes

- The test suite uses mock implementations for UI dependencies to allow testing without a full Streamlit environment.
- Tests are designed to be independent and can be run in any order.
- For the "Order of an Element" test case, it's expected that the cube returns to its initial state after executing the sequence, demonstrating the mathematical property that the commutator has order 6.

## Contribution Guidelines

When adding new functionality to the application, please:
1. Add corresponding tests to the appropriate test file
2. Ensure all existing tests pass
3. Follow the existing test naming and structure conventions

This will help maintain the reliability and correctness of the Rubik's Cube Teaching Robot application.
