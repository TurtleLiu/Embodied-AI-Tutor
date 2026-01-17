import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time
import random
import plotly.graph_objects as go


class Quaternion:
    """Quaternion Rotation:

    Class to aid in representing 3D rotations via quaternions.
    """
    @classmethod
    def from_v_theta(cls, v, theta):
        """
        Construct quaternions from unit vectors v and rotation angles theta

        Parameters
        ----------
        v : array_like
            array of vectors, last dimension 3. Vectors will be normalized.
        theta : array_like
            array of rotation angles in radians, shape = v.shape[:-1].

        Returns
        -------
        q : quaternion object
            quaternion representing the rotations
        """
        theta = np.asarray(theta)
        v = np.asarray(v)
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)

        v = v * s / np.sqrt(np.sum(v * v, -1))
        x_shape = v.shape[:-1] + (4,)

        x = np.ones(x_shape).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other):
        # multiplication of two quaternions.
        # we don't implement multiplication by a scalar
        sxr = self.x.reshape(self.x.shape[:-1] + (4, 1))
        oxr = other.x.reshape(other.x.shape[:-1] + (1, 4))

        prod = sxr * oxr
        return_shape = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret = np.array([(prod[0, 0] - prod[1, 1]
                         - prod[2, 2] - prod[3, 3]),
                        (prod[0, 1] + prod[1, 0]
                         + prod[2, 3] - prod[3, 2]),
                        (prod[0, 2] - prod[1, 3]
                         + prod[2, 0] + prod[3, 1]),
                        (prod[0, 3] + prod[1, 2]
                         - prod[2, 1] + prod[3, 0])],
                       dtype=np.float,
                       order='F').T
        return self.__class__(ret.reshape(return_shape))

    def as_v_theta(self):
        """Return the v, theta equivalent of the (normalized) quaternion"""
        x = self.x.reshape((-1, 4)).T

        # compute theta
        norm = np.sqrt((x ** 2).sum(0))
        theta = 2 * np.arccos(x[0] / norm)

        # compute the unit vector
        v = np.array(x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        # reshape the results
        v = v.T.reshape(self.x.shape[:-1] + (3,))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    def as_rotation_matrix(self):
        """Return the rotation matrix of the (normalized) quaternion"""
        v, theta = self.as_v_theta()

        shape = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c = np.cos(theta)
        s = np.sin(theta)

        mat = np.array([[v[0] * v[0] * (1. - c) + c,
                         v[0] * v[1] * (1. - c) - v[2] * s,
                         v[0] * v[2] * (1. - c) + v[1] * s],
                        [v[1] * v[0] * (1. - c) + v[2] * s,
                         v[1] * v[1] * (1. - c) + c,
                         v[1] * v[2] * (1. - c) - v[0] * s],
                        [v[2] * v[0] * (1. - c) - v[1] * s,
                         v[2] * v[1] * (1. - c) + v[0] * s,
                         v[2] * v[2] * (1. - c) + c]],
                       order='F').T
        return mat.reshape(shape + (3, 3))

    def rotate(self, points):
        M = self.as_rotation_matrix()
        return np.dot(points, M.T)


# Rubik's cube color definition
COLORS = {
    'W': '#ffffff',  # White - Up
    'Y': '#ffff00',  # Yellow - Down
    'R': '#ff0000',  # Red - Right
    'O': '#ff8c00',  # Orange - Left
    'B': '#0000ff',  # Blue - Front
    'G': '#00ff00'   # Green - Back
}


class CubeSimulator:
    """
    Rubik's Cube Simulator Class
    Used to implement realistic cube rotation logic
    """
    
    def __init__(self, N=3):
        self.N = N
        
        # Legal moves (qtm format)
        self.legalPlays_qtm = [[f, n] for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]
        
        # Solved state
        self.solvedState = np.array([], dtype=int)
        for i in range(6):
            self.solvedState = np.concatenate((self.solvedState, np.arange(i * (N ** 2), (i + 1) * (N ** 2))))
        
        # Pre-compute rotation indices
        self.rotateIdxs_old = dict()
        self.rotateIdxs_new = dict()
        for f, n in self.legalPlays_qtm:
            move = "_".join([f, str(n)])
            
            self.rotateIdxs_new[move] = np.array([], dtype=int)
            self.rotateIdxs_old[move] = np.array([], dtype=int)
            
            colors = np.zeros((6, N, N), dtype=np.int64)
            colors_new = np.copy(colors)
            
            # Color mapping: WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
            adjFaces = {0: np.array([2, 5, 3, 4]),
                        1: np.array([2, 4, 3, 5]),
                        2: np.array([0, 4, 1, 5]),
                        3: np.array([0, 5, 1, 4]),
                        4: np.array([0, 3, 1, 2]),
                        5: np.array([0, 2, 1, 3])}
            
            adjIdxs = {0: {2: [range(0, N), N-1], 3: [range(0, N), N-1], 4: [range(0, N), N-1], 5: [range(0, N), N-1]},
                       1: {2: [range(0, N), 0], 3: [range(0, N), 0], 4: [range(0, N), 0], 5: [range(0, N), 0]},
                       2: {0: [0, range(0, N)], 1: [0, range(0, N)], 4: [N-1, range(N-1, -1, -1)], 5: [0, range(0, N)]},
                       3: {0: [N-1, range(0, N)], 1: [N-1, range(0, N)], 4: [0, range(N-1, -1, -1)], 5: [N-1, range(0, N)]},
                       4: {0: [range(0, N), N-1], 1: [range(N-1, -1, -1), 0], 2: [0, range(0, N)], 3: [N-1, range(N-1, -1, -1)]},
                       5: {0: [range(0, N), 0], 1: [range(N-1, -1, -1), N-1], 2: [N-1, range(0, N)], 3: [0, range(N-1, -1, -1)]}}
            
            faceDict = {'U': 0, 'D': 1, 'L': 2, 'R': 3, 'B': 4, 'F': 5}
            face = faceDict[f]
            
            sign = 1
            if n < 0:
                sign = -1
            
            facesTo = adjFaces[face]
            if sign == 1:
                facesFrom = facesTo[(np.arange(0, len(facesTo)) + 1) % len(facesTo)]
            elif sign == -1:
                facesFrom = facesTo[(np.arange(len(facesTo)-1, len(facesTo)-1 + len(facesTo))) % len(facesTo)]
            
            # Rotate face
            cubesIdxs = [[0, range(0, N)], [range(0, N), N-1], [N-1, range(N-1, -1, -1)], [range(N-1, -1, -1), 0]]
            cubesTo = np.array([0, 1, 2, 3])
            if sign == 1:
                cubesFrom = cubesTo[(np.arange(len(cubesTo)-1, len(cubesTo)-1 + len(cubesTo))) % len(cubesTo)]
            elif sign == -1:
                cubesFrom = cubesTo[(np.arange(0, len(cubesTo)) + 1) % len(cubesTo)]
            
            for i in range(4):
                idxsNew = [[idx1, idx2] for idx1 in np.array([cubesIdxs[cubesTo[i]][0]]).flatten() for idx2 in np.array([cubesIdxs[cubesTo[i]][1]]).flatten()]
                idxsOld = [[idx1, idx2] for idx1 in np.array([cubesIdxs[cubesFrom[i]][0]]).flatten() for idx2 in np.array([cubesIdxs[cubesFrom[i]][1]]).flatten()]
                for idxNew, idxOld in zip(idxsNew, idxsOld):
                    flatIdx_new = np.ravel_multi_index((face, idxNew[0], idxNew[1]), colors_new.shape)
                    flatIdx_old = np.ravel_multi_index((face, idxOld[0], idxOld[1]), colors.shape)
                    self.rotateIdxs_new[move] = np.concatenate((self.rotateIdxs_new[move], [flatIdx_new]))
                    self.rotateIdxs_old[move] = np.concatenate((self.rotateIdxs_old[move], [flatIdx_old]))
            
            # Rotate adjacent faces
            faceIdxs = adjIdxs[face]
            for i in range(0, len(facesTo)):
                faceTo = facesTo[i]
                faceFrom = facesFrom[i]
                idxsNew = [[idx1, idx2] for idx1 in np.array([faceIdxs[faceTo][0]]).flatten() for idx2 in np.array([faceIdxs[faceTo][1]]).flatten()]
                idxsOld = [[idx1, idx2] for idx1 in np.array([faceIdxs[faceFrom][0]]).flatten() for idx2 in np.array([faceIdxs[faceFrom][1]]).flatten()]
                for idxNew, idxOld in zip(idxsNew, idxsOld):
                    flatIdx_new = np.ravel_multi_index((faceTo, idxNew[0], idxNew[1]), colors_new.shape)
                    flatIdx_old = np.ravel_multi_index((faceFrom, idxOld[0], idxOld[1]), colors.shape)
                    self.rotateIdxs_new[move] = np.concatenate((self.rotateIdxs_new[move], [flatIdx_new]))
                    self.rotateIdxs_old[move] = np.concatenate((self.rotateIdxs_old[move], [flatIdx_old]))

    def next_state(self, colors, move, layer=0):
        """Rotate Face"""
        colorsNew = colors.copy()
        
        if type(move[0]) == type(list()):
            for move_sub in move:
                colorsNew = self.next_state(colorsNew, move_sub)
        else:
            moveStr = "_".join([move[0], str(move[1])])
            
            if len(colors.shape) == 1:
                colorsNew[self.rotateIdxs_new[moveStr]] = colors[self.rotateIdxs_old[moveStr]].copy()
            else:
                colorsNew[:, self.rotateIdxs_new[moveStr]] = colors[:, self.rotateIdxs_old[moveStr]].copy()
        
        return colorsNew


class Cube:
    def __init__(self):
        self.Environment = CubeSimulator()
        self.state = np.copy(self.Environment.solvedState)
        self.history = []
        self.tempo = 1.0
    
    def reset(self):
        self.state = np.copy(self.Environment.solvedState)
        self.history = []
    
    def turn_face(self, face, direction=1):
        """
        Rotate a face of the cube
        face: 'U', 'D', 'R', 'L', 'F', 'B'
        direction: 1 = clockwise, -1 = counterclockwise
        """
        self.history.append(np.copy(self.state))
        self.state = self.Environment.next_state(self.state, [face, direction])
    
    def undo(self):
        if len(self.history) > 0:
            self.state = self.history.pop()
    
    def set_tempo(self, tempo):
        self.tempo = tempo
    
    def apply_sequence(self, sequence):
        """
        Apply a move sequence
        sequence: string of moves, e.g., "R U R' U'"
        """
        moves = sequence.split()
        for move in moves:
            face = move[0]
            direction = -1 if "'" in move else 1
            self.turn_face(face, direction)
    
    def get_current_state(self):
        return np.copy(self.state)


# Main UI component function definitions
def draw_cube_3d(cube_state):
    """Draw 3D visualization of the cube - optimized version"""
    fig = go.Figure()
    
    # Color mapping: from cube_interactive_simple.py to our COLORS dictionary
    # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
    color_map = ['W', 'Y', 'B', 'G', 'O', 'R']
    
    # Cube dimensions and parameters - optimized for performance
    N = 3
    cubie_width = 2.0 / N
    sticker_width = 0.92  # Reduced sticker size for natural black gaps
    sticker_margin = 0.5 * (1.0 - sticker_width)
    sticker_thickness = 0.001  # Slightly thicker stickers for visual effect but reduced rendering complexity
    # Removed black border faces, using sticker gaps for black separation
    
    # Basic face and sticker shape
    base_face = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=float)
    d1, d2, d3 = (1 - sticker_margin, 1 - 2 * sticker_margin, 1 + sticker_thickness)
    base_sticker = np.array([[d2, d2, d3], [d2, -d2, d3], [-d2, -d2, d3], [-d2, d2, d3], [d2, d2, d3]], dtype=float)
    
    # Six face rotations
    x, y, z = np.eye(3)
    rots = [Quaternion.from_v_theta(x, theta) for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(y, theta) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]
    
    # Translation for each cubie on each face
    translations = np.array([[[-1 + (i + 0.5) * cubie_width,
                               -1 + (j + 0.5) * cubie_width, 0]]
                             for i in range(N)
                             for j in range(N)])
    
    factor = np.array([1.0 / N, 1.0 / N, 1])
    
    # Iterate through six faces
    for face_idx in range(6):
        M = rots[face_idx].as_rotation_matrix()
        
        # Calculate sticker positions
        stickers_t = np.dot(factor * base_sticker + translations, M.T)
        
        # Iterate through all cubies on this face
        for cubie_idx in range(N * N):
            # Calculate sticker color
            state_idx = face_idx * N * N + cubie_idx
            color_code = color_map[cube_state[state_idx] // (N * N)]
            color = COLORS.get(color_code, '#cccccc')
            
            # Use Mesh3d to draw 3D shapes, more efficient with fill support
            sticker = stickers_t[cubie_idx]
            # Convert quadrilateral to two triangles for Mesh3d
            x_coords = sticker[:, 0]
            y_coords = sticker[:, 1]
            z_coords = sticker[:, 2]
            
            fig.add_trace(
                go.Mesh3d(
                    x=[x_coords[0], x_coords[1], x_coords[2], x_coords[3]],
                    y=[y_coords[0], y_coords[1], y_coords[2], y_coords[3]],
                    z=[z_coords[0], z_coords[1], z_coords[2], z_coords[3]],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color=color,
                    opacity=1.0,
                    showlegend=False
                )
            )
    
    # Set 3D plot properties - optimized performance
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, visible=False, showticklabels=False),
            yaxis=dict(showgrid=False, visible=False, showticklabels=False),
            zaxis=dict(showgrid=False, visible=False, showticklabels=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Moderate camera distance
            aspectmode='cube'
        ),
        width=500,  # Appropriately reduced size for faster rendering
        height=500,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode='orbit',
        showlegend=False,
        autosize=False,  # Disable auto-resize
        uirevision='constant'  # Maintain UI state, reduce redraws
    )
    
    # Optimize rendering settings
    fig.update_traces(
        hoverinfo='none',  # Disable hover info
        selector=dict(type='mesh3d')
    )
    
    return fig


def generate_scramble_sequence(length=20, create_special_sequence=False):
    """Generate random scramble sequence
    create_special_sequence: If True, generate a special sequence where subsequent moves undo the first move
    """
    faces = ['U', 'D', 'L', 'R', 'F', 'B']
    sequence = []
    last_face = None
    
    if create_special_sequence and length >= 2:
        # Generate special sequence ensuring subsequent moves undo the first move
        # Two modes:
        # 1. First move + inverse move (length=2)
        # 2. First move + self-inverse move sequence + inverse move (length>=3)
        
        if length == 2:
            # Mode 1: Direct inverse sequence
            face1 = random.choice(faces)
            direction1 = random.choice([1, -1])
            move1 = face1 + ("'" if direction1 == -1 else "")
            sequence.append(move1)
            
            # Second move: Inverse of first move
            reverse_direction1 = -direction1
            reverse_move1 = face1 + ("'" if reverse_direction1 == -1 else "")
            sequence.append(reverse_move1)
        else:
            # Mode 2: First move + self-inverse moves + inverse move
            # First move
            face1 = random.choice(faces)
            direction1 = random.choice([1, -1])
            move1 = face1 + ("'" if direction1 == -1 else "")
            sequence.append(move1)
            last_face = face1
            
            # Middle steps: Generate self-inverse move sequence (moves that cancel each other)
            # e.g., U R R' U' such sequence
            middle_length = length - 2
            for i in range(middle_length // 2):
                # Choose a face different from the last one
                possible_faces = [f for f in faces if f != last_face]
                face = random.choice(possible_faces)
                
                # Randomly choose rotation direction
                direction = random.choice([1, -1])
                move = face + ("'" if direction == -1 else "")
                
                # Add move and its inverse
                sequence.append(move)
                sequence.append(face + ("" if direction == -1 else "'"))  # Inverse move
                
                last_face = face
            
            # If middle length is odd, add an extra self-inverse move (same move twice)
            if middle_length % 2 != 0:
                possible_faces = [f for f in faces if f != last_face]
                face = random.choice(possible_faces)
                move = face + ("" if random.choice([0, 1]) else "'")  # Random direction
                sequence.append(move)
                sequence.append(move)  # Execute same move again, equivalent to inverse
                last_face = face
            
            # Last move: Inverse of first move
            reverse_direction1 = -direction1
            reverse_move1 = face1 + ("'" if reverse_direction1 == -1 else "")
            sequence.append(reverse_move1)
        
    else:
        # Normal random sequence generation
        for _ in range(length):
            # Randomly select a face different from the last one
            possible_faces = [f for f in faces if f != last_face]
            face = random.choice(possible_faces)
            
            # Randomly choose rotation direction
            direction = random.choice([1, -1])
            move = face + ("'" if direction == -1 else "")
            
            sequence.append(move)
            last_face = face
    
    return " ".join(sequence)


def solve_cube(cube_state):
    """Calculate solution path (currently returns simplified version)"""
    return "R U R' U' R' F R2 U' R' U' R U R' F'"


def generate_prediction_question(operation_type=""):
    """Generate prediction question
    operation_type: Operation type, optional values: "forward" (step forward), "fast_forward" (fast forward)
    """
    # Get current step sequence and cube state
    sequence = st.session_state.rotation_sequence
    moves = sequence.split()
    current_step = st.session_state.current_step
    
    if operation_type == "forward":
        # Single-step prediction question, focusing only on the next move
        if current_step < len(moves) - 1:
            next_move = moves[current_step + 1]
            
            # Generate related question based on next move
            face = next_move[0]
            is_clockwise = "'" not in next_move
            
            # Define possible single-step question types
            question_types = [
                "face_change",  # Ask if specific face will change
                "direction",    # Ask about rotation direction
                "center_change" # Ask if center piece will change
            ]
            
            selected_type = random.choice(question_types)
            
            if selected_type == "face_change":
                # Ensure balanced probability for "yes" and "no" answers
                if random.random() < 0.5:
                    # Generate question with "yes" answer: ask about the face to be operated
                    question_face = face
                    correct_answer = "yes"
                    question = f"After executing {next_move}, will the color distribution on the {face} face change?"
                else:
                    # Generate question with "no" answer: ask about another face
                    other_faces = [f for f in ['U', 'D', 'L', 'R', 'F', 'B'] if f != face]
                    question_face = random.choice(other_faces)
                    correct_answer = "no"
                    question = f"After executing {next_move}, will the color distribution on the {question_face} face change?"
                
            elif selected_type == "direction":
                # Ask if rotation direction is clockwise
                correct_answer = "yes" if is_clockwise else "no"
                question = f"Is {next_move} a clockwise rotation?"
                
            else:  # center_change
                # Ask if center piece color will change (center piece never changes)
                correct_answer = "no"
                question = f"After executing {next_move}, will the center piece color on the {face} face change?"
                
        else:
            # Already at last step, generate generic question
            question = "Is this the last move in the sequence?"
            correct_answer = "yes"
            
    elif operation_type == "fast_forward":
        # Multi-step prediction question, focusing on the effect of the entire sequence
        if len(moves) >= 2:
            # Get first and last moves
            first_move = moves[0]
            last_move = moves[-1]
            
            # Define possible multi-step question types
            question_types = [
                "first_restore",      # Will subsequent moves undo the first move
                "state_change",       # Will overall state change
                "center_change_multi" # Will center pieces change
            ]
            
            # For specific case types, prioritize their special logic
            if st.session_state.selected_case == "Identity and inverse":
                # Check if sequence actually undoes the first move effect
                # Simple mode: Check if it's move + inverse move (like F F')
                # Medium and hard modes: Check if it's forward move + inverse move sequence (like R U U' R')
                sequence_reduces_to_identity = False
                
                if len(moves) >= 2:
                    # For simple mode (move + inverse move)
                    if len(moves) == 2:
                        first_face = moves[0][0]
                        second_face = moves[1][0]
                        # Check if same face
                        if first_face == second_face:
                            # Check if inverse move
                            first_is_clockwise = "'" not in moves[0]
                            second_is_clockwise = "'" not in moves[1]
                            if first_is_clockwise != second_is_clockwise:
                                sequence_reduces_to_identity = True
                    else:
                        # For medium and hard modes (forward move + inverse move sequence)
                        # Check if second half is inverse sequence of first half
                        half_length = len(moves) // 2
                        if len(moves) % 2 == 0:
                            sequence_reduces_to_identity = True
                            for i in range(half_length):
                                forward_move = moves[i]
                                reverse_move = moves[-1 - i]
                                
                                # Check if same face
                                if forward_move[0] != reverse_move[0]:
                                    sequence_reduces_to_identity = False
                                    break
                                
                                # Check if inverse move
                                forward_is_clockwise = "'" not in forward_move
                                reverse_is_clockwise = "'" not in reverse_move
                                if forward_is_clockwise == reverse_is_clockwise:
                                    sequence_reduces_to_identity = False
                                    break
                
                # Set answer based on actual check result
                correct_answer = "yes" if sequence_reduces_to_identity else "no"
                question = f"Will the subsequent moves in the sequence undo the effect of the first move {first_move}?"
            elif st.session_state.selected_case == "Composition and non-commutativity":
                # Handle questions for Composition and non-commutativity case
                if len(moves) >= 2:
                    # Question types for non-commutativity
                    question_types = [
                        "order_effect",        # Does operation order affect result
                        "specific_face_change", # Changes on specific face under different orders
                        "invariant_property"   # Which properties remain unchanged under different orders
                    ]
                    
                    selected_question_type = random.choice(question_types)
                    
                    if selected_question_type == "order_effect":
                        # Generate question about operation order effect
                        # Take first two or three moves
                        if len(moves) >= 3:
                            # Take first three moves
                            sub_moves = moves[:3]
                            sequence_part = " ".join(sub_moves)
                            # Generate reversed order sequence
                            reversed_sequence_part = " ".join(reversed(sub_moves))
                            question = f"Are the results of executing '{sequence_part}' and '{reversed_sequence_part}' the same?"
                        else:
                            # Take first two moves
                            move1, move2 = moves[:2]
                            sequence_part = f"{move1} {move2}"
                            reversed_sequence_part = f"{move2} {move1}"
                            question = f"Are the results of executing '{sequence_part}' and '{reversed_sequence_part}' the same?"
                        
                        # Correct answer is almost always "no" because cube operations are generally non-commutative
                        correct_answer = "no"
                        
                        # Save two sequences for comparison
                        st.session_state.sequence_comparison = {
                            "sequence1": sequence_part,
                            "sequence2": reversed_sequence_part,
                            "comparison_type": "order_effect"
                        }
                    
                    elif selected_question_type == "specific_face_change":
                        # Generate question about specific face changes under different orders
                        # Choose a face likely affected by operations
                        affected_faces = list(set([move[0] for move in moves[:2]]))
                        if affected_faces:
                            target_face = random.choice(affected_faces)
                            move1, move2 = moves[:2]
                            sequence1 = f"{move1} {move2}"
                            sequence2 = f"{move2} {move1}"
                            
                            # Determine question type
                            question_subtypes = [
                                "both_change",     # Both orders will change target face
                                "one_changes",     # Only one order changes target face
                                "different_effect"  # Different effects on target face under different orders
                            ]
                            subtype = random.choice(question_subtypes)
                            
                            if subtype == "both_change":
                                question = f"After executing '{sequence1}' and '{sequence2}', will the color distribution on the {target_face} face change in both cases?"
                                correct_answer = "yes"
                            elif subtype == "one_changes":
                                question = f"Will the {target_face} face change after executing '{sequence1}' but not after '{sequence2}'?"
                                # In reality, this might always be "no", but for teaching purpose set to "yes"
                                correct_answer = "yes"
                            else:
                                question = f"After executing '{sequence1}' and '{sequence2}', will the color distribution changes on the {target_face} face be the same?"
                                correct_answer = "no"
                            
                            # Save two sequences for comparison
                            st.session_state.sequence_comparison = {
                                "sequence1": sequence1,
                                "sequence2": sequence2,
                                "comparison_type": "specific_face_change",
                                "target_face": target_face
                            }
                    
                    else:  # invariant_property
                        # Generate question about invariant properties
                        question = "After executing the same operations in different orders, which of the following properties always remain unchanged?"
                        # Convert to simple yes/no question (since we're using radio)
                        invariant_options = [
                            "Center piece positions",
                            "Parity of total rotation count",
                            "Positions of all corner pieces"
                        ]
                        selected_invariant = random.choice(invariant_options)
                        question = f"After executing the same operations in different orders, does {selected_invariant} always remain unchanged?"
                        
                        # Correct answer
                        if selected_invariant == "Center piece positions":
                            correct_answer = "yes"  # Center positions never change
                        else:
                            correct_answer = "no"  # Other properties might change
                else:
                    # Only one move, generate basic order question
                    question = "If we reverse the order of operations in the sequence, will the result be the same?"
                    correct_answer = "no"
            elif st.session_state.selected_case == "Reusable composites (macro-operators)":
                # Handle questions for Reusable composites (macro-operators) case
                if len(moves) >= 4:
                    # Question types for macro-operators
                    question_types = [
                        "macro_identification",    # Identify macro-operator type
                        "macro_effect",           # Predict macro-operator effect
                        "macro_reuse",            # Macro-operator reuse
                        "commutator_understanding", # Understanding of commutators
                    ]
                    
                    selected_question_type = random.choice(question_types)
                    
                    if selected_question_type == "macro_identification":
                        # Generate question about identifying macro-operator type
                        # Check if sequence contains commutator structure [X, Y] = X Y X' Y'
                        has_commutator = False
                        has_trigger = False
                        
                        # Check for commutator
                        for i in range(len(moves) - 3):
                            move1 = moves[i]
                            move2 = moves[i+1]
                            move3 = moves[i+2]
                            move4 = moves[i+3]
                            
                            # Check if commutator structure
                            if ((move3 == move1 + "'" and move1 != move3) or (move3 == move1[:-1] and move1 == move3 + "'")) and \
                               ((move4 == move2 + "'" and move2 != move4) or (move4 == move2[:-1] and move2 == move4 + "'")):
                                has_commutator = True
                                break
                        
                        # Check for triggers
                        trigger_patterns = [
                            ["R", "U", "R'", "U'"],
                            ["F", "U", "F'", "U'"],
                            ["L", "U", "L'", "U'"],
                            ["B", "U", "B'", "U'"],
                            ["R", "F", "R'", "F'"],
                        ]
                        
                        for pattern in trigger_patterns:
                            if any(moves[i:i+len(pattern)] == pattern for i in range(len(moves) - len(pattern) + 1)):
                                has_trigger = True
                                break
                        
                        # Generate question and options
                        if has_commutator:
                            question = f"What type of macro-operator does the sequence '{sequence}' contain?"
                            options = ["Commutator", "Trigger", "Simple sequence", "Complex algorithm"]
                            correct_answer = "Commutator"
                        elif has_trigger:
                            question = f"What type of macro-operator does the sequence '{sequence}' contain?"
                            options = ["Commutator", "Trigger", "Simple sequence", "Complex algorithm"]
                            correct_answer = "Trigger"
                        else:
                            question = f"Does the sequence '{sequence}' contain a reusable macro-operator?"
                            options = ["yes", "no"]
                            correct_answer = "yes"
                        
                        # Save question type
                        st.session_state.prediction_question_type = "multiple_choice"
                        st.session_state.prediction_options = options
                        
                    elif selected_question_type == "macro_effect":
                        # Generate question about macro-operator effect
                        # Check if contains commutator
                        has_commutator = False
                        for i in range(len(moves) - 3):
                            move1 = moves[i]
                            move2 = moves[i+1]
                            move3 = moves[i+2]
                            move4 = moves[i+3]
                            
                            if ((move3 == move1 + "'" and move1 != move3) or (move3 == move1[:-1] and move1 == move3 + "'")) and \
                               ((move4 == move2 + "'" and move2 != move4) or (move4 == move2[:-1] and move2 == move4 + "'")):
                                has_commutator = True
                                break
                        
                        if has_commutator:
                            question = "What is the main function of the commutator macro-operator [X, Y] = X Y X' Y'?"
                            options = [
                                "Swap specific pieces on the cube without affecting others",
                                "Rotate the entire cube",
                                "Change color distribution on all faces",
                                "Restore the cube to initial state"
                            ]
                            correct_answer = "Swap specific pieces on the cube without affecting others"
                        else:
                            question = "What is the main effect of this macro-operator?"
                            options = [
                                "Change color distribution on specific faces",
                                "Swap or rotate specific pieces",
                                "Completely scramble the cube",
                                "Restore the cube to initial state"
                            ]
                            correct_answer = "Swap or rotate specific pieces"
                        
                        # Save question type
                        st.session_state.prediction_question_type = "multiple_choice"
                        st.session_state.prediction_options = options
                        
                    elif selected_question_type == "macro_reuse":
                        # Generate question about macro-operator reuse
                        question = "If we execute this macro-operator sequence twice, will the result be the same as executing it once?"
                        # Actual answer depends on specific macro, but for teaching purpose set to "no"
                        correct_answer = "no"
                        
                    else:  # commutator_understanding
                        # Generate question about commutator understanding
                        question = "In the commutator [X, Y] = X Y X' Y', what is the purpose of the X' Y' part?"
                        options = [
                            "Cancel the effect of X Y on most pieces, leaving only specific piece changes",
                            "Enhance the effect of X Y",
                            "Change the overall structure of the cube",
                            "No specific purpose, just random operations"
                        ]
                        correct_answer = "Cancel the effect of X Y on most pieces, leaving only specific piece changes"
                        
                        # Save question type
                        st.session_state.prediction_question_type = "multiple_choice"
                        st.session_state.prediction_options = options
                        
                else:
                    # Sequence too short, generate basic macro-operator question
                    question = "Can this sequence be used as a reusable macro-operator?"
                    correct_answer = "yes"
            else:
                # Other case types use original answer balance logic
                # Ensure balanced probability for "yes" and "no" answers
                answer_balance = random.random() < 0.5
                
                if answer_balance:
                    # Generate question with "yes" answer
                    correct_answer = "yes"
                    
                    # Randomly select question type
                    will_question_types = [
                        "state_change",  # Will overall state change
                        "first_restore"  # Will subsequent moves undo first move
                    ]
                    will_selected_type = random.choice(will_question_types)
                    
                    if will_selected_type == "first_restore":
                        # Generate special sequence where subsequent moves undo first move
                        sequence_length = len(moves)
                        special_sequence = generate_scramble_sequence(sequence_length, create_special_sequence=True)
                        
                        # Update session state sequence and state
                        st.session_state.rotation_sequence = special_sequence
                        st.session_state.current_step = -1
                        st.session_state.is_playing = False
                        
                        # Pre-compute all states for new sequence
                        temp_cube = Cube()
                        temp_cube.state = np.copy(st.session_state.cube.state)
                        new_states = [np.copy(temp_cube.state)]
                        new_moves = special_sequence.split()
                        for move in new_moves:
                            face = move[0]
                            direction = -1 if "'" in move else 1
                            temp_cube.turn_face(face, direction)
                            new_states.append(np.copy(temp_cube.state))
                        st.session_state.sequence_states = new_states
                        
                        # Update current cube state
                        st.session_state.cube.state = np.copy(new_states[0])
                        
                        # Get first move of new sequence
                        new_first_move = new_moves[0]
                        
                        # Generate question
                        question = f"Will the subsequent moves in the sequence undo the effect of the first move {new_first_move}?"
                        
                    else:  # state_change
                        question = "After executing all moves, will the cube state change?"
                        
                else:
                    # Generate question with "no" answer
                    question_types = [
                        "first_restore",      # Will subsequent moves undo first move
                        "center_change_multi" # Will center pieces change
                    ]
                    selected_type = random.choice(question_types)
                    
                    if selected_type == "first_restore":
                        # Default answer for other case types is "no"
                        correct_answer = "no"
                        question = f"Will the subsequent moves in the sequence undo the effect of the first move {first_move}?"
                    else:  # center_change_multi
                        # Ask if any center piece will change
                        correct_answer = "no"
                        question = "After executing all moves, will any face's center piece color change?"
                
        else:
            # Only one move, generate single-step question
            move = moves[0]
            question = f"After executing {move}, will the cube state change?"
            correct_answer = "yes"
            
    else:
        # Default simple question
        question = "After executing these moves, will the cube state change?"
        correct_answer = "yes"
    
    # Define options
    # Check if options already set for specific question type
    if hasattr(st.session_state, 'prediction_question_type') and st.session_state.prediction_question_type == "multiple_choice":
        # For multiple choice type, use already set options
        options = st.session_state.prediction_options
    else:
        # For regular questions, use default options
        options = ["yes", "no"] if correct_answer in ["yes", "no"] else ["yes", "no"]
    
    # Save to session state
    st.session_state.prediction_question = question
    st.session_state.prediction_options = options
    st.session_state.correct_answer = correct_answer
    
    # Clear temporary question type marker
    if hasattr(st.session_state, 'prediction_question_type'):
        del st.session_state.prediction_question_type


# Initialize app
if 'cube' not in st.session_state:
    st.session_state.cube = Cube()
if 'rotation_sequence' not in st.session_state:
    st.session_state.rotation_sequence = ""
if 'current_step' not in st.session_state:
    st.session_state.current_step = -1
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'sequence_states' not in st.session_state:
    st.session_state.sequence_states = []
if 'play_timer' not in st.session_state:
    st.session_state.play_timer = time.time()
if 'play_start_time' not in st.session_state:
    st.session_state.play_start_time = time.time()
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

# Auto-play logic - simplified implementation as suggested by user (moved before UI rendering)
if st.session_state.is_playing:
    # Ensure there's a rotation sequence and pre-computed states
    if st.session_state.sequence_states and st.session_state.rotation_sequence:
        moves = st.session_state.rotation_sequence.split()
        
        # Check if there are more steps to execute
        if st.session_state.current_step < len(moves) - 1:
            # Execute next step
            st.session_state.current_step += 1
            
            # Ensure index is within valid range
            if st.session_state.current_step + 1 < len(st.session_state.sequence_states):
                # Update cube state
                st.session_state.cube.state = st.session_state.sequence_states[st.session_state.current_step + 1]
                
                # Control animation speed
                time.sleep(0.6)
                
                # Force re-run script to show new state
                st.rerun()
        else:
            # Playback finished
            st.session_state.is_playing = False
            st.session_state.current_step = len(moves) - 1


# Set page title
st.set_page_config(page_title="Rubik's Cube Teaching Bot", layout="wide")

# App title
st.title("Rubik's Cube Teaching Bot")

# Sidebar - Micro-experiment Case Generation Settings
st.sidebar.title("Micro-experiment Case Generation")

# Case selection
st.sidebar.subheader("Case Selection")
# Define case types
case_types = [
    "Action-to-operator mapping", 
    "Identity and inverse",
    "Composition and non-commutativity",
    "Reusable composites (macro-operators)"
]
# Select case type
selected_case = st.sidebar.selectbox(
    "Select Case Type",
    case_types,
    index=0  # Default to first case
)

# Case difficulty settings
st.sidebar.subheader("Difficulty Settings")
# Set different difficulty options based on selected case type
if selected_case == "Action-to-operator mapping":
    difficulty_options = [
        "Easy (2-3 move sequence, basic operation combinations)", 
        "Medium (4-5 move sequence, includes inverse moves)", 
        "Hard (6-8 move sequence, complex operation combinations)"
    ]
elif selected_case == "Identity and inverse":
    difficulty_options = [
        "Easy (single move and its inverse, e.g., R R')", 
        "Medium (two moves and their inverses, e.g., R U U' R')", 
        "Hard (three or more moves and their inverses, e.g., R U F F' U' R')"
    ]
elif selected_case == "Composition and non-commutativity":
    difficulty_options = [
        "Easy (two basic operations, e.g., R U vs U R, directly showing ab≠ba)", 
        "Medium (three operation combinations, e.g., R U F vs F U R, showing more complex non-commutativity)", 
        "Hard (four or more operation combinations, may include inverse moves, e.g., R U' F R vs R F U' R, more challenging non-commutativity demonstration)"
    ]
elif selected_case == "Reusable composites (macro-operators)":
    difficulty_options = [
        "Easy (basic 2-3 move macro-operators, like commutators or triggers, e.g., [R, U] = R U R' U')", 
        "Medium (4-5 move macro-operator combinations, like two commutators chained, or simple cube algorithms)", 
        "Hard (6+ move complex macro-operators, may include inverse moves and nested macro combinations, like classic OLL or PLL algorithm fragments)"
    ]
else:
    # Default difficulty options for other case types
    difficulty_options = [
        "Easy (basic operations)", 
        "Medium (complex operations)", 
        "Hard (advanced operations)"
    ]
# Select difficulty
selected_difficulty = st.sidebar.selectbox(
    "Select Difficulty",
    difficulty_options,
    index=0  # Default to easy difficulty
)

# Case generation button
st.sidebar.subheader("Case Generation")
if st.sidebar.button("Generate Case"):
    # Initialize case generation related session states
    st.session_state.selected_case = selected_case
    st.session_state.selected_difficulty = selected_difficulty
    
    # Determine sequence length based on difficulty
    if "Easy" in selected_difficulty:
        sequence_length = 2  # Easy mode fixed to 2 moves: operation + inverse
    elif "Medium" in selected_difficulty:
        sequence_length = random.choice([4, 6])  # Medium mode uses even length for reversible sequence structure
    else:
        sequence_length = random.choice([6, 8, 10])  # Hard mode uses even length for reversible sequence structure
    
    # Generate corresponding move sequence based on case type
    if selected_case == "Identity and inverse":
        # Generate special sequence for Identity and inverse case, demonstrating operation reversibility
        faces = ['U', 'D', 'L', 'R', 'F', 'B']
        sequence = []
        last_face = None
        
        # Determine sequence structure based on difficulty
        if "Easy" in selected_difficulty:
            # Easy difficulty: single move and its inverse
            face = random.choice(faces)
            direction = random.choice([1, -1])
            move = face + ("'" if direction == -1 else "")
            sequence.append(move)
            # Add inverse move
            inverse_move = face + ("" if direction == -1 else "'")
            sequence.append(inverse_move)
        
        elif "Medium" in selected_difficulty:
            # Medium difficulty: multiple moves and their inverses
            steps = sequence_length // 2  # Half are forward moves, half are inverse moves
            
            # Generate forward moves
            for i in range(steps):
                # Randomly select a face different from last one
                possible_faces = [f for f in faces if f != last_face]
                face = random.choice(possible_faces)
                
                # Randomly choose rotation direction
                direction = random.choice([1, -1])
                move = face + ("'" if direction == -1 else "")
                
                sequence.append(move)
                last_face = face
            
            # Add inverse move sequence
            for i in range(steps-1, -1, -1):
                move = sequence[i]
                face = move[0]
                is_clockwise = "'" not in move
                inverse_move = face + ("" if is_clockwise else "'")
                sequence.append(inverse_move)
        
        else:  # Hard difficulty
            # Hard difficulty: three or more moves and their inverses
            steps = sequence_length // 2  # Half are forward moves, half are inverse moves
            
            # Generate forward moves
            for i in range(steps):
                # Randomly select a face different from last one
                possible_faces = [f for f in faces if f != last_face]
                face = random.choice(possible_faces)
                
                # Randomly choose rotation direction
                direction = random.choice([1, -1])
                move = face + ("'" if direction == -1 else "")
                
                sequence.append(move)
                last_face = face
            
            # Add inverse move sequence
            for i in range(steps-1, -1, -1):
                move = sequence[i]
                face = move[0]
                is_clockwise = "'" not in move
                inverse_move = face + ("" if is_clockwise else "'")
                sequence.append(inverse_move)
        
        # Check if generated sequence length matches requirement
        # Since we used sequence_length // 2 to generate forward and inverse moves, length should be exactly sequence_length
        # If discrepancy, it might be due to sequence_length being odd
        if len(sequence) != sequence_length:
            # Regenerate sequence ensuring correct length
            sequence = []
            last_face = None
            
            if "Easy" in selected_difficulty:
                # Easy difficulty: single move and its inverse
                face = random.choice(faces)
                direction = random.choice([1, -1])
                move = face + ("'" if direction == -1 else "")
                sequence.append(move)
                # Add inverse move
                inverse_move = face + ("" if direction == -1 else "'")
                sequence.append(inverse_move)
            else:
                # Medium and Hard difficulty: multiple moves and their inverses
                steps = sequence_length // 2
                
                # Generate forward moves
                for i in range(steps):
                    possible_faces = [f for f in faces if f != last_face]
                    face = random.choice(possible_faces)
                    direction = random.choice([1, -1])
                    move = face + ("'" if direction == -1 else "")
                    sequence.append(move)
                    last_face = face
                
                # Add inverse move sequence
                for i in range(steps-1, -1, -1):
                    move = sequence[i]
                    face = move[0]
                    is_clockwise = "'" not in move
                    inverse_move = face + ("" if is_clockwise else "'")
                    sequence.append(inverse_move)
        random_sequence = " ".join(sequence)
    elif selected_case == "Composition and non-commutativity":
        # Generate special sequence for Composition and non-commutativity case, demonstrating non-commutativity
        faces = ['U', 'D', 'L', 'R', 'F', 'B']
        sequence = []
        
        # Determine sequence structure based on difficulty
        if "Easy" in selected_difficulty:
            # Easy difficulty: two basic operations on different faces, e.g., R U
            # Choose two different faces
            face1 = random.choice(faces)
            face2 = random.choice([f for f in faces if f != face1])
            
            # Generate two-move sequence
            move1 = face1 + ("'" if random.choice([0, 1]) else "")
            move2 = face2 + ("'" if random.choice([0, 1]) else "")
            
            sequence.append(move1)
            sequence.append(move2)
        
        elif "Medium" in selected_difficulty:
            # Medium difficulty: three operations on different faces, e.g., R U F
            # Choose three different faces
            face1 = random.choice(faces)
            face2 = random.choice([f for f in faces if f != face1])
            face3 = random.choice([f for f in faces if f not in [face1, face2]])
            
            # Generate three-move sequence
            move1 = face1 + ("'" if random.choice([0, 1]) else "")
            move2 = face2 + ("'" if random.choice([0, 1]) else "")
            move3 = face3 + ("'" if random.choice([0, 1]) else "")
            
            sequence.append(move1)
            sequence.append(move2)
            sequence.append(move3)
        
        else:  # Hard difficulty
            # Hard difficulty: four or more move sequence, may include inverse moves, e.g., R U' F R
            # Choose four or more different faces (can repeat but try to diversify)
            num_moves = random.randint(4, 6)
            last_face = None
            
            for i in range(num_moves):
                # Randomly select a face different from last one
                possible_faces = [f for f in faces if f != last_face]
                face = random.choice(possible_faces)
                
                # Randomly choose rotation direction
                move = face + ("'" if random.choice([0, 1]) else "")
                
                sequence.append(move)
                last_face = face
        
        random_sequence = " ".join(sequence)
    elif selected_case == "Reusable composites (macro-operators)":
        # Generate sequence containing macro-operators for Reusable composites case
        faces = ['U', 'D', 'L', 'R', 'F', 'B']
        sequence = []
        
        # Determine sequence structure based on difficulty
        if "Easy" in selected_difficulty:
            # Easy difficulty: basic 2-3 move macro-operators, like commutators or triggers
            macro_types = [
                "commutator",  # Commutator [X, Y] = X Y X' Y'
                "trigger",     # Trigger, like R U R' U' or F R U R' U' F'
            ]
            
            selected_macro = random.choice(macro_types)
            
            if selected_macro == "commutator":
                # Generate commutator [X, Y] = X Y X' Y'
                # Choose two different faces
                face1 = random.choice(faces)
                face2 = random.choice([f for f in faces if f != face1])
                
                # Generate commutator
                move1 = face1 + ("'" if random.choice([0, 1]) else "")
                move2 = face2 + ("'" if random.choice([0, 1]) else "")
                move3 = face1 + ("'" if "'" not in move1 else "")  # Inverse move
                move4 = face2 + ("'" if "'" not in move2 else "")  # Inverse move
                
                sequence = [move1, move2, move3, move4]
            else:  # trigger
                # Generate trigger, like R U R' U' or F R U R' U' F'
                # Choose a face as starting face
                face = random.choice(faces)
                
                # Generate trigger sequence
                if random.choice([0, 1]):
                    # 4-move trigger
                    move1 = face
                    move2 = 'U' if random.choice([0, 1]) else 'R'
                    move3 = face + "'"
                    move4 = move2 + "'"
                    sequence = [move1, move2, move3, move4]
                else:
                    # 6-move trigger
                    move1 = face
                    move2 = 'R' if random.choice([0, 1]) else 'U'
                    move3 = 'U'
                    move4 = move2 + "'"
                    move5 = 'U' + "'"
                    move6 = face + "'"
                    sequence = [move1, move2, move3, move4, move5, move6]
        
        elif "Medium" in selected_difficulty:
            # Medium difficulty: 4-5 move macro-operator combinations, like two commutators chained, or simple cube algorithms
            macro_types = [
                "double_commutator",  # Two commutators combined
                "simple_algorithm",   # Simple cube algorithms, like Sune or Anti-Sune
            ]
            
            selected_macro = random.choice(macro_types)
            
            if selected_macro == "double_commutator":
                # Generate two commutators combined
                # First commutator
                face1 = random.choice(faces)
                face2 = random.choice([f for f in faces if f != face1])
                
                move1 = face1 + ("'" if random.choice([0, 1]) else "")
                move2 = face2 + ("'" if random.choice([0, 1]) else "")
                move3 = face1 + ("'" if "'" not in move1 else "")
                move4 = face2 + ("'" if "'" not in move2 else "")
                
                # Second commutator
                face3 = random.choice([f for f in faces if f not in [face1, face2]])
                face4 = random.choice([f for f in faces if f not in [face1, face2]])
                
                move5 = face3 + ("'" if random.choice([0, 1]) else "")
                move6 = face4 + ("'" if random.choice([0, 1]) else "")
                move7 = face3 + ("'" if "'" not in move5 else "")
                move8 = face4 + ("'" if "'" not in move6 else "")
                
                sequence = [move1, move2, move3, move4, move5, move6, move7, move8]
            else:  # simple_algorithm
                # Generate simple cube algorithms, like Sune or Anti-Sune
                algorithms = [
                    "R U R' U R U2 R'",    # Sune
                    "R' U' R U' R' U2 R",  # Anti-Sune
                    "F R U R' U' F'",      # Top cross fix
                    "R U R' U R U2 R'",    # Top corner orientation
                ]
                
                sequence = random.choice(algorithms).split()
        
        else:  # Hard difficulty
            # Hard difficulty: 6+ move complex macro-operators, may include inverse moves and nested macro combinations
            complex_algorithms = [
                "R U R' F' R U R' U' R' F R2 U' R' U'",  # Complex top layer algorithm
                "F R U R' U' F' R U R' U' R' F R F'",   # Two triggers combined
                "R2 D R' U2 R D' R' U2 R'",             # Top corner positioning
                "U R U' L' U R' U' L",                  # Swap adjacent corners
                "R U R' U R U2 R' L' U L U2 L'",        # Complex OLL algorithm
            ]
            
            # Randomly choose a complex algorithm, or generate nested macro combinations
            if random.choice([0, 1]):
                # Use predefined complex algorithm
                sequence = random.choice(complex_algorithms).split()
            else:
                # Generate nested macro combination
                # Outer macro
                face1 = random.choice(faces)
                face2 = random.choice([f for f in faces if f != face1])
                
                move1 = face1
                move2 = face2
                
                # Inner macro (as Y part of outer)
                face3 = random.choice([f for f in faces if f not in [face1, face2]])
                face4 = random.choice([f for f in faces if f not in [face1, face2]])
                
                inner_move1 = face3
                inner_move2 = face4
                inner_move3 = face3 + "'"
                inner_move4 = face4 + "'"
                
                # Complete commutator structure
                move3 = face1 + "'"
                move4 = face2 + "'"
                
                sequence = [move1, inner_move1, inner_move2, inner_move3, inner_move4, move3, move4]
        
        random_sequence = " ".join(sequence)
    
    else:
        # Other case types generate normal sequence
        random_sequence = generate_scramble_sequence(sequence_length)
    
    st.session_state.rotation_sequence = random_sequence
    st.session_state.current_step = -1
    st.session_state.is_playing = False
    
    # Randomly decide initial cube state: 50% chance solved cube, 50% scrambled cube
    use_scrambled_initial = random.choice([True, False])
    
    # Pre-compute all states for the sequence
    temp_cube = Cube()
    
    # Set initial state
    if use_scrambled_initial:
        # Generate a scrambled initial state
        initial_scramble_length = random.randint(5, 10)
        initial_scramble = generate_scramble_sequence(initial_scramble_length)
        temp_cube.state = np.copy(st.session_state.cube.state)  # Start with solved state
        
        # Execute initial scramble
        for move in initial_scramble.split():
            face = move[0]
            direction = -1 if "'" in move else 1
            temp_cube.turn_face(face, direction)
    else:
        # Use solved initial state
        temp_cube.reset()
    
    # Save initial state and generate step state sequence
    states = [np.copy(temp_cube.state)]
    moves = random_sequence.split()
    for move in moves:
        face = move[0]
        direction = -1 if "'" in move else 1
        temp_cube.turn_face(face, direction)
        states.append(np.copy(temp_cube.state))
    st.session_state.sequence_states = states
    
    # Update current cube state to new initial state
    st.session_state.cube.state = np.copy(states[0])
    
    # Set case generation status
    st.session_state.case_generated = True
    
    # Generate associated prediction question based on generated move sequence
    # Determine question type: if sequence has only 1 move, generate single-step prediction, else multi-step
    if sequence_length == 1:
        operation_type = "forward"
    else:
        operation_type = "fast_forward"
    
    # Generate prediction question
    generate_prediction_question(operation_type=operation_type)
    
    # Reset case learning related states
    st.session_state.student_answer = None
    st.session_state.prediction_correct = None
    st.session_state.execution_started = False
    st.session_state.answer_checked = False
    
    # Force re-run script to show new case
    st.rerun()

# Case reset button
if 'case_generated' in st.session_state and st.session_state.case_generated:
    if st.sidebar.button("Reset Case"):
        # Clear case generation related session states
        if 'selected_case' in st.session_state:
            del st.session_state.selected_case
        if 'selected_difficulty' in st.session_state:
            del st.session_state.selected_difficulty
        if 'case_generated' in st.session_state:
            del st.session_state.case_generated
        if 'prediction_question' in st.session_state:
            del st.session_state.prediction_question
        if 'prediction_options' in st.session_state:
            del st.session_state.prediction_options
        if 'student_answer' in st.session_state:
            del st.session_state.student_answer
        if 'prediction_correct' in st.session_state:
            del st.session_state.prediction_correct
        if 'execution_started' in st.session_state:
            del st.session_state.execution_started
        
        # Reset cube state
        st.session_state.rotation_sequence = ""
        st.session_state.current_step = -1
        st.session_state.is_playing = False
        st.session_state.sequence_states = []
        st.session_state.cube.reset()
        
        # Force re-run script to update UI
        st.rerun()


# Main content area
# If case generated, display case content
if 'case_generated' in st.session_state and st.session_state.case_generated:
    # Display different concept introduction based on case type
    if st.session_state.selected_case == "Action-to-operator mapping":
        # Original Operator concept introduction
        st.markdown("## Rubik's Cube Operator Concept Introduction")
        st.write("In Rubik's Cube, an Operator refers to a basic operation on the cube. Each operator represents a specific face rotation:")
        st.write("- **R**: Right face clockwise rotation")
        st.write("- **L**: Left face clockwise rotation")
        st.write("- **U**: Up face clockwise rotation")
        st.write("- **D**: Down face clockwise rotation")
        st.write("- **F**: Front face clockwise rotation")
        st.write("- **B**: Back face clockwise rotation")
        st.write("- Operators with apostrophe (like R') represent counterclockwise rotation")
    elif st.session_state.selected_case == "Identity and inverse":
        # Identity and inverse concept introduction
        st.markdown("## Identity and Inverse Concept Introduction")
        st.write("In Rubik's Cube operations, each operation has its inverse operation, which can cancel the effect of the original operation.")
        st.write("- **Identity**: Executing a series of operations that returns the cube to initial state")
        st.write("- **Inverse**: An operation that cancels the effect of another operation")
        st.write("- **Example 1 (Easy)**: R R' → Execute right face clockwise rotation then counterclockwise rotation, cube returns to initial state")
        st.write("- **Example 2 (Medium)**: R U U' R' → Execute right face clockwise rotation, then up face clockwise rotation, then up face counterclockwise rotation, finally right face counterclockwise rotation, cube returns to initial state")
        st.write("- **Inverse notation**: Original operation with apostrophe (like R') means counterclockwise rotation, the inverse of clockwise rotation R")
    elif st.session_state.selected_case == "Composition and non-commutativity":
        # Composition and non-commutativity concept introduction
        st.markdown("## Composition and Non-Commutativity Concept Introduction")
        st.write("In Rubik's Cube operations, the order of operation composition is very important because cube operations are generally non-commutative.")
        st.write("- **Composition**: Executing multiple cube operations consecutively")
        st.write("- **Non-Commutativity**: Operation order affects final result, i.e., ab≠ba")
        st.write("- **Invariants**: Certain properties of the cube remain unchanged regardless of operation order (like center piece positions, parity of total rotation count)")
        st.write("- **Example 1 (Easy)**: R U vs U R → The result of right face clockwise then up face clockwise is different from up face clockwise then right face clockwise")
        st.write("- **Example 2 (Medium)**: R U F vs F U R → Different composition orders of three operations produce different results")
    elif st.session_state.selected_case == "Reusable composites (macro-operators)":
        # Reusable composites (macro-operators) concept introduction
        st.markdown("## Reusable Composites (Macro-Operators) Concept Introduction")
        st.write("In Rubik's Cube operations, reusable composite operations (macro-operators) are sets of operations with specific functions that can be used as a whole.")
        st.write("- **Macro-Operator**: A sequence of operations with specific function that can be reused as a unit")
        st.write("- **Commutator**: Operation sequence of form [X, Y] = X Y X' Y', typically used to swap pieces on the cube")
        st.write("- **Trigger**: Short operation sequences that frequently appear in cube algorithms")
        st.write("- **Example 1 (Easy)**: [R, U] = R U R' U' → Commutator, used to swap specific pieces")
        st.write("- **Example 2 (Medium)**: R U R' U' R' F R F' → Sune algorithm, used to solve top corner orientation")
        st.write("- **Advantage**: Macro-operators can break down complex problems into smaller, manageable units, improving cube solving efficiency")
    else:
        # Default introduction for other case types
        st.markdown("## Case Concept Introduction")
        st.write("This is a Rubik's Cube teaching case demonstrating basic concepts and effects of cube operations.")
        st.write("Use the control buttons on the right to execute and observe cube operations.")
    
    # Middle area divided into left and right columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Rubik's Cube 3D Display")
        fig = draw_cube_3d(st.session_state.cube.state)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display randomly generated move sequence
        st.subheader("Move Sequence")
        if st.session_state.rotation_sequence:
            
            # Check if sequence comparison exists
            if st.session_state.selected_case == "Composition and non-commutativity" and 'sequence_comparison' in st.session_state:
                comparison_data = st.session_state.sequence_comparison
                
                # Create dropdown for user to select which sequence to view
                sequence_options = [
                    ("sequence1", f"Sequence 1: {comparison_data['sequence1']}"),
                    ("sequence2", f"Sequence 2: {comparison_data['sequence2']}")
                ]
                
                # Default to sequence 1
                selected_sequence = st.selectbox(
                    "Select sequence to view",
                    options=[opt[1] for opt in sequence_options],
                    index=0,
                    key="sequence_comparison_select"
                )
                
                # Get currently selected sequence
                selected_sequence_key = sequence_options[[opt[1] for opt in sequence_options].index(selected_sequence)][0]
                current_sequence = comparison_data[selected_sequence_key]
                
                # Check if sequence switched, if yes, reset cube to initial state
                if 'previous_selected_sequence' not in st.session_state or st.session_state.previous_selected_sequence != selected_sequence_key:
                    # Reset cube to initial state
                    st.session_state.current_step = -1
                    st.session_state.is_playing = False
                    if st.session_state.sequence_states:
                        st.session_state.cube.state = st.session_state.sequence_states[0]
                    # Save currently selected sequence
                    st.session_state.previous_selected_sequence = selected_sequence_key
                    st.session_state.current_comparison_sequence = current_sequence
                    st.rerun()
                
                # If first load or no current comparison sequence, set default
                if 'current_comparison_sequence' not in st.session_state:
                    st.session_state.current_comparison_sequence = comparison_data['sequence1']
            else:
                # Normal case, only display one sequence
                current_sequence = st.session_state.rotation_sequence
            
            # Display currently selected sequence
            moves = current_sequence.split()
            # Display rotation track, executed steps and current step colored
            display_sequence = []
            for i, move in enumerate(moves):
                if i < st.session_state.current_step:
                    display_sequence.append(f"<span style='color: green; text-decoration: line-through;'>{move}</span>")
                elif i == st.session_state.current_step:
                    display_sequence.append(f"<span style='color: red; font-weight: bold;'>{move}</span>")
                else:
                    display_sequence.append(f"<span style='color: black;'>{move}</span>")
            
            st.markdown(" ".join(display_sequence), unsafe_allow_html=True)
        
        # Operation buttons - Fast Backward x steps, Step Backward, Step Forward, Fast Forward x steps
        st.subheader("Operation Controls")
        col_controls1, col_controls2, col_controls3, col_controls4 = st.columns(4)
        
        with col_controls1:
            if st.button("Fast Backward", help="Return to initial state"):
                # Execute fast backward, directly return to initial state
                st.session_state.is_playing = False
                st.session_state.current_step = -1
                if st.session_state.sequence_states:
                    st.session_state.cube.state = st.session_state.sequence_states[0]
                # Force re-run script to immediately update UI
                st.rerun()
                
        with col_controls2:
            if st.button("Step Backward", help="Step backward"):
                st.session_state.is_playing = False
                
                if st.session_state.current_step > -1:
                    st.session_state.current_step -= 1
                    
                    # Check current sequence
                    if st.session_state.selected_case == "Composition and non-commutativity" and 'current_comparison_sequence' in st.session_state:
                        # Use current comparison sequence
                        current_sequence = st.session_state.current_comparison_sequence
                        moves = current_sequence.split()
                        
                        # Create temporary cube and calculate current step state
                        temp_cube = Cube()
                        temp_cube.state = st.session_state.sequence_states[0].copy()
                        
                        # Execute up to current step
                        for i in range(st.session_state.current_step + 1):
                            move = moves[i]
                            face = move[0]
                            direction = -1 if "'" in move else 1
                            temp_cube.turn_face(face, direction)
                        
                        # Update cube state
                        st.session_state.cube.state = temp_cube.state.copy()
                    elif st.session_state.sequence_states:
                        st.session_state.cube.state = st.session_state.sequence_states[st.session_state.current_step + 1]
                
                # Force re-run script to immediately update UI
                st.rerun()
                
        with col_controls3:
            if st.button("Step Forward", help="Step forward"):
                st.session_state.is_playing = False
                
                # Check current sequence
                if st.session_state.selected_case == "Composition and non-commutativity" and 'current_comparison_sequence' in st.session_state:
                    # Use current comparison sequence
                    current_sequence = st.session_state.current_comparison_sequence
                else:
                    # Use default sequence
                    current_sequence = st.session_state.rotation_sequence
                
                moves = current_sequence.split()
                
                # Check if can step forward (constrained by move sequence)
                if st.session_state.current_step < len(moves) - 1:
                    st.session_state.current_step += 1
                    
                    # If comparison sequence, need to recalculate state
                    if st.session_state.selected_case == "Composition and non-commutativity" and 'current_comparison_sequence' in st.session_state:
                        # Create temporary cube and calculate current step state
                        temp_cube = Cube()
                        temp_cube.state = st.session_state.sequence_states[0].copy()
                        
                        # Execute up to current step
                        for i in range(st.session_state.current_step + 1):
                            move = moves[i]
                            face = move[0]
                            direction = -1 if "'" in move else 1
                            temp_cube.turn_face(face, direction)
                        
                        # Update cube state
                        st.session_state.cube.state = temp_cube.state.copy()
                    elif st.session_state.sequence_states:
                        st.session_state.cube.state = st.session_state.sequence_states[st.session_state.current_step + 1]
                else:
                    st.warning("Already at the last step!")
                
                # Force re-run script to immediately update UI
                st.rerun()
                
        with col_controls4:
            if st.button("Fast Forward", help="Fast forward to last step"):
                # Execute fast forward, directly advance to last step
                st.session_state.is_playing = False
                
                # Check current sequence
                if st.session_state.selected_case == "Composition and non-commutativity" and 'current_comparison_sequence' in st.session_state:
                    # Use current comparison sequence
                    current_sequence = st.session_state.current_comparison_sequence
                    moves = current_sequence.split()
                    new_step = len(moves) - 1
                    
                    if new_step != st.session_state.current_step:
                        st.session_state.current_step = new_step
                        
                        # Create temporary cube and calculate last step state
                        temp_cube = Cube()
                        temp_cube.state = st.session_state.sequence_states[0].copy()
                        
                        # Execute all steps
                        for move in moves:
                            face = move[0]
                            direction = -1 if "'" in move else 1
                            temp_cube.turn_face(face, direction)
                        
                        # Update cube state
                        st.session_state.cube.state = temp_cube.state.copy()
                else:
                    # Use default sequence
                    moves = st.session_state.rotation_sequence.split()
                    new_step = len(moves) - 1
                    if new_step != st.session_state.current_step:
                        st.session_state.current_step = new_step
                        if st.session_state.sequence_states:
                            st.session_state.cube.state = st.session_state.sequence_states[st.session_state.current_step + 1]
                
                # Force re-run script to immediately update UI
                st.rerun()
        
        # Question display area
        st.subheader("Prediction Question")
        
        # Display current case's move sequence
        st.write(f"Current question's move sequence:")
        st.code(st.session_state.rotation_sequence, language="text")
        
        # Display question
        if 'prediction_question' in st.session_state and st.session_state.prediction_question:
            st.write(st.session_state.prediction_question)
            
            if 'prediction_options' in st.session_state and st.session_state.prediction_options:
                # Use dynamic key to ensure radio resets when switching questions
                if 'question_counter' not in st.session_state:
                    st.session_state.question_counter = 0
                
                # Use radio component for student to select answer
                student_answer = st.radio(
                    "Please select your answer:",
                    st.session_state.prediction_options,
                    index=None,
                    key=f"student_answer_radio_{st.session_state.question_counter}"
                )
                
                # Save student answer
                if student_answer is not None:
                    st.session_state.student_answer = student_answer
        
        # Check answer and switch next question buttons
        st.write("\n")
        col_answer, col_next = st.columns(2)
        
        with col_answer:
            if st.button("Check Answer"):
                if 'student_answer' in st.session_state and st.session_state.student_answer is not None:
                    st.session_state.answer_checked = True
                    
                    # Determine if prediction is correct
                    if 'correct_answer' in st.session_state:
                        if st.session_state.student_answer == st.session_state.correct_answer:
                            st.success("Prediction correct!")
                            st.session_state.prediction_correct = True
                        else:
                            st.error("Prediction incorrect!")
                            st.session_state.prediction_correct = False
                            st.write(f"Correct answer is: {st.session_state.correct_answer}")
                else:
                    st.warning("Please select an answer first!")
        
        with col_next:
            if st.button("Switch to Next Question"):
                # Randomly decide initial cube state: 50% solved, 50% scrambled
                use_scrambled_initial = random.choice([True, False])
                
                # Generate new move sequence
                if "Easy" in st.session_state.selected_difficulty:
                    new_sequence_length = random.randint(2, 3)
                    # Easy mode fixed to 2 moves for reversible or non-commutative sequences
                    if st.session_state.selected_case in ["Identity and inverse", "Composition and non-commutativity"]:
                        new_sequence_length = 2
                elif "Medium" in st.session_state.selected_difficulty:
                    new_sequence_length = random.randint(4, 5)
                    # Medium mode fixed to 4 moves for reversible sequences, 3 for non-commutative
                    if st.session_state.selected_case == "Identity and inverse":
                        new_sequence_length = 4
                    elif st.session_state.selected_case == "Composition and non-commutativity":
                        new_sequence_length = 3
                else:
                    new_sequence_length = random.randint(6, 8)
                    # Hard mode for non-commutative sequences may need 4-6 moves
                    if st.session_state.selected_case == "Composition and non-commutativity":
                        new_sequence_length = random.randint(4, 6)
                
                # Generate new move sequence
                if st.session_state.selected_case == "Identity and inverse":
                    # 50% chance reversible sequence, 50% non-reversible
                    if random.random() < 0.5:
                        # Generate reversible sequence
                        if new_sequence_length == 2:
                            # Easy mode: single move + inverse
                            faces = ['U', 'D', 'L', 'R', 'F', 'B']
                            face = random.choice(faces)
                            direction = random.choice([1, -1])
                            move = face + ("'" if direction == -1 else "")
                            inverse_move = face + ("" if direction == -1 else "'")
                            new_random_sequence = f"{move} {inverse_move}"
                        else:
                            # Medium and hard modes: forward moves + inverse sequence
                            new_random_sequence = generate_scramble_sequence(new_sequence_length, create_special_sequence=True)
                    else:
                        # Generate non-reversible sequence
                        new_random_sequence = generate_scramble_sequence(new_sequence_length)
                elif st.session_state.selected_case == "Composition and non-commutativity":
                    # Generate non-commutative sequence
                    faces = ['U', 'D', 'L', 'R', 'F', 'B']
                    new_sequence = []
                    
                    if "Easy" in st.session_state.selected_difficulty:
                        # Easy difficulty: two basic operations on different faces
                        face1 = random.choice(faces)
                        face2 = random.choice([f for f in faces if f != face1])
                        move1 = face1 + ("'" if random.choice([0, 1]) else "")
                        move2 = face2 + ("'" if random.choice([0, 1]) else "")
                        new_sequence = [move1, move2]
                    elif "Medium" in st.session_state.selected_difficulty:
                        # Medium difficulty: three operations on different faces
                        face1 = random.choice(faces)
                        face2 = random.choice([f for f in faces if f != face1])
                        face3 = random.choice([f for f in faces if f not in [face1, face2]])
                        move1 = face1 + ("'" if random.choice([0, 1]) else "")
                        move2 = face2 + ("'" if random.choice([0, 1]) else "")
                        move3 = face3 + ("'" if random.choice([0, 1]) else "")
                        new_sequence = [move1, move2, move3]
                    else:
                        # Hard difficulty: four or more move sequence
                        num_moves = new_sequence_length
                        last_face = None
                        for i in range(num_moves):
                            possible_faces = [f for f in faces if f != last_face]
                            face = random.choice(possible_faces)
                            move = face + ("'" if random.choice([0, 1]) else "")
                            new_sequence.append(move)
                            last_face = face
                    
                    new_random_sequence = " ".join(new_sequence)
                elif st.session_state.selected_case == "Reusable composites (macro-operators)":
                    # Generate sequence containing macro-operators for Reusable composites case
                    faces = ['U', 'D', 'L', 'R', 'F', 'B']
                    new_sequence = []
                    
                    # Determine sequence structure based on difficulty
                    if "Easy" in st.session_state.selected_difficulty:
                        # Easy difficulty: basic 2-3 move macro-operators, like commutators or triggers
                        macro_types = [
                            "commutator",  # Commutator [X, Y] = X Y X' Y'
                            "trigger",     # Trigger, like R U R' U' or F R U R' U' F'
                        ]
                        
                        selected_macro = random.choice(macro_types)
                        
                        if selected_macro == "commutator":
                            # Generate commutator [X, Y] = X Y X' Y'
                            # Choose two different faces
                            face1 = random.choice(faces)
                            face2 = random.choice([f for f in faces if f != face1])
                            
                            # Generate commutator
                            move1 = face1 + ("'" if random.choice([0, 1]) else "")
                            move2 = face2 + ("'" if random.choice([0, 1]) else "")
                            move3 = face1 + ("'" if "'" not in move1 else "")  # Inverse move
                            move4 = face2 + ("'" if "'" not in move2 else "")  # Inverse move
                            
                            new_sequence = [move1, move2, move3, move4]
                        else:  # trigger
                            # Generate trigger, like R U R' U' or F R U R' U' F'
                            # Choose a face as starting face
                            face = random.choice(faces)
                            
                            # Generate trigger sequence
                            if random.choice([0, 1]):
                                # 4-move trigger
                                move1 = face
                                move2 = 'U' if random.choice([0, 1]) else 'R'
                                move3 = face + "'"
                                move4 = move2 + "'"
                                new_sequence = [move1, move2, move3, move4]
                            else:
                                # 6-move trigger
                                move1 = face
                                move2 = 'R' if random.choice([0, 1]) else 'U'
                                move3 = 'U'
                                move4 = move2 + "'"
                                move5 = 'U' + "'"
                                move6 = face + "'"
                                new_sequence = [move1, move2, move3, move4, move5, move6]
                    
                    elif "Medium" in st.session_state.selected_difficulty:
                        # Medium difficulty: 4-5 move macro-operator combinations, like two commutators chained, or simple cube algorithms
                        macro_types = [
                            "double_commutator",  # Two commutators combined
                            "simple_algorithm",   # Simple cube algorithms, like Sune or Anti-Sune
                        ]
                        
                        selected_macro = random.choice(macro_types)
                        
                        if selected_macro == "double_commutator":
                            # Generate two commutators combined
                            # First commutator
                            face1 = random.choice(faces)
                            face2 = random.choice([f for f in faces if f != face1])
                            
                            move1 = face1 + ("'" if random.choice([0, 1]) else "")
                            move2 = face2 + ("'" if random.choice([0, 1]) else "")
                            move3 = face1 + ("'" if "'" not in move1 else "")
                            move4 = face2 + ("'" if "'" not in move2 else "")
                            
                            # Second commutator
                            face3 = random.choice([f for f in faces if f not in [face1, face2]])
                            face4 = random.choice([f for f in faces if f not in [face1, face2]])
                            
                            move5 = face3 + ("'" if random.choice([0, 1]) else "")
                            move6 = face4 + ("'" if random.choice([0, 1]) else "")
                            move7 = face3 + ("'" if "'" not in move5 else "")
                            move8 = face4 + ("'" if "'" not in move6 else "")
                            
                            new_sequence = [move1, move2, move3, move4, move5, move6, move7, move8]
                        else:  # simple_algorithm
                            # Generate simple cube algorithms, like Sune or Anti-Sune
                            algorithms = [
                                "R U R' U R U2 R'",    # Sune
                                "R' U' R U' R' U2 R",  # Anti-Sune
                                "F R U R' U' F'",      # Top cross fix
                                "R U R' U R U2 R'",    # Top corner orientation
                            ]
                            
                            new_sequence = random.choice(algorithms).split()
                    
                    else:  # Hard difficulty
                        # Hard difficulty: 6+ move complex macro-operators, may include inverse moves and nested macro combinations
                        complex_algorithms = [
                            "R U R' F' R U R' U' R' F R2 U' R' U'",  # Complex top layer algorithm
                            "F R U R' U' F' R U R' U' R' F R F'",   # Two triggers combined
                            "R2 D R' U2 R D' R' U2 R'",             # Top corner positioning
                            "U R U' L' U R' U' L",                  # Swap adjacent corners
                            "R U R' U R U2 R' L' U L U2 L'",        # Complex OLL algorithm
                        ]
                        
                        # Randomly choose a complex algorithm, or generate nested macro combinations
                        if random.choice([0, 1]):
                            # Use predefined complex algorithm
                            new_sequence = random.choice(complex_algorithms).split()
                        else:
                            # Generate nested macro combination
                            # Outer macro
                            face1 = random.choice(faces)
                            face2 = random.choice([f for f in faces if f != face1])
                            
                            move1 = face1
                            move2 = face2
                            
                            # Inner macro (as Y part of outer)
                            face3 = random.choice([f for f in faces if f not in [face1, face2]])
                            face4 = random.choice([f for f in faces if f not in [face1, face2]])
                            
                            inner_move1 = face3
                            inner_move2 = face4
                            inner_move3 = face3 + "'"
                            inner_move4 = face4 + "'"
                            
                            # Complete commutator structure
                            move3 = face1 + "'"
                            move4 = face2 + "'"
                            
                            new_sequence = [move1, inner_move1, inner_move2, inner_move3, inner_move4, move3, move4]
                    
                    new_random_sequence = " ".join(new_sequence)
                else:
                    # Other case types generate normal sequence
                    new_random_sequence = generate_scramble_sequence(new_sequence_length)
                
                st.session_state.rotation_sequence = new_random_sequence
                st.session_state.current_step = -1
                st.session_state.is_playing = False
                
                # Pre-compute all states for new sequence
                temp_cube = Cube()
                
                # Set initial state
                if use_scrambled_initial:
                    # Generate a scrambled initial state
                    initial_scramble_length = random.randint(5, 10)
                    initial_scramble = generate_scramble_sequence(initial_scramble_length)
                    temp_cube.state = np.copy(st.session_state.cube.state)  # Start with solved state
                    
                    # Execute initial scramble
                    for move in initial_scramble.split():
                        face = move[0]
                        direction = -1 if "'" in move else 1
                        temp_cube.turn_face(face, direction)
                else:
                    # Use solved initial state
                    temp_cube.reset()
                
                # Save initial state and generate step state sequence
                new_states = [np.copy(temp_cube.state)]
                new_moves = new_random_sequence.split()
                for move in new_moves:
                    face = move[0]
                    direction = -1 if "'" in move else 1
                    temp_cube.turn_face(face, direction)
                    new_states.append(np.copy(temp_cube.state))
                st.session_state.sequence_states = new_states
                
                # Update current cube state to new initial state
                st.session_state.cube.state = np.copy(new_states[0])
                
                # Generate associated prediction question based on new move sequence
                if new_sequence_length == 1:
                    operation_type = "forward"
                else:
                    operation_type = "fast_forward"
                
                # Generate new prediction question
                generate_prediction_question(operation_type=operation_type)
                
                # Reset answer states
                if 'student_answer' in st.session_state:
                    del st.session_state.student_answer
                if 'answer_checked' in st.session_state:
                    del st.session_state.answer_checked
                if 'prediction_correct' in st.session_state:
                    del st.session_state.prediction_correct
                
                # Clean up sequence comparison related states
                if 'sequence_comparison' in st.session_state:
                    del st.session_state.sequence_comparison
                if 'previous_selected_sequence' in st.session_state:
                    del st.session_state.previous_selected_sequence
                if 'current_comparison_sequence' in st.session_state:
                    del st.session_state.current_comparison_sequence
                
                # Increment question counter for resetting radio component
                if 'question_counter' not in st.session_state:
                    st.session_state.question_counter = 0
                st.session_state.question_counter += 1
                
                # Force re-run to show new question and move sequence
                st.rerun()

else:
    # If case not generated, display default interface
    st.title("🎯 Exploring Algebra Through Rubik's Cube")
    st.write("Welcome to the Rubik's Cube Algebra Learning Platform! Here you'll intuitively understand abstract algebraic concepts through cube operations.")
    
    st.subheader("🌟 Learning Goals")
    
    # Learning Goal LG1
    st.write("### 🎮 LG1: Action-to-operator mapping")
    st.write("Treat cube face rotations as **operators** in algebra, and predict results of combined operations (e.g., execute R then U).")
    
    # Learning Goal LG2
    st.write("### 🔄 LG2: Identity and inverse as testable phenomena")
    st.write("Recognize and explain reversibility through short experiments (e.g., aa⁻¹ returns to initial state), not just memorizing 'prime' notation.")
    
    # Learning Goal LG3
    st.write("### ⚡ LG3: Composition and non-commutativity")
    st.write("Generate and explain counterexamples where ab≠ba in cube operations, clearly articulate changes (and unchanged properties) under different orders.")
    
    # Learning Goal LG4
    st.write("### 🧩 LG4: Reusable composites (macro-operators)")
    st.write("Identify and reuse short move sequences as meaningful units (like 'execute this commutator'), connecting procedural chunks to algebraic composition.")
    
    st.subheader("📋 Usage Logic")
    st.write("1. Select **Case Type** of interest in left sidebar")
    st.write("2. Choose **Difficulty** based on your learning level")
    st.write("3. Click **Generate Case** button to begin learning journey")
    st.write("4. Deeply understand algebraic concepts through observation, prediction, and verification")
    
    st.success("Let's get started! Select a case type to explore the algebraic world behind Rubik's Cube 🔍")