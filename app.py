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


# Cube color definitions
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
    Cube Simulator Class
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
        
        # Precompute rotation indices
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
        Apply a sequence of moves
        sequence: move sequence string, e.g., "R U R' U'"
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
    """Draw 3D visual representation of the cube - optimized version"""
    fig = go.Figure()
    
    # Color mapping: from cube_interactive_simple.py to our COLORS dictionary
    # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
    color_map = ['W', 'Y', 'B', 'G', 'O', 'R']
    
    # Cube dimensions and parameters - optimized for performance
    N = 3
    cubie_width = 2.0 / N
    sticker_width = 0.92  # Reduce sticker size to create natural black gaps
    sticker_margin = 0.5 * (1.0 - sticker_width)
    sticker_thickness = 0.001  # Slightly thicker stickers, maintain visual effect but reduce rendering complexity
    # Remove black border faces, achieve black separation through sticker gaps
    
    # Base face and sticker shape
    base_face = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=float)
    d1, d2, d3 = (1 - sticker_margin, 1 - 2 * sticker_margin, 1 + sticker_thickness)
    base_sticker = np.array([[d2, d2, d3], [d2, -d2, d3], [-d2, -d2, d3], [-d2, d2, d3], [d2, d2, d3]], dtype=float)
    
    # Rotations for six faces
    x, y, z = np.eye(3)
    rots = [Quaternion.from_v_theta(x, theta) for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(y, theta) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]
    
    # Translation for N^2 cubies per face
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
        
        # Iterate through all cubies of this face
        for cubie_idx in range(N * N):
            # Calculate sticker color
            state_idx = face_idx * N * N + cubie_idx
            color_code = color_map[cube_state[state_idx] // (N * N)]
            color = COLORS.get(color_code, '#cccccc')
            
            # Use Mesh3d to draw 3D shapes, more efficient and supports filling
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
    
    # Set 3D plot properties - optimized for performance
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, visible=False, showticklabels=False),
            yaxis=dict(showgrid=False, visible=False, showticklabels=False),
            zaxis=dict(showgrid=False, visible=False, showticklabels=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Moderate camera distance
            aspectmode='cube'
        ),
        width=500,  # Appropriately reduce size for faster rendering
        height=500,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode='orbit',
        showlegend=False,
        autosize=False,  # Disable auto resize
        uirevision='constant'  # Maintain UI state, reduce redraws
    )
    
    # Optimize rendering settings
    fig.update_traces(
        hoverinfo='none',  # Disable hover info
        selector=dict(type='mesh3d')
    )
    
    return fig


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

# Auto-play logic - simplified implementation based on user suggestion (moved before UI rendering)
if st.session_state.is_playing:
    # Ensure there is a rotation sequence and precomputed states
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
                
                # Force rerun script to display new state
                st.rerun()
        else:
            # Playback ended
            st.session_state.is_playing = False
            st.session_state.current_step = len(moves) - 1


# Set page title
st.set_page_config(page_title="Rubik's Cube Teaching Robot", layout="wide")

# App title
# st.title("Rubik's Cube Teaching Robot")

# Increase font size for all text elements using custom CSS
st.markdown("""
<style>
    /* Increase font size for main content text */
    .stMarkdown p {
        font-size: 18px;
    }
    
    /* Increase font size for list items */
    .stMarkdown ul li, .stMarkdown ol li {
        font-size: 18px;
    }
    
    /* Increase font size for buttons */
    .stButton > button {
        font-size: 16px;
    }
    
    /* Increase font size for subheaders */
    h3 {
        font-size: 24px;
    }
    
    /* Increase font size for prediction questions */
    .stMarkdown h4 {
        font-size: 20px;
    }
    
    /* Increase font size for radio button options */
    .stRadio > label {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Fixed top content for all interfaces
st.markdown("""
### 🧠 **An Embodied AI Tutor that Turns Rubik’s Cube Interaction into Learnable Group Structure** 🧠

Welcome to this interactive Rubik's Cube learning experience! This program demonstrates an embodied AI tutor that helps you understand group theory concepts through hands-on cube manipulation.

### 🎯 **Activity Case: From Commutators to Conjugates** 🎯

- **Goal**: Verify non-commutativity and explore cycle decomposition through cube sequences.
- **Setup**: Start from a solved cube state ($e$).

### 📋 Learning Activities:

Please select an activity to experience!
""")

# Create buttons for each activity case
### st.subheader("Select an Activity Case")

# Initialize selected_case if not exists
if 'selected_case' not in st.session_state:
    st.session_state.selected_case = "Proving Non-Commutativity"
    # Set up default case
    st.session_state.rotation_sequence = "R F' F' R"
    st.session_state.current_step = -1
    st.session_state.is_playing = False
    
    # Set up initial cube state (solved)
    temp_cube = Cube()
    temp_cube.reset()
    
    # Generate sequence states
    sequence_states = [np.copy(temp_cube.state)]
    moves = st.session_state.rotation_sequence.split()
    for move in moves:
        face = move[0]
        direction = -1 if "'" in move else 1
        temp_cube.turn_face(face, direction)
        sequence_states.append(np.copy(temp_cube.state))
    st.session_state.sequence_states = sequence_states
    
    # Update current cube state
    st.session_state.cube.state = np.copy(sequence_states[0])
    
    # Set up prediction question: "Does R F' have the same effect as F' R?"
    st.session_state.prediction_question = "Does the sequence R F' have the same effect as the sequence F' R?"
    st.session_state.prediction_options = ["Yes", "No"]
    st.session_state.correct_answer = "No"

# Create buttons with conditional styling
col1, col2, col3, col4 = st.columns(4)

# Button 1 - Proving Non-Commutativity
with col1:
    if st.session_state.selected_case == "Proving Non-Commutativity":
        # Red button for selected case
        if st.button("Proving Non-Commutativity", key="case1", type="primary"):
            pass  # Already selected
    else:
        if st.button("Proving Non-Commutativity", key="case1"):
            # Set up Non-Commutativity case
            st.session_state.selected_case = "Proving Non-Commutativity"
            st.session_state.rotation_sequence = "R F' F' R"
            st.session_state.current_step = -1
            st.session_state.is_playing = False
            
            # Set up initial cube state (solved)
            temp_cube = Cube()
            temp_cube.reset()
            
            # Generate sequence states
            sequence_states = [np.copy(temp_cube.state)]
            moves = st.session_state.rotation_sequence.split()
            for move in moves:
                face = move[0]
                direction = -1 if "'" in move else 1
                temp_cube.turn_face(face, direction)
                sequence_states.append(np.copy(temp_cube.state))
            st.session_state.sequence_states = sequence_states
            
            # Update current cube state
            st.session_state.cube.state = np.copy(sequence_states[0])
            
            # Set up prediction question
            st.session_state.prediction_question = "Will the subsequent moves F' R in the sequence undo the effect of the first moves R F'?"
            st.session_state.prediction_options = ["Yes", "No"]
            st.session_state.correct_answer = "No"
            
            # Reset answer state
            if 'student_answer' in st.session_state:
                del st.session_state.student_answer
            if 'answer_checked' in st.session_state:
                del st.session_state.answer_checked
            if 'prediction_correct' in st.session_state:
                del st.session_state.prediction_correct
            
            # Force rerun to update button styling immediately
            st.rerun()

# Button 2 - The Commutator
with col2:
    if st.session_state.selected_case == "The Commutator":
        # Red button for selected case
        if st.button("The Commutator C", key="case2", type="primary"):
            pass  # Already selected
    else:
        if st.button("The Commutator C", key="case2"):
            # Set up Commutator case
            st.session_state.selected_case = "The Commutator"
            st.session_state.rotation_sequence = "R F' R' F"
            st.session_state.current_step = -1
            st.session_state.is_playing = False
            
            # Set up initial cube state (solved)
            temp_cube = Cube()
            temp_cube.reset()
            
            # Generate sequence states
            sequence_states = [np.copy(temp_cube.state)]
            moves = st.session_state.rotation_sequence.split()
            for move in moves:
                face = move[0]
                direction = -1 if "'" in move else 1
                temp_cube.turn_face(face, direction)
                sequence_states.append(np.copy(temp_cube.state))
            st.session_state.sequence_states = sequence_states
            
            # Update current cube state
            st.session_state.cube.state = np.copy(sequence_states[0])
            
            # Set up prediction question
            st.session_state.prediction_question = "After executing the sequence R F' R' F, what happens to the cube?"
            st.session_state.prediction_options = [
                "Nothing changes",
                "Most of the cube remains fixed, but specific corners and edges are swapped"
            ]
            st.session_state.correct_answer = "Most of the cube remains fixed, but specific corners and edges are swapped"
            
            # Reset answer state
            if 'student_answer' in st.session_state:
                del st.session_state.student_answer
            if 'answer_checked' in st.session_state:
                del st.session_state.answer_checked
            if 'prediction_correct' in st.session_state:
                del st.session_state.prediction_correct
            
            # Force rerun to update button styling immediately
            st.rerun()

# Button 3 - Order of an Element
with col3:
    if st.session_state.selected_case == "Order of an Element":
        # Red button for selected case
        if st.button("Order of an Element", key="case3", type="primary"):
            pass  # Already selected
    else:
        if st.button("Order of an Element", key="case3"):
            # Set up Order of an Element case
            st.session_state.selected_case = "Order of an Element"
            # Create sequence: 6 repetitions of R F' R' F
            st.session_state.rotation_sequence = "R F' R' F R F' R' F R F' R' F R F' R' F R F' R' F R F' R' F"
            st.session_state.current_step = -1
            st.session_state.is_playing = False
            
            # Set up initial cube state (solved)
            temp_cube = Cube()
            temp_cube.reset()
            
            # Generate sequence states
            sequence_states = [np.copy(temp_cube.state)]
            moves = st.session_state.rotation_sequence.split()
            for move in moves:
                face = move[0]
                direction = -1 if "'" in move else 1
                temp_cube.turn_face(face, direction)
                sequence_states.append(np.copy(temp_cube.state))
            st.session_state.sequence_states = sequence_states
            
            # Update current cube state
            st.session_state.cube.state = np.copy(sequence_states[0])
            
            # Set up prediction question with Markdown formula
            st.session_state.prediction_question = "After repeating the sequence $C = R F' R' F$ six times, which can be represented by $C^6 = e$, what is the cube's state?"
            st.session_state.prediction_options = [
                "Returned to the solved state",
                "Remained in a scrambled state"
            ]
            st.session_state.correct_answer = "Returned to the solved state"
            
            # Reset answer state
            if 'student_answer' in st.session_state:
                del st.session_state.student_answer
            if 'answer_checked' in st.session_state:
                del st.session_state.answer_checked
            if 'prediction_correct' in st.session_state:
                del st.session_state.prediction_correct
            
            # Force rerun to update button styling immediately
            st.rerun()

# Button 4 - Conjugation
with col4:
    if st.session_state.selected_case == "Conjugation":
        # Red button for selected case
        if st.button("Conjugation", key="case4", type="primary"):
            pass  # Already selected
    else:
        if st.button("Conjugation", key="case4"):
            # Set up Conjugation case
            st.session_state.selected_case = "Conjugation"
            st.session_state.rotation_sequence = "U R F' R' F U'"
            st.session_state.current_step = -1
            st.session_state.is_playing = False
            
            # Set up initial cube state (solved)
            temp_cube = Cube()
            temp_cube.reset()
            
            # Generate sequence states
            sequence_states = [np.copy(temp_cube.state)]
            moves = st.session_state.rotation_sequence.split()
            for move in moves:
                face = move[0]
                direction = -1 if "'" in move else 1
                temp_cube.turn_face(face, direction)
                sequence_states.append(np.copy(temp_cube.state))
            st.session_state.sequence_states = sequence_states
            
            # Update current cube state
            st.session_state.cube.state = np.copy(sequence_states[0])
            
            # Set up prediction question with Markdown formula
            st.session_state.prediction_question = "After applying setup move $U$, performing commutator $C = R F' R' F$, then undoing setup $U'$, what happens to the cube?"
            st.session_state.prediction_options = [
                "Returns to solved state",
                "The permutation effect of $C$ is transported to a new location on the Upper face"
            ]
            st.session_state.correct_answer = "The permutation effect of $C$ is transported to a new location on the Upper face"
            
            # Reset answer state
            if 'student_answer' in st.session_state:
                del st.session_state.student_answer
            if 'answer_checked' in st.session_state:
                del st.session_state.answer_checked
            if 'prediction_correct' in st.session_state:
                del st.session_state.prediction_correct
            
            # Force rerun to update button styling immediately
            st.rerun()

# Add a separator
st.markdown("---")


# Main content area
# If case has been selected, display case content
if 'selected_case' in st.session_state:
    # Middle divided into left and right columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Cube 3D Display")
        fig = draw_cube_3d(st.session_state.cube.state)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display randomly generated step sequence
        st.subheader("Step Sequence")
        if st.session_state.rotation_sequence:
            moves = st.session_state.rotation_sequence.split()
            # Display rotation trajectory, color executed steps and current step
            display_sequence = []
            for i, move in enumerate(moves):
                if i < st.session_state.current_step:
                    display_sequence.append(f"<span style='color: green; text-decoration: line-through;'>{move}</span>")
                elif i == st.session_state.current_step:
                    display_sequence.append(f"<span style='color: red; font-weight: bold;'>{move}</span>")
                else:
                    display_sequence.append(f"<span style='color: black;'>{move}</span>")
            
            st.markdown(" ".join(display_sequence), unsafe_allow_html=True)
        
        # Operation buttons - fast rewind x steps, rewind, forward, fast forward x steps
        st.subheader("Operation Control")
        col_controls1, col_controls2, col_controls3, col_controls4 = st.columns(4)
        
        with col_controls1:
            if st.button("Fast Rewind", help="Return to initial state"):
                # Execute fast rewind, directly return to initial state
                st.session_state.is_playing = False
                st.session_state.current_step = -1
                if st.session_state.sequence_states:
                    st.session_state.cube.state = st.session_state.sequence_states[st.session_state.current_step + 1]
                # Force rerun script to update UI immediately
                st.rerun()
                
        with col_controls2:
            if st.button("Rewind One Step", help="Rewind one step"):
                st.session_state.is_playing = False
                if st.session_state.current_step > -1:
                    st.session_state.current_step -= 1
                    if st.session_state.sequence_states:
                        st.session_state.cube.state = st.session_state.sequence_states[st.session_state.current_step + 1]
                # Force rerun script to update UI immediately
                st.rerun()
                
        with col_controls3:
            if st.button("Forward One Step", help="Forward one step"):
                st.session_state.is_playing = False
                
                # Check if can forward (constrained by step sequence)
                moves = st.session_state.rotation_sequence.split()
                if st.session_state.current_step < len(moves) - 1:
                    st.session_state.current_step += 1
                    if st.session_state.sequence_states:
                        st.session_state.cube.state = st.session_state.sequence_states[st.session_state.current_step + 1]
                else:
                    st.warning("Already at the last step!")
                # Force rerun script to update UI immediately
                st.rerun()
                
        with col_controls4:
            if st.button("Fast Forward", help="Advance to last step"):
                # Execute fast forward, directly advance to last step
                st.session_state.is_playing = False
                moves = st.session_state.rotation_sequence.split()
                new_step = len(moves) - 1
                if new_step != st.session_state.current_step:
                    st.session_state.current_step = new_step
                    if st.session_state.sequence_states:
                        st.session_state.cube.state = st.session_state.sequence_states[st.session_state.current_step + 1]
                # Force rerun script to update UI immediately
                st.rerun()
        
        # Question display area
        st.subheader("Prediction Question")
        
        # Display the step sequence for the current case
        st.write(f"Step sequence involved in the current question:")
        st.code(st.session_state.rotation_sequence, language="text")
        
        # Display question with Markdown rendering for formulas
        if 'prediction_question' in st.session_state and st.session_state.prediction_question:
            st.markdown(st.session_state.prediction_question)
            
            if 'prediction_options' in st.session_state and st.session_state.prediction_options:
                # Use dynamic key to ensure selected state resets when switching questions
                if 'question_counter' not in st.session_state:
                    st.session_state.question_counter = 0
                
                # Use radio component for students to select answer
                student_answer = st.radio(
                    "Please select your answer:",
                    st.session_state.prediction_options,
                    index=None,
                    key=f"student_answer_radio_{st.session_state.question_counter}"
                )
                
                # Save student answer
                if student_answer is not None:
                    st.session_state.student_answer = student_answer
        
        # Check answer and switch to next question buttons
        st.write("\n")
        with st.container():
            if st.button("Check Answer"):
                if 'student_answer' in st.session_state and st.session_state.student_answer is not None:
                    st.session_state.answer_checked = True
                    
                    # Determine if prediction is correct
                    if 'correct_answer' in st.session_state:
                        if st.session_state.student_answer == st.session_state.correct_answer:
                            st.success("Prediction correct!")
                            st.session_state.prediction_correct = True
                            
                            # Show case-specific thinking prompt with emoji
                            if st.session_state.selected_case == "Proving Non-Commutativity":
                                st.info("🤔 **Think about it:** Identify the pieces that moved differently. Why does order matter?")
                            elif st.session_state.selected_case == "The Commutator":
                                st.info(f"🤔 **Think about it:** This is an instance of a commutator! Formula: $C = R F' R' F$ (\"Out, Out, In\"). Notice how most of the cube remains fixed.")
                            elif st.session_state.selected_case == "Order of an Element":
                                st.info("🤔 **Think about it:** Why did it take 6 cycles to return to identity? (Hint: Look at the 3-cycle of edges and 2-cycle of corners)")
                            elif st.session_state.selected_case == "Conjugation":
                                st.info("""🤔 **Think about it:** 
- **Algebraic inverse definition:** If $S$ is a move, $S'$ (read 'S inverse') is a move that undoes the effect of $S$. Mathematically, $S S' = e$ (identity).
- **Conjugation frame:** This sequence is an example of conjugation: $S A S'$, where $S = U$ (setup move), $A = C = R F' R' F$ (commutator), and $S' = U'$ (undo setup).
- Conjugation allows us to transport the effect of $A$ to a new location!""")
                        else:
                            st.error("Prediction incorrect! Please try again!")
                            st.session_state.prediction_correct = False
                            # st.write(f"Correct answer is: {st.session_state.correct_answer}")
                else:
                    st.warning("Please select an answer first!")