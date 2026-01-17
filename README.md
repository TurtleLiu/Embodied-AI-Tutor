# 🧠 Embodied AI Rubik's Cube Tutor

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-blue?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An interactive web application that transforms Rubik's Cube manipulation into learnable algebraic group structure through embodied AI tutoring. This project implements the educational framework from the paper **"An Embodied AI Tutor that Turns Rubik’s Cube Interaction into Learnable Group Structure"**, demonstrating how abstract mathematical concepts can emerge from concrete physical interaction.

**Live Demo:** [https://cube-tutor.streamlit.app/](https://cube-tutor.streamlit.app/)

---

## 🎯 Overview

| Aspect | Details |
|--------|---------|
| **Language** | Python 3.8+ |
| **UI Framework** | Streamlit |
| **Visualization** | Plotly & Matplotlib |
| **Data Processing** | NumPy & Pandas |
| **Main File** | `app.py` |
| **Research Based** | Luo et al. (2026) *Nature Science of Learning* |

> [!NOTE]
> This tutor provides an interactive learning environment that translates Rubik's Cube operations into algebraic group theory concepts through reversible micro-experiments.

---

## ✨ Features

- ✅ **3D Cube Visualization** — Interactive 3D rendering of Rubik's Cube with Plotly
- ✅ **Action-to-Operator Mapping** — Translate face turns into algebraic operators
- ✅ **Reversible Micro-Experiments** — Pause, step, rewind, and replay cube operations
- ✅ **Identity and Inverse Demonstrations** — Visualize reversible operations and their properties
- ✅ **Composition and Non-Commutativity** — Explore order-dependent operation results
- ✅ **Reusable Composites (Macro-Operators)** — Identify and reuse meaningful move sequences
- ✅ **Adaptive Learning Paths** — Multiple difficulty levels for different learning stages
- ✅ **Predictive Assessment** — Generate questions to test understanding of group theory concepts
- ✅ **Responsive Dashboard** — Clean, intuitive UI with real-time cube manipulation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/TurtleLiu/Rubiks-Cube-AI-Tutor.git
   cd Rubiks-Cube-AI-Tutor
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

5. **Open in browser:**
   Streamlit will print a URL (typically `http://localhost:8501`). Open it in your browser.

---

## 🏗️ Architecture

### Three-Tier Learning Framework

```
┌─────────────────────────────────────────────┐
│       Embodied Interaction Layer            │
│  ┌──────────────────────────────────────┐   │
│  │ 3D Cube Visualization                │   │
│  │ Real-time Operation Execution         │   │
│  │ Interactive Control Interface         │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│    Symbolic Representation Layer            │
│  ┌──────────────────────────────────────┐   │
│  │ Operator Mapping (Action → Symbol)     │   │
│  │ Sequence Composition Analysis         │   │
│  │ Inverse Operation Detection           │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│    Conceptual Learning Layer                │
│  ┌──────────────────────────────────────┐   │
│  │ Predictive Question Generation        │   │
│  │ Group Property Identification        │   │
│  │ Misconception Detection               │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Learning Pipeline

```
Micro-Experiment → Prediction → Observation → Explanation → Concept Reinforcement
        │               │             │              │                    │
    Cube Operation   Learner Guess   Result Display  Student Explanation  Group Theory Concept
        │               │             │              │                    │
   Reversible       Immediate       Visual         Formative          Operationalized
   Execution        Feedback        Comparison      Assessment         Mathematical Ideas
```

---

## 📚 Learning Objectives

### LG1: Action-to-operator mapping
Learners can treat a face turn as an **operator** and predict qualitative consequences of composing operators (e.g., "apply R then U").

### LG2: Identity and inverse as testable phenomena
Learners can recognize and explain reversibility via short experiments (e.g., $a a^{-1}$ returns to the same state), rather than memorizing "prime" notation.

### LG3: Composition and non-commutativity
Learners can generate and interpret counterexamples showing $ab \neq ba$ for cube moves, and articulate what changes (and what stays invariant) under each order.

### LG4: Reusable composites (macro-operators)
Learners can identify and reuse short move sequences as meaningful units ("do this commutator"), connecting procedural chunks to algebraic composition.

---

## 🧠 Tutor Functionality

### 1. 🎮 **Action-to-Operator Mapping**
- Interactive cube manipulation interface
- Real-time translation of physical actions to symbolic operators
- Visual feedback on operation effects

### 2. 🔄 **Identity and Inverse Operations**
- Demonstrate reversible operation sequences
- Visual comparison of operation and inverse operation effects
- Test for identity preservation

### 3. ⚡ **Composition and Non-Commutativity**
- Compare operation sequences in different orders
- Visualize non-commutative behavior
- Highlight invariant properties

### 4. 🧩 **Macro-Operator Identification**
- Recognize and label reusable move sequences
- Demonstrate composition of complex operations
- Connect procedural chunks to algebraic concepts

---

## 📊 Evaluation Metrics

### **Conceptual Understanding** ↑
- **Definition**: Ability to map cube operations to algebraic concepts
- **Measurement**: Predictive accuracy on operation outcomes
- **Ideal**: High prediction success rate across concept types

### **Procedural Fluency** ↑
- **Definition**: Ability to execute and reverse operation sequences
- **Measurement**: Speed and accuracy of cube manipulation
- **Ideal**: Smooth execution with minimal errors

### **Transfer Performance** ↑
- **Definition**: Application of learned concepts to new contexts
- **Measurement**: Success on novel operation sequences
- **Ideal**: Strong performance on unfamiliar problems

### **Cognitive Load Reduction** ↓
- **Definition**: Manageability of information processing demands
- **Measurement**: Time spent and errors made during complex operations
- **Ideal**: Efficient problem-solving with low error rates

---

## 🎓 Learning Workflow

### Step 1: Concept Selection
1. Choose a learning objective (LG1-LG4)
2. Select difficulty level (Simple, Medium, Hard)
3. Generate a micro-experiment

### Step 2: Predictive Exploration
1. Observe the initial cube state
2. Predict the outcome of operation sequences
3. Test predictions through step-by-step execution

### Step 3: Interactive Manipulation
1. Use control buttons to manipulate the cube
   - Step Forward/Backward through sequences
   - Rewind/Fast Forward to specific states
   - Compare different operation orders

### Step 4: Formative Assessment
1. Answer prediction questions about operation outcomes
2. Receive immediate feedback on responses
3. Identify and correct misconceptions

### Step 5: Concept Reinforcement
1. Review operation effects and algebraic properties
2. Practice applying concepts to new scenarios
3. Build reusable macro-operators

---

## 📁 Code Structure

```
app.py
├── Cube Core
│   ├── Cube class with rotation logic
│   ├── Quaternion-based 3D rotation
│   ├── Color and face mapping
│   └── Operation sequence handling
│
├── Learning Framework
│   ├── Micro-experiment generation
│   ├── Concept-based case selection
│   ├── Difficulty level adaptation
│   └── Predictive question engine
│
├── Visualization Functions
│   ├── 3D cube rendering with Plotly
│   ├── Interactive cube manipulation
│   ├── Operation sequence visualization
│   └── Comparison view for non-commutativity
│
├── UI Components
│   ├── Sidebar for case selection
│   ├── Main cube visualization area
│   ├── Operation control buttons
│   └── Assessment question interface
│
└── Streamlit App
    ├── Session state management
    ├── Two-column layout design
    ├── Real-time cube state updates
    └── Formative assessment interface
```

---

## 🔧 Configuration & Tuning

### Learning Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Case Type** | LG1-LG4 | LG1 | Learning objective focus |
| **Difficulty Level** | Simple/Medium/Hard | Simple | Complexity of operation sequences |
| **Sequence Length** | 2-10 | 4 | Number of operations in sequence |
| **Initial Cube State** | Solved/Scrambled | Solved | Starting configuration |
| **Visualization Speed** | Slow/Normal/Fast | Normal | Animation speed of operations |

### Performance Optimization

> [!TIP]
> For optimal performance:
> - Use modern browser (Chrome/Firefox recommended)
> - Ensure sufficient GPU memory for 3D visualization
> - Close other browser tabs to reduce resource usage

---

## 📚 Academic Context

This tutor implements the framework from:

**Luo, T., et al. (2026).** *An Embodied AI Tutor that Turns Rubik’s Cube Interaction into Learnable Group Structure. Nature Science of Learning.*

### Key Research Contributions:
1. **Embodied Learning Design**: Transforms concrete cube operations into abstract algebraic concepts
2. **Distributed Inquiry Cycle**: Creates a teacher-student-agent triad for collaborative learning
3. **Reversible Micro-Experiments**: Enables exploration through pause, step, rewind, and replay
4. **Cognitive Load Management**: Makes comparisons visible and repeatable to reduce learning barriers
5. **Symbolic Readouts**: Exposes operational tests for group properties (identity, inverses, commutativity)

### Abstract
How do abstract algebraic ideas emerge from concrete action, and what role can educational agents play in that process? We introduce an **embodied AI Rubik’s Cube tutor** that reframes group theory learning as an inquiry cycle distributed across a teacher–student–agent triad. Rather than presenting definitions first, the agent orchestrates short, reversible micro-experiments on a physical cube—pause, step, rewind, replay, and slow-motion—that invite learners to predict outcomes, test hypotheses, and explain contrasts. This design targets core cognitive hurdles of early group reasoning (objectifying operations, definition checking, and translating between embodied and symbolic representations), while managing cognitive load by making comparisons visible and repeatable.

Technically, the tutor learns from action–consequence data. From tuples $(o_t,a_t,o_{t+1})$, it trains a compact transition model and exposes **symbolic readouts**—operational tests for identity/inverses, composition equivalence, and commuting vs. non-commuting move pairs. These readouts drive formative prompts and misconception-focused contrasts, positioning the agent as an experiment generator and classroom assistant rather than a black-box solver.

---

## 🚀 Deployment

### Local Deployment
Follow the [Quick Start](#quick-start) instructions above.

### Streamlit Cloud Deployment
1. Push code to GitHub repository
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set `app.py` as main file
4. Deploy with default settings

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🔮 Future Extensions

### Planned Features
- [ ] **Physical Cube Integration** — Connect to real Rubik's Cube hardware
- [ ] **Personalized Learning Paths** — Adaptive difficulty based on performance
- [ ] **Multi-language Support** — Extend to additional languages
- [ ] **Export Functionality** — Save learning progress and experiment results
- [ ] **Collaborative Learning** — Multi-user learning sessions

### Research Extensions
- [ ] **Generalization to Other Groups** — Apply framework to other algebraic structures
- [ ] **Long-term Learning Studies** — Track knowledge retention over time
- [ ] **Cross-cultural Evaluation** — Test with diverse learner populations
- [ ] **Teacher Dashboard** — Monitor and guide student progress

---

## 🤝 Contributing

We welcome contributions to enhance this educational tool:

1. **Report Issues**: Use GitHub Issues to report bugs or suggest features
2. **Submit Pull Requests**: Implement improvements or new features
3. **Share Use Cases**: How are you using this tutor in teaching or research?

### Development Guidelines
- Follow PEP 8 coding standards
- Include docstrings for all functions
- Add tests for new functionality
- Update documentation accordingly

---

## 📦 Dependencies

```
streamlit>=1.28.0
numpy>=1.24.0
plotly>=5.17.0
matplotlib>=3.7.0
pillow>=9.0.0
```

See `requirements.txt` for exact version specifications.

---

## 📄 License

This project is licensed under the **MIT License**.

```
Copyright (c) 2026 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## 📞 Contact & Citation

For questions about the research:
- **Corresponding Author**: Tiejian Luo (tjluo@ucas.ac.cn)

If you use this tutor in your research or teaching, please cite:

```bibtex
@article{luo2026embodied,
  title={An Embodied AI Tutor that Turns Rubik's Cube Interaction into Learnable Group Structure},
  author={Luo, Tiejian and others},
  journal={Nature Science of Learning},
  year={2026},
  publisher={Nature Research}
}
```

---

*This tutor is an educational tool for exploring algebraic concepts through embodied interaction. While based on peer-reviewed research, it uses simplified representations for educational purposes.*