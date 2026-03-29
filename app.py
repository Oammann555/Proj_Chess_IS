import streamlit as st
import joblib
import numpy as np
import chess
import chess.svg

# Load models
ensemble = joblib.load("ensemble_model.pkl")
nn_model = joblib.load("nn_model_sklearn.pkl")

# Piece values for material calculation
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

def extract_features(fen):
    board = chess.Board(fen)

    material = 0
    for piece_type in piece_values:
        material += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        material -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    mobility = board.legal_moves.count()

    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    center = sum(board.is_attacked_by(chess.WHITE, sq) for sq in center_squares) - \
             sum(board.is_attacked_by(chess.BLACK, sq) for sq in center_squares)

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)

    king_safety = int(board.is_attacked_by(chess.BLACK, white_king)) - \
                  int(board.is_attacked_by(chess.WHITE, black_king))

    return np.array([[material, mobility, center, king_safety]])


# ---------------- UI ----------------
st.title("Chess AI Evaluation System")

page = st.sidebar.selectbox(
    "Select Page",
    ["ML Model", "Neural Network", "Test ML", "Test NN", "Interactive Board"]
)

# ---------------- PAGE 1: ML Model ----------------
if page == "ML Model":
    st.header("Machine Learning Model (Ensemble)")

    st.subheader("1. Data Preparation")
    st.write("""
    This project uses two chess datasets:

    - **chessData.csv** — Chess positions in FEN format, each paired with an evaluation score computed by the Stockfish engine.
    - **chess_games.csv** — Historical chess game records downloaded from Kaggle.

    **Preparation steps:**
    - Merged both datasets and retained only the `fen` and `eval` columns.
    - Removed rows with null values and positions with mate scores (denoted by `#`), since mate scores are not numeric and cannot be used for regression.
    - Converted evaluation values from centipawns to pawns by dividing by 100.
    - Clamped evaluation scores to the range [-10, +10] to reduce the impact of extreme outliers.
    - Extracted 4 features from each FEN string: Material Balance, Mobility, Center Control, and King Safety.
    - Sampled 2,000 rows to keep the dataset size manageable for training.
    """)

    st.subheader("2. Algorithm Theory")
    st.write("""
    The model is a **Voting Regressor**, an Ensemble Learning technique that combines multiple base models.
    Each base model produces an independent prediction, and the final output is the average of all predictions.
    This averaging process reduces variance and generally produces more stable and accurate results than any single model alone.

    **Three base models used in the Voting Regressor:**

    **Random Forest Regressor**
    - Builds multiple Decision Trees using Bootstrap Sampling (random subsets of training data).
    - Each tree independently predicts a score, and the final prediction is the mean across all trees.
    - Effective at reducing overfitting due to the averaging of many uncorrelated trees.

    **Gradient Boosting Regressor**
    - Builds models sequentially, where each new model focuses on correcting the residual errors of the previous one.
    - Uses Gradient Descent to minimize the loss function at each step.
    - Performs well on structured data with complex non-linear patterns.

    **Linear Regression**
    - Models the relationship between features and target as a linear function.
    - Acts as a simple, low-variance base estimator that stabilizes the overall ensemble predictions.
    """)

    st.subheader("3. Model Development Process")
    st.write("""
    1. Extracted 4 features from each FEN string:
       - **Material Balance** — Difference in total piece value between White and Black (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9).
       - **Mobility** — Number of legal moves available to the current player.
       - **Center Control** — Number of center squares (d4, e4, d5, e5) attacked by each side, expressed as a net difference.
       - **King Safety** — Whether each king is currently under attack (1 if attacked, 0 if not), expressed as a net difference.
    2. Split data into training and test sets using an 80:20 ratio via `train_test_split`.
    3. Constructed a `VotingRegressor` combining Random Forest, Gradient Boosting, and Linear Regression.
    4. Trained the model on the training set (X_train, y_train).
    5. Evaluated performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE) on the test set.
    6. Saved the trained model using `joblib`.
    """)

    st.subheader("4. References")
    st.write("""
    - Kaggle Chess Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
    - Scikit-learn VotingRegressor: https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor
    - Python-chess Library: https://python-chess.readthedocs.io/
    - Stockfish Chess Engine: https://stockfishchess.org/
    """)

# ---------------- PAGE 2: Neural Network ----------------
elif page == "Neural Network":
    st.header("Neural Network Model")

    st.subheader("1. Data Preparation")
    st.write("""
    The same dataset and preparation pipeline used for the ML Model was applied here:
    - Merged chessData.csv and chess_games.csv.
    - Removed null values and mate scores.
    - Converted evaluation from centipawns to pawns and clamped to [-10, +10].
    - Extracted 4 features: Material Balance, Mobility, Center Control, King Safety.
    - Applied an 80:20 train/test split.
    """)

    st.subheader("2. Algorithm Theory")
    st.write("""
    The model is a **Multilayer Perceptron (MLP)**, a feedforward Neural Network capable of learning
    non-linear relationships between input features and the target evaluation score.

    **Model Architecture:**
    - Input Layer: 4 neurons (Material Balance, Mobility, Center Control, King Safety)
    - Hidden Layer 1: 64 neurons with ReLU activation
    - Hidden Layer 2: 32 neurons with ReLU activation
    - Output Layer: 1 neuron (predicted evaluation score)

    **ReLU Activation Function:**
    Defined as f(x) = max(0, x). ReLU introduces non-linearity into the network, enabling it to learn
    complex patterns. It also mitigates the vanishing gradient problem commonly associated with Sigmoid
    and Tanh activations, which makes training deeper networks more stable.

    **Loss Function:**
    Mean Squared Error (MSE) is used to measure the average squared difference between predicted and
    actual evaluation scores. Minimizing MSE during training drives the model to produce more accurate predictions.
    """)

    st.subheader("3. Model Development Process")
    st.write("""
    1. Used the same 4 features and target variable as the ML Model.
    2. Built an `MLPRegressor` using Scikit-learn with the following configuration:
       - `hidden_layer_sizes = (64, 32)`
       - `activation = 'relu'`
       - `max_iter = 500`
    3. Trained the model on X_train and y_train.
    4. Evaluated performance using MAE and MSE on X_test.
    5. Compared results against the Ensemble Model to assess relative accuracy.
    6. Saved the trained model using `joblib`.
    """)

    st.subheader("4. References")
    st.write("""
    - Scikit-learn MLPRegressor: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    - Kaggle Chess Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
    - Python-chess Library: https://python-chess.readthedocs.io/
    - ReLU Activation: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """)

# ---------------- PAGE 3: Test ML ----------------
elif page == "Test ML":
    st.header("Test Ensemble Model")

    st.info("Enter a FEN string representing a chess position and click Predict to receive an evaluation score from the Ensemble model.")

    fen = st.text_input("Enter FEN", placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    if st.button("Predict (ML)"):
        try:
            features = extract_features(fen)
            pred = ensemble.predict(features)[0]
            st.success(f"Evaluation Score: {pred:.2f}")
            if pred > 0:
                st.write("White is better")
            elif pred < 0:
                st.write("Black is better")
            else:
                st.write("Equal position")
        except Exception as e:
            st.error(f"Invalid FEN or prediction error: {e}")

# ---------------- PAGE 4: Test NN ----------------
elif page == "Test NN":
    st.header("Test Neural Network")

    st.info("Enter a FEN string representing a chess position and click Predict to receive an evaluation score from the Neural Network model.")

    fen = st.text_input("Enter FEN", placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    if st.button("Predict (NN)"):
        try:
            features = extract_features(fen)
            pred_nn = nn_model.predict(features)[0]
            st.success(f"Evaluation Score: {pred_nn:.2f}")
            if pred_nn > 0:
                st.write("White is better")
            elif pred_nn < 0:
                st.write("Black is better")
            else:
                st.write("Equal position")
        except Exception as e:
            st.error(f"Invalid FEN or prediction error: {e}")

# ---------------- PAGE 5: Interactive Board ----------------
elif page == "Interactive Board":
    st.header("Interactive Board Evaluation")
    st.write(
        "Drag and drop pieces to set up any position. "
        "The FEN updates automatically — click **Copy FEN**, "
        "paste it below, then evaluate with either model."
    )

    # --- Preset selector ---
    preset_fens = {
        "Starting position":
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Sicilian Defense (after 1.e4 c5)":
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "Ruy Lopez (after 1.e4 e5 2.Nf3 Nc6 3.Bb5)":
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "Queen's Gambit (after 1.d4 d5 2.c4)":
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
        "King's Indian Defense (after 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7)":
            "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    }

    preset = st.selectbox("Load a preset position", list(preset_fens.keys()))
    init_fen = preset_fens[preset]
    # Extract just the position part for chessboard.js (first token)
    init_pos = init_fen.split()[0]

    # --- Interactive board HTML (chessboard.js + chess.js) ---
    board_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', sans-serif;
    background: transparent;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px;
    gap: 10px;
  }}
  #board {{ width: 400px; }}
  .controls {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
  }}
  button {{
    padding: 7px 18px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: filter 0.15s;
  }}
  button:hover {{ filter: brightness(1.1); }}
  .btn-secondary {{ background: #374151; color: #f3f4f6; }}
  .btn-primary   {{ background: #2563eb; color: #ffffff; }}
  .side-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: #374151;
  }}
  .side-btn {{
    padding: 4px 14px;
    border-radius: 20px;
    border: 1.5px solid #9ca3af;
    background: #f9fafb;
    color: #374151;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
  }}
  .side-btn.active {{
    background: #1d4ed8;
    border-color: #1d4ed8;
    color: #fff;
  }}
  #fen-box {{
    width: 100%;
    max-width: 440px;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: monospace;
    font-size: 11.5px;
    color: #1e3a5f;
    word-break: break-all;
    text-align: center;
    user-select: all;
  }}
  #copy-msg {{
    font-size: 12px;
    color: #16a34a;
    min-height: 16px;
  }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
</head>
<body>

<div id="board"></div>

<div class="side-row">
  Side to move:
  <button class="side-btn active" id="btn-w" onclick="setSide('w')">White</button>
  <button class="side-btn"        id="btn-b" onclick="setSide('b')">Black</button>
</div>

<div id="fen-box">Loading...</div>

<div class="controls">
  <button class="btn-secondary" onclick="resetBoard()">Reset</button>
  <button class="btn-secondary" onclick="board.flip()">Flip</button>
  <button class="btn-secondary" onclick="clearBoard()">Clear</button>
  <button class="btn-primary"   onclick="copyFen()">Copy FEN</button>
</div>
<div id="copy-msg"></div>

<script>
  var side = 'w';

  function getFen() {{
    return board.fen() + ' ' + side + ' KQkq - 0 1';
  }}

  function updateFenBox() {{
    document.getElementById('fen-box').textContent = getFen();
  }}

  function setSide(s) {{
    side = s;
    document.getElementById('btn-w').classList.toggle('active', s === 'w');
    document.getElementById('btn-b').classList.toggle('active', s === 'b');
    updateFenBox();
  }}

  function resetBoard() {{
    board.position('{init_pos}', false);
    setSide('{init_fen.split()[1]}');
  }}

  function clearBoard() {{
    board.position('8/8/8/8/8/8/8/8', false);
    updateFenBox();
  }}

  function copyFen() {{
    var fen = getFen();
    navigator.clipboard.writeText(fen).then(function() {{
      var el = document.getElementById('copy-msg');
      el.textContent = 'Copied! Paste it in the field below.';
      setTimeout(function() {{ el.textContent = ''; }}, 2500);
    }});
  }}

  // Map piece codes to lichess CDN SVG images (avoids template URL issues)
  var pieceMap = {{
    'wK': 'https://lichess1.org/assets/piece/cburnett/wK.svg',
    'wQ': 'https://lichess1.org/assets/piece/cburnett/wQ.svg',
    'wR': 'https://lichess1.org/assets/piece/cburnett/wR.svg',
    'wB': 'https://lichess1.org/assets/piece/cburnett/wB.svg',
    'wN': 'https://lichess1.org/assets/piece/cburnett/wN.svg',
    'wP': 'https://lichess1.org/assets/piece/cburnett/wP.svg',
    'bK': 'https://lichess1.org/assets/piece/cburnett/bK.svg',
    'bQ': 'https://lichess1.org/assets/piece/cburnett/bQ.svg',
    'bR': 'https://lichess1.org/assets/piece/cburnett/bR.svg',
    'bB': 'https://lichess1.org/assets/piece/cburnett/bB.svg',
    'bN': 'https://lichess1.org/assets/piece/cburnett/bN.svg',
    'bP': 'https://lichess1.org/assets/piece/cburnett/bP.svg'
  }};

  var board = Chessboard('board', {{
    draggable: true,
    position: '{init_pos}',
    onSnapEnd: updateFenBox,
    pieceTheme: function(piece) {{ return pieceMap[piece]; }}
  }});

  updateFenBox();
</script>
</body>
</html>
"""

    st.components.v1.html(board_html, height=580, scrolling=False)

    # --- FEN input + evaluation ---
    st.write("---")
    st.subheader("Evaluate the Position")

    fen_input = st.text_input(
        "Paste FEN from board above",
        placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        key="fen_eval_input"
    )

    if fen_input.strip():
        try:
            board_obj = chess.Board(fen_input.strip())
            features  = extract_features(fen_input.strip())

            # Preview + stats side by side
            col_board, col_stats = st.columns([1, 1])
            with col_board:
                st.write("**Position Preview**")
                svg = chess.svg.board(board_obj, size=280)
                st.components.v1.html(svg, height=295)

            with col_stats:
                st.write("**Position Info**")
                st.metric("Legal Moves",  board_obj.legal_moves.count())
                st.metric("In Check",     "Yes" if board_obj.is_check() else "No")
                st.metric("Full Move",    board_obj.fullmove_number)
                f = features[0]
                st.write(f"Material Balance: **{f[0]:+.0f}**")
                st.write(f"Mobility: **{f[1]}** moves")
                st.write(f"Center Control: **{f[2]:+.0f}**")
                st.write(f"King Safety: **{f[3]:+.0f}**")

            # Evaluate buttons
            st.write("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                run_ml  = st.button("Ensemble (ML)",     use_container_width=True)
            with c2:
                run_nn  = st.button("Neural Network",    use_container_width=True)
            with c3:
                run_both = st.button("Compare Both",     use_container_width=True)

            def advantage_label(s):
                if   s >  1.0: return "White has a clear advantage"
                elif s >  0.2: return "White has a slight advantage"
                elif s < -1.0: return "Black has a clear advantage"
                elif s < -0.2: return "Black has a slight advantage"
                else:          return "Position is roughly equal"

            if run_ml or run_both:
                st.session_state["ml_result"] = ensemble.predict(features)[0]
            if run_nn or run_both:
                st.session_state["nn_result"] = nn_model.predict(features)[0]

            if "ml_result" in st.session_state or "nn_result" in st.session_state:
                st.write("### Evaluation Results")
                r1, r2 = st.columns(2)
                if "ml_result" in st.session_state:
                    s = st.session_state["ml_result"]
                    with r1:
                        st.metric("Ensemble (ML)", f"{s:+.2f} pawns")
                        st.caption(advantage_label(s))
                if "nn_result" in st.session_state:
                    s = st.session_state["nn_result"]
                    with r2:
                        st.metric("Neural Network", f"{s:+.2f} pawns")
                        st.caption(advantage_label(s))

                if "ml_result" in st.session_state and "nn_result" in st.session_state:
                    diff = abs(st.session_state["ml_result"] - st.session_state["nn_result"])
                    if diff < 0.3:
                        st.success(f"Both models agree on the evaluation (difference: {diff:.2f})")
                    else:
                        st.warning(f"Models give different evaluations (difference: {diff:.2f}). This position may be complex or unusual.")

        except Exception as e:
            st.error(f"Invalid FEN: {e}")
