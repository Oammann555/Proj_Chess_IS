import streamlit as st
import joblib
import numpy as np
import chess
import chess.svg

# Load models
ensemble = joblib.load("ensemble_model.pkl")
nn_model = joblib.load("nn_model_sklearn.pkl")

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

    current_moves = board.legal_moves.count()
    try:
        board.push(chess.Move.null())
        opponent_moves = board.legal_moves.count()
        board.pop()
    except Exception:
        opponent_moves = 0

    if board.turn == chess.WHITE:
        mobility = current_moves - opponent_moves
    else:
        mobility = opponent_moves - current_moves

    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    center = (
        sum(board.is_attacked_by(chess.WHITE, sq) for sq in center_squares) -
        sum(board.is_attacked_by(chess.BLACK, sq) for sq in center_squares)
    )

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    king_safety = (
        int(board.is_attacked_by(chess.BLACK, white_king)) -
        int(board.is_attacked_by(chess.WHITE, black_king))
    )

    return np.array([[material, mobility, center, king_safety]])


def advantage_label(score):
    if   score >  2.0: return "White has a decisive advantage."
    elif score >  0.5: return "White has a clear advantage."
    elif score >  0.1: return "White has a slight advantage."
    elif score < -2.0: return "Black has a decisive advantage."
    elif score < -0.5: return "Black has a clear advantage."
    elif score < -0.1: return "Black has a slight advantage."
    else:              return "The position is roughly equal."


PRESET_FENS = {
    "Starting Position":
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


def make_board_html(init_pos, init_side):
    return f"""<!DOCTYPE html>
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
  .row {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
  }}
  button {{
    padding: 7px 16px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: filter 0.15s;
  }}
  button:hover {{ filter: brightness(1.12); }}
  .btn-gray  {{ background: #374151; color: #f3f4f6; }}
  .btn-blue  {{ background: #2563eb; color: #ffffff; }}
  .side-btn {{
    padding: 5px 16px;
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
    text-align: center;
  }}
  .label {{
    font-size: 12px;
    color: #6b7280;
  }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
</head>
<body>

<div id="board"></div>

<div class="row">
  <span class="label">Side to move:</span>
  <button class="side-btn {'active' if init_side == 'w' else ''}" id="btn-w" onclick="setSide('w')">White</button>
  <button class="side-btn {'active' if init_side == 'b' else ''}" id="btn-b" onclick="setSide('b')">Black</button>
</div>

<div id="fen-box">Loading...</div>

<div class="row">
  <button class="btn-gray" onclick="resetBoard()">Reset</button>
  <button class="btn-gray" onclick="board.flip()">Flip</button>
  <button class="btn-gray" onclick="clearBoard()">Clear</button>
  <button class="btn-blue" onclick="copyFen()">Copy FEN</button>
</div>
<div id="copy-msg"></div>

<script>
  var side = '{init_side}';

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
    setSide('{init_side}');
  }}
  function clearBoard() {{
    board.position('8/8/8/8/8/8/8/8', false);
    updateFenBox();
  }}
  function copyFen() {{
    navigator.clipboard.writeText(getFen()).then(function() {{
      var el = document.getElementById('copy-msg');
      el.textContent = 'Copied! Paste it in the field below.';
      setTimeout(function() {{ el.textContent = ''; }}, 2500);
    }});
  }}

  var pieceMap = {{
    'wK':'https://lichess1.org/assets/piece/cburnett/wK.svg',
    'wQ':'https://lichess1.org/assets/piece/cburnett/wQ.svg',
    'wR':'https://lichess1.org/assets/piece/cburnett/wR.svg',
    'wB':'https://lichess1.org/assets/piece/cburnett/wB.svg',
    'wN':'https://lichess1.org/assets/piece/cburnett/wN.svg',
    'wP':'https://lichess1.org/assets/piece/cburnett/wP.svg',
    'bK':'https://lichess1.org/assets/piece/cburnett/bK.svg',
    'bQ':'https://lichess1.org/assets/piece/cburnett/bQ.svg',
    'bR':'https://lichess1.org/assets/piece/cburnett/bR.svg',
    'bB':'https://lichess1.org/assets/piece/cburnett/bB.svg',
    'bN':'https://lichess1.org/assets/piece/cburnett/bN.svg',
    'bP':'https://lichess1.org/assets/piece/cburnett/bP.svg'
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
</html>"""


def render_test_page(model, model_name, key_prefix):
    """Full test page: drag-and-drop board + FEN paste + evaluate."""

    # --- Preset selector ---
    preset = st.selectbox(
        "Load a preset position",
        list(PRESET_FENS.keys()),
        key=f"{key_prefix}_preset"
    )
    init_fen  = PRESET_FENS[preset]
    init_pos  = init_fen.split()[0]
    init_side = init_fen.split()[1]

    # --- Drag-and-drop board ---
    st.write("Drag and drop pieces to set up your position, then click **Copy FEN** and paste it below.")
    st.components.v1.html(make_board_html(init_pos, init_side), height=555, scrolling=False)

    st.divider()

    # --- FEN input ---
    fen_input = st.text_input(
        "Paste FEN here",
        placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        key=f"{key_prefix}_fen"
    )

    if not fen_input.strip():
        st.info("Copy the FEN from the board above and paste it here to evaluate.")
        return

    try:
        board_obj = chess.Board(fen_input.strip())
    except Exception as e:
        st.error(f"Invalid FEN: {e}")
        return

    # --- Board preview + stats ---
    col_svg, col_info = st.columns([1, 1])

    with col_svg:
        st.write("**Position Preview**")
        svg = chess.svg.board(board_obj, size=280)
        st.components.v1.html(svg, height=295)

    with col_info:
        st.write("**Position Info**")
        st.metric("Legal Moves", board_obj.legal_moves.count())
        st.metric("In Check",    "Yes" if board_obj.is_check() else "No")
        st.metric("Full Move",   board_obj.fullmove_number)

        features = extract_features(fen_input.strip())
        f = features[0]
        st.write(f"Material Balance: **{f[0]:+.0f}**")
        st.write(f"Net Mobility: **{f[1]:+.0f}**")
        st.write(f"Center Control: **{f[2]:+.0f}**")
        st.write(f"King Safety: **{f[3]:+.0f}**")

    # --- Terminal position check ---
    if board_obj.is_checkmate():
        winner = "Black" if board_obj.turn == chess.WHITE else "White"
        st.error(f"Checkmate — {winner} wins. Evaluation is not meaningful for terminal positions.")
        return
    if board_obj.is_stalemate():
        st.warning("Stalemate — the game is drawn.")
        return

    # --- Evaluate button ---
    st.divider()
    if st.button(f"Evaluate with {model_name}", key=f"{key_prefix}_btn", use_container_width=True):
        features = extract_features(fen_input.strip())
        score    = model.predict(features)[0]
        st.session_state[f"{key_prefix}_score"] = score

    if f"{key_prefix}_score" in st.session_state:
        score = st.session_state[f"{key_prefix}_score"]
        st.metric(f"{model_name} Evaluation", f"{score:+.2f} pawns")
        if score > 0:
            st.success(advantage_label(score))
        elif score < 0:
            st.error(advantage_label(score))
        else:
            st.info(advantage_label(score))


# ──────────────────────────────────────────────
#  MAIN UI
# ──────────────────────────────────────────────
st.title("Chess AI Evaluation System")

page = st.sidebar.selectbox(
    "Select Page",
    ["ML Model", "Neural Network", "Test ML", "Test NN"]
)

# ── PAGE 1: ML Model description ──────────────
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
    - Extracted 4 features from each FEN string: Material Balance, Net Mobility, Center Control, and King Safety.
    - Sampled 2,000 rows to keep the dataset size manageable for training.
    """)

    st.subheader("2. Algorithm Theory")
    st.write("""
    The model is a **Voting Regressor**, an Ensemble Learning technique that combines multiple base models.
    Each base model produces an independent prediction, and the final output is the average of all predictions.
    This averaging process reduces variance and generally produces more stable results than any single model alone.

    **Random Forest Regressor**
    Builds multiple Decision Trees using Bootstrap Sampling. Each tree independently predicts a score,
    and the final prediction is the mean across all trees. Effective at reducing overfitting.

    **Gradient Boosting Regressor**
    Builds models sequentially, where each new model corrects the residual errors of the previous one.
    Uses Gradient Descent to minimize the loss function. Performs well on complex non-linear patterns.

    **Linear Regression**
    Models the relationship between features and target as a linear function.
    Acts as a simple, low-variance base estimator that stabilizes the overall ensemble.
    """)

    st.subheader("3. Model Development Process")
    st.write("""
    1. Extracted 4 features from each FEN string:
       - **Material Balance** — Difference in total piece value between White and Black (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9).
       - **Net Mobility** — White's legal move count minus Black's legal move count. Positive means White has more options.
       - **Center Control** — Net difference in center squares (d4, e4, d5, e5) attacked by each side.
       - **King Safety** — Whether each king is under attack, expressed as a net difference from White's perspective.
    2. Split data into training and test sets using an 80:20 ratio via `train_test_split`.
    3. Constructed a `VotingRegressor` combining Random Forest, Gradient Boosting, and Linear Regression.
    4. Trained the model on the training set (X_train, y_train).
    5. Evaluated performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
    6. Saved the trained model using `joblib`.
    """)

    st.subheader("4. References")
    st.write("""
    - Kaggle Chess Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
    - Scikit-learn VotingRegressor: https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor
    - Python-chess Library: https://python-chess.readthedocs.io/
    - Stockfish Chess Engine: https://stockfishchess.org/
    """)

# ── PAGE 2: Neural Network description ────────
elif page == "Neural Network":
    st.header("Neural Network Model")

    st.subheader("1. Data Preparation")
    st.write("""
    The same dataset and preparation pipeline used for the ML Model was applied here:
    - Merged chessData.csv and chess_games.csv.
    - Removed null values and mate scores.
    - Converted evaluation from centipawns to pawns and clamped to [-10, +10].
    - Extracted 4 features: Material Balance, Net Mobility, Center Control, King Safety.
    - Applied an 80:20 train/test split.
    """)

    st.subheader("2. Algorithm Theory")
    st.write("""
    The model is a **Multilayer Perceptron (MLP)**, a feedforward Neural Network capable of learning
    non-linear relationships between input features and the target evaluation score.

    **Model Architecture:**
    - Input Layer: 4 neurons (Material Balance, Net Mobility, Center Control, King Safety)
    - Hidden Layer 1: 64 neurons with ReLU activation
    - Hidden Layer 2: 32 neurons with ReLU activation
    - Output Layer: 1 neuron (predicted evaluation score)

    **ReLU Activation Function:**
    Defined as f(x) = max(0, x). ReLU introduces non-linearity into the network, enabling it to learn
    complex patterns. It also mitigates the vanishing gradient problem commonly seen with Sigmoid and Tanh.

    **Loss Function:**
    Mean Squared Error (MSE) measures the average squared difference between predicted and actual scores.
    Minimizing MSE during training drives the model toward more accurate predictions.
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

# ── PAGE 3: Test ML ───────────────────────────
elif page == "Test ML":
    st.header("Test — Ensemble Model")
    render_test_page(ensemble, "Ensemble (ML)", "ml")

# ── PAGE 4: Test NN ───────────────────────────
elif page == "Test NN":
    st.header("Test — Neural Network")
    render_test_page(nn_model, "Neural Network", "nn")
