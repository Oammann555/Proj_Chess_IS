import streamlit as st
import joblib
import numpy as np
import chess
import chess.svg

# ── Load models ───────────────────────────────────────────────
ensemble = joblib.load("ensemble_model.pkl")
nn_model  = joblib.load("nn_model_sklearn.pkl")

piece_values = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9,
}


# ── Feature extraction ────────────────────────────────────────
def extract_features(fen):
    board = chess.Board(fen)

    # 1. Material balance (White − Black)
    material = 0
    for pt in piece_values:
        material += len(board.pieces(pt, chess.WHITE)) * piece_values[pt]
        material -= len(board.pieces(pt, chess.BLACK)) * piece_values[pt]

    # 2. Net mobility (White moves − Black moves)
    current_moves = board.legal_moves.count()
    try:
        board.push(chess.Move.null())
        opponent_moves = board.legal_moves.count()
        board.pop()
    except Exception:
        opponent_moves = 0
    mobility = (current_moves - opponent_moves) if board.turn == chess.WHITE \
               else (opponent_moves - current_moves)

    # 3. Center control (d4, e4, d5, e5)
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    center = (sum(board.is_attacked_by(chess.WHITE, sq) for sq in center_squares) -
              sum(board.is_attacked_by(chess.BLACK, sq) for sq in center_squares))

    # 4. King safety
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    king_safety = (int(board.is_attacked_by(chess.BLACK, wk)) -
                   int(board.is_attacked_by(chess.WHITE, bk)))

    return np.array([[material, mobility, center, king_safety]])


def advantage_label(score):
    if   score >  2.0: return "White has a decisive advantage."
    elif score >  0.5: return "White has a clear advantage."
    elif score >  0.1: return "White has a slight advantage."
    elif score < -2.0: return "Black has a decisive advantage."
    elif score < -0.5: return "Black has a clear advantage."
    elif score < -0.1: return "Black has a slight advantage."
    else:              return "The position is roughly equal."


# ── Preset positions ──────────────────────────────────────────
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


# ── Interactive board HTML ────────────────────────────────────
def make_board_html(init_pos, init_side):
    active_w = "active" if init_side == "w" else ""
    active_b = "active" if init_side == "b" else ""
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
  .main-row {{
    display: flex;
    gap: 12px;
    align-items: flex-start;
    justify-content: center;
  }}
  #board {{ width: 380px; flex-shrink: 0; }}
  .tray {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px;
    width: 52px;
  }}
  .tray-label {{
    font-size: 10px;
    font-weight: 700;
    text-align: center;
    color: #64748b;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2px;
  }}
  .spare-piece {{
    width: 40px;
    height: 40px;
    cursor: grab;
    user-select: none;
    border-radius: 4px;
    transition: background 0.15s;
  }}
  .spare-piece:hover {{ background: #e2e8f0; }}
  .spare-piece img {{ width: 100%; height: 100%; pointer-events: none; }}
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
  .btn-gray {{ background: #374151; color: #f3f4f6; }}
  .btn-blue {{ background: #2563eb; color: #ffffff; }}
  .btn-red  {{ background: #dc2626; color: #ffffff; }}
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
  .side-btn.active {{ background: #1d4ed8; border-color: #1d4ed8; color: #fff; }}
  #fen-box {{
    width: 100%;
    max-width: 480px;
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
  #copy-msg {{ font-size: 12px; color: #16a34a; min-height: 16px; text-align: center; }}
  .label {{ font-size: 12px; color: #6b7280; }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
</head>
<body>

<div class="main-row">
  <div class="tray">
    <div class="tray-label">White</div>
    <div class="spare-piece" draggable="true" data-piece="wK"><img src="https://lichess1.org/assets/piece/cburnett/wK.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="wQ"><img src="https://lichess1.org/assets/piece/cburnett/wQ.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="wR"><img src="https://lichess1.org/assets/piece/cburnett/wR.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="wB"><img src="https://lichess1.org/assets/piece/cburnett/wB.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="wN"><img src="https://lichess1.org/assets/piece/cburnett/wN.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="wP"><img src="https://lichess1.org/assets/piece/cburnett/wP.svg"></div>
  </div>

  <div id="board"></div>

  <div class="tray">
    <div class="tray-label">Black</div>
    <div class="spare-piece" draggable="true" data-piece="bK"><img src="https://lichess1.org/assets/piece/cburnett/bK.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="bQ"><img src="https://lichess1.org/assets/piece/cburnett/bQ.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="bR"><img src="https://lichess1.org/assets/piece/cburnett/bR.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="bB"><img src="https://lichess1.org/assets/piece/cburnett/bB.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="bN"><img src="https://lichess1.org/assets/piece/cburnett/bN.svg"></div>
    <div class="spare-piece" draggable="true" data-piece="bP"><img src="https://lichess1.org/assets/piece/cburnett/bP.svg"></div>
  </div>
</div>

<div class="row">
  <span class="label">Side to move:</span>
  <button class="side-btn {active_w}" id="btn-w" onclick="setSide('w')">White</button>
  <button class="side-btn {active_b}" id="btn-b" onclick="setSide('b')">Black</button>
</div>

<div id="fen-box">Loading...</div>

<div class="row">
  <button class="btn-gray" onclick="resetBoard()">Reset</button>
  <button class="btn-gray" onclick="board.flip()">Flip</button>
  <button class="btn-red"  onclick="clearBoard()">Clear</button>
  <button class="btn-blue" onclick="copyFen()">Copy FEN</button>
</div>
<div id="copy-msg"></div>

<script>
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

  var side = '{init_side}';

  function getFen() {{ return board.fen() + ' ' + side + ' KQkq - 0 1'; }}
  function updateFenBox() {{ document.getElementById('fen-box').textContent = getFen(); }}
  function setSide(s) {{
    side = s;
    document.getElementById('btn-w').classList.toggle('active', s === 'w');
    document.getElementById('btn-b').classList.toggle('active', s === 'b');
    updateFenBox();
  }}
  function resetBoard() {{ board.position('{init_pos}', false); setSide('{init_side}'); }}
  function clearBoard() {{ board.position('8/8/8/8/8/8/8/8', false); updateFenBox(); }}
  function copyFen() {{
    navigator.clipboard.writeText(getFen()).then(function() {{
      var el = document.getElementById('copy-msg');
      el.textContent = 'Copied! Paste it in the field below.';
      setTimeout(function() {{ el.textContent = ''; }}, 2500);
    }});
  }}

  var board = Chessboard('board', {{
    draggable: true,
    position: '{init_pos}',
    onSnapEnd: updateFenBox,
    dropOffBoard: 'trash',
    pieceTheme: function(p) {{ return pieceMap[p]; }}
  }});
  updateFenBox();

  var draggedPiece = null;
  document.querySelectorAll('.spare-piece').forEach(function(el) {{
    el.addEventListener('dragstart', function(e) {{
      draggedPiece = el.getAttribute('data-piece');
      e.dataTransfer.effectAllowed = 'copy';
    }});
  }});
  document.getElementById('board').addEventListener('dragover', function(e) {{
    e.preventDefault(); e.dataTransfer.dropEffect = 'copy';
  }});
  document.getElementById('board').addEventListener('drop', function(e) {{
    e.preventDefault();
    if (!draggedPiece) return;
    var rect = document.getElementById('board').getBoundingClientRect();
    var sq   = rect.width / 8;
    var col  = Math.floor((e.clientX - rect.left) / sq);
    var row  = Math.floor((e.clientY - rect.top)  / sq);
    var file = String.fromCharCode('a'.charCodeAt(0) + col);
    var rank = 8 - row;
    var pos  = board.position();
    pos[file + rank] = draggedPiece;
    board.position(pos, false);
    updateFenBox();
    draggedPiece = null;
  }});
</script>
</body>
</html>"""


# ── Test page (shared for ML and NN) ─────────────────────────
def render_test_page(model, model_name, key_prefix):
    preset = st.selectbox("Load a preset position",
                          list(PRESET_FENS.keys()), key=f"{key_prefix}_preset")
    init_fen  = PRESET_FENS[preset]
    init_pos  = init_fen.split()[0]
    init_side = init_fen.split()[1]

    st.write("Drag pieces to set up your position, then click **Copy FEN** and paste it below.")
    st.components.v1.html(make_board_html(init_pos, init_side), height=590, scrolling=False)
    st.divider()

    fen_input = st.text_input("Paste FEN here",
                              placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                              key=f"{key_prefix}_fen")
    if not fen_input.strip():
        st.info("Copy the FEN from the board above and paste it here to evaluate.")
        return

    try:
        board_obj = chess.Board(fen_input.strip())
    except Exception as e:
        st.error(f"Invalid FEN: {e}")
        return

    col_svg, col_info = st.columns([1, 1])
    with col_svg:
        st.write("**Position Preview**")
        st.components.v1.html(chess.svg.board(board_obj, size=280), height=295)
    with col_info:
        st.write("**Position Info**")
        st.metric("Legal Moves", board_obj.legal_moves.count())
        st.metric("In Check",    "Yes" if board_obj.is_check() else "No")
        st.metric("Full Move",   board_obj.fullmove_number)
        f = extract_features(fen_input.strip())[0]
        st.write(f"Material Balance: **{f[0]:+.0f}**")
        st.write(f"Net Mobility: **{f[1]:+.0f}**")
        st.write(f"Center Control: **{f[2]:+.0f}**")
        st.write(f"King Safety: **{f[3]:+.0f}**")

    if board_obj.is_checkmate():
        winner = "Black" if board_obj.turn == chess.WHITE else "White"
        st.error(f"Checkmate — {winner} wins. Evaluation is not applicable.")
        return
    if board_obj.is_stalemate():
        st.warning("Stalemate — the game is drawn.")
        return

    st.divider()
    if st.button(f"Evaluate with {model_name}", key=f"{key_prefix}_btn", use_container_width=True):
        score = model.predict(extract_features(fen_input.strip()))[0]
        st.session_state[f"{key_prefix}_score"] = score

    if f"{key_prefix}_score" in st.session_state:
        score = st.session_state[f"{key_prefix}_score"]
        st.metric(f"{model_name} Evaluation", f"{score:+.2f} pawns")
        fn = st.success if score > 0 else (st.error if score < 0 else st.info)
        fn(advantage_label(score))


# ═════════════════════════════════════════════
#  MAIN UI
# ═════════════════════════════════════════════
st.title("Chess AI Evaluation System")

page = st.sidebar.selectbox("Select Page",
    ["ML Model", "Neural Network", "Test ML", "Test NN"])


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — ML Model
# ══════════════════════════════════════════════════════════════
if page == "ML Model":
    st.header("Machine Learning Model — Ensemble (Voting Regressor)")

    # ── 1. Data Preparation ───────────────────────────────────
    st.subheader("1. Data Preparation")
    st.write("""
    **Datasets used:**
    - **chessData.csv** — Chess positions in FEN notation, each paired with a numeric evaluation
      score (in centipawns) computed by the Stockfish engine. Stockfish is a top-tier open-source
      chess engine that calculates evaluations to very high depth.
    - **chess_games.csv** — Historical game records from Kaggle containing move sequences and
      metadata such as player ratings and opening names.

    **Why two datasets?**
    Combining datasets increases the variety of positions the model is trained on, reducing the
    risk of overfitting to a narrow distribution of positions.

    **Preparation steps:**
    1. Merged both datasets and retained only the `fen` and `eval` columns.
    2. Removed rows with null values.
    3. Removed positions where the evaluation was a mate score (denoted `#N`), because mate scores
       are categorical (forced win/loss) rather than numeric, and cannot be used for regression.
    4. Converted evaluation from centipawns to pawns by dividing by 100. This puts scores on
       a human-readable scale where 1.0 = one pawn advantage.
    5. Clamped scores to the range [-10, +10] to limit the influence of extreme outlier positions
       (e.g. positions where one side has a +50 material advantage).
    6. Randomly sampled 2,000 rows (`random_state=42`) to reduce training time while maintaining
       a representative distribution of positions.
    """)

    # ── 2. Feature Engineering ────────────────────────────────
    st.subheader("2. Feature Engineering")
    st.write("""
    Four features are extracted from each FEN string. All features are expressed from
    **White's perspective** — positive values favour White, negative values favour Black.

    **Feature 1 — Material Balance**
    The sum of White's piece values minus the sum of Black's piece values, using standard weights:
    Pawn = 1, Knight = 3, Bishop = 3, Rook = 5, Queen = 9.
    Material advantage is the single strongest predictor of evaluation in chess.

    **Feature 2 — Net Mobility**
    White's number of legal moves minus Black's number of legal moves.
    A higher value means White has more available options, indicating greater piece activity
    and control. The opponent's move count is estimated using a null-move trick — temporarily
    passing the turn to count the opponent's responses without actually making a move.

    **Feature 3 — Center Control**
    The number of center squares (d4, e4, d5, e5) attacked by White minus those attacked by Black.
    Control of the center is a fundamental strategic principle, as centralized pieces have greater
    influence over the board.

    **Feature 4 — King Safety**
    Returns +1 if White's king is currently under attack (in check), 0 otherwise — minus the same
    value for Black's king. A negative score indicates White's king is under more immediate pressure.
    """)

    # ── 3. Algorithm Theory ───────────────────────────────────
    st.subheader("3. Algorithm Theory")
    st.write("""
    The model is a **Voting Regressor** — a meta-estimator that fits multiple base regressors
    and averages their predictions. Because the three base models have different inductive biases
    (linear, tree-based additive, tree-based averaging), their errors are partially uncorrelated,
    and averaging reduces the overall variance compared to any single model.

    ---

    **Random Forest Regressor** (`n_estimators=100, random_state=42`)

    Builds 100 Decision Trees independently, each on a bootstrap sample (random draw with replacement)
    of the training data. At every node split, only a random subset of features is considered —
    a technique called feature bagging. This decorrelates the trees and significantly reduces
    variance compared to a single deep Decision Tree.

    Final prediction = average of all 100 tree predictions.

    ---

    **Gradient Boosting Regressor** (`n_estimators=100, learning_rate=0.1, random_state=42`)

    Builds trees sequentially. The first tree fits the raw targets. Each subsequent tree fits
    the **residual errors** (the gradient of the loss with respect to the current prediction),
    progressively correcting the ensemble. The learning rate (0.1) scales each tree's contribution,
    preventing any single tree from dominating and reducing overfitting.

    Gradient Boosting typically achieves lower bias than Random Forest on structured data, at the
    cost of being more sensitive to hyperparameters.

    ---

    **Linear Regression**

    Fits a linear relationship: `score = w1·material + w2·mobility + w3·center + w4·king_safety + b`.
    Although simple, Linear Regression is unbiased when the true relationship is approximately linear
    (e.g. material advantage is nearly linear in evaluation). Including it in the ensemble adds a
    stable, low-variance component that anchors predictions in regions where the tree-based models
    may overfit.
    """)

    # ── 4. Model Development Process ─────────────────────────
    st.subheader("4. Model Development Process")
    st.write("""
    1. Extracted the 4 features above from each FEN string in the dataset.
    2. Split data 80 / 20 into training and test sets using `train_test_split(random_state=42)`.
    3. Constructed a `VotingRegressor` with `estimators=[('rf', rf), ('gb', gb), ('lr', lr)]`.
    4. Called `ensemble.fit(X_train, y_train)` — each base model is trained independently.
    5. Evaluated on X_test using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
    6. Saved the trained model to `ensemble_model.pkl` via `joblib.dump()`.
    """)

    # ── 5. Model Performance ──────────────────────────────────
    st.subheader("5. Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("MAE — Ensemble", "1.9981 pawns")
    c2.metric("MSE — Ensemble", "8.7472")
    st.caption(
        "MAE (Mean Absolute Error) = average prediction error in pawn units. "
        "MSE (Mean Squared Error) = average squared error, penalising large mistakes more heavily. "
        "Both metrics are on the test set (unseen during training). Lower is better."
    )

    # ── 6. Known Limitations ─────────────────────────────────
    st.subheader("6. Known Limitations")
    st.warning("""
    This model uses only 4 hand-crafted features and has the following inherent limitations:

    - **No tactical awareness.** The model cannot detect checkmate threats, forks, pins, skewers,
      or discovered attacks. A position that is equal by material may contain a forced mate in 1.
    - **No pawn structure analysis.** Isolated pawns, doubled pawns, and passed pawns are not
      represented in any feature.
    - **No piece coordination.** Whether pieces actively support each other or are passively placed
      is invisible to the model.
    - **No game phase awareness.** The same features are used in the opening, middlegame, and
      endgame, even though their relative importance changes significantly across phases.

    These are fundamental constraints of feature-based models. A stronger evaluation function
    would use full board encoding (e.g. 768-dimensional bitboard representation covering all
    12 piece types × 64 squares) trained on millions of engine-annotated positions.
    """)

    # ── 7. References ─────────────────────────────────────────
    st.subheader("7. References")
    st.write("""
    - Kaggle Chess Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
    - Scikit-learn VotingRegressor: https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor
    - Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
    - Scikit-learn Gradient Boosting: https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting
    - Python-chess Library: https://python-chess.readthedocs.io/
    - Stockfish Chess Engine: https://stockfishchess.org/
    """)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — Neural Network
# ══════════════════════════════════════════════════════════════
elif page == "Neural Network":
    st.header("Neural Network Model — Multilayer Perceptron (MLP)")

    # ── 1. Data Preparation ───────────────────────────────────
    st.subheader("1. Data Preparation")
    st.write("""
    The Neural Network model uses the same dataset and preparation pipeline as the Ensemble model:

    - Merged **chessData.csv** and **chess_games.csv**, retaining only `fen` and `eval` columns.
    - Removed null values and positions with mate scores (`#N`) which are non-numeric.
    - Converted evaluation from centipawns to pawns (÷ 100) and clamped to [-10, +10].
    - Extracted the same 4 features: Material Balance, Net Mobility, Center Control, King Safety.
    - Randomly sampled 2,000 rows (`random_state=42`) and applied an 80 / 20 train/test split.

    Sharing the same data pipeline ensures a fair performance comparison between the two models,
    as differences in results reflect model architecture rather than data differences.
    """)

    # ── 2. Algorithm Theory ───────────────────────────────────
    st.subheader("2. Algorithm Theory")
    st.write("""
    The model is a **Multilayer Perceptron (MLP)** — a fully connected feedforward Neural Network.
    Unlike the Ensemble model which combines separate linear and tree-based learners, the MLP learns
    a single unified function that can approximate complex non-linear mappings between inputs and outputs.

    ---

    **Model Architecture**

    ```
    Input (4)  →  Hidden Layer 1 (64, ReLU)  →  Hidden Layer 2 (32, ReLU)  →  Output (1)
    ```

    - **Input layer:** 4 neurons, one per feature.
    - **Hidden Layer 1 — 64 neurons, ReLU activation.**
      Learns first-order interactions between features (e.g. high material + high mobility = stronger advantage).
    - **Hidden Layer 2 — 32 neurons, ReLU activation.**
      Learns higher-order combinations from the representations produced by Layer 1.
      The smaller size (32 < 64) compresses the representation, acting as a bottleneck that
      forces the network to retain only the most informative patterns.
    - **Output layer — 1 neuron, no activation.**
      Produces a single continuous value: the predicted evaluation score in pawns.

    The architecture is intentionally small. A larger network would risk overfitting on only 2,000
    training samples — the model would memorise positions rather than generalise.

    ---

    **ReLU Activation — f(x) = max(0, x)**

    Applied to both hidden layers. ReLU introduces non-linearity, which is essential for the network
    to learn anything beyond a linear relationship. Without activation functions, stacking multiple
    layers would collapse into a single linear transformation.

    Compared to Sigmoid (σ) and Tanh, ReLU avoids the **vanishing gradient problem**: for large
    positive inputs, the gradient of ReLU is always 1, whereas the gradient of Sigmoid and Tanh
    approaches 0 — making training slow or stalling entirely in deeper networks.

    ---

    **Optimizer — Adam** (Adaptive Moment Estimation, default in Scikit-learn MLPRegressor)

    Adam maintains per-parameter adaptive learning rates using running estimates of the first moment
    (mean of gradients) and second moment (uncentered variance of gradients). This makes it
    significantly faster to converge than vanilla Stochastic Gradient Descent (SGD) and robust
    to sparse or noisy gradients.

    ---

    **Loss Function — Mean Squared Error (MSE)**

    `MSE = (1/n) × Σ (y_true − y_pred)²`

    MSE penalises large errors more heavily than small ones due to the squaring operation. This is
    appropriate here: a prediction of +3.0 when the true score is -3.0 is far worse than being
    off by 0.1, and MSE reflects that asymmetry. The network minimises MSE via backpropagation —
    computing the gradient of the loss with respect to every weight, then updating weights in the
    direction that reduces the loss.
    """)

    # ── 3. Model Development Process ─────────────────────────
    st.subheader("3. Model Development Process")
    st.write("""
    1. Used the same 4 features and target variable (`eval` in pawns) as the Ensemble model.
    2. Constructed an `MLPRegressor` with the following configuration:
       - `hidden_layer_sizes = (64, 32)` — two hidden layers of 64 and 32 neurons
       - `activation = 'relu'` — ReLU applied to all hidden neurons
       - `solver = 'adam'` — Adam optimizer (Scikit-learn default)
       - `max_iter = 500` — maximum number of training epochs
       - `random_state = 42` — for reproducibility
    3. Trained on X_train / y_train. Scikit-learn handles the forward pass, MSE loss computation,
       backpropagation, and Adam weight updates automatically each epoch.
    4. Evaluated on X_test using MAE and MSE.
    5. Compared results against the Ensemble model. The MLP can capture non-linear feature
       interactions that Linear Regression inside the ensemble cannot, but may underperform
       if the dataset is too small for the network to generalise.
    6. Saved the trained model to `nn_model_sklearn.pkl` via `joblib.dump()`.
    """)

    # ── 4. Model Performance ──────────────────────────────────
    st.subheader("4. Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("MAE — Neural Network", "1.9875 pawns")
    c2.metric("MSE — Neural Network", "8.5424")
    st.caption(
        "MAE (Mean Absolute Error) = average prediction error in pawn units. "
        "MSE (Mean Squared Error) = average squared error. "
        "Both measured on the held-out test set. Lower is better."
    )

    # ── 5. Known Limitations ─────────────────────────────────
    st.subheader("5. Known Limitations")
    st.warning("""
    The MLP shares the same feature-level limitations as the Ensemble model, plus additional
    constraints specific to neural networks trained on small datasets:

    - **No tactical awareness.** Cannot detect checkmate threats, forks, pins, or discovered attacks.
    - **No pawn structure or piece coordination.** These are not captured by any of the 4 features.
    - **Small dataset constraint.** Neural networks typically require far more data than tree-based
      models to generalise well. With only 2,000 training samples, the MLP may not have learned
      a robust representation of chess evaluation.
    - **Feature ceiling.** Regardless of network depth or width, the model is bounded by the
      information contained in its 4 input features. No architecture can recover information
      that was never provided.

    A production-strength chess neural network (such as those used in AlphaZero or Leela Chess Zero)
    uses full board representation and is trained on tens of millions of self-play games.
    """)

    # ── 6. References ─────────────────────────────────────────
    st.subheader("6. References")
    st.write("""
    - Scikit-learn MLPRegressor: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    - Adam Optimizer (Kingma & Ba, 2014): https://arxiv.org/abs/1412.6980
    - Kaggle Chess Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
    - Python-chess Library: https://python-chess.readthedocs.io/
    - ReLU Activation: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    - AlphaZero (DeepMind): https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go
    """)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — Test ML
# ══════════════════════════════════════════════════════════════
elif page == "Test ML":
    st.header("Test — Ensemble Model")
    render_test_page(ensemble, "Ensemble (ML)", "ml")


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — Test NN
# ══════════════════════════════════════════════════════════════
elif page == "Test NN":
    st.header("Test — Neural Network")
    render_test_page(nn_model, "Neural Network", "nn")
