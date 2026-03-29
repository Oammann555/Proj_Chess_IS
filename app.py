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


def get_advantage_label(score):
    if score > 2.0:
        return "White has a decisive advantage."
    elif score > 0.5:
        return "White has a clear advantage."
    elif score > 0.1:
        return "White has a slight advantage."
    elif score < -2.0:
        return "Black has a decisive advantage."
    elif score < -0.5:
        return "Black has a clear advantage."
    elif score < -0.1:
        return "Black has a slight advantage."
    else:
        return "The position is roughly equal."


PRESET_FENS = {
    "Starting Position": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "Sicilian Defense (after 1.e4 c5)": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "Ruy Lopez (after 1.e4 e5 2.Nf3 Nc6 3.Bb5)": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "Queen's Gambit (after 1.d4 d5 2.c4)": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "King's Indian Defense": "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "Custom FEN": ""
}

MODEL_LIMITATION_NOTE = """
**Note on Model Accuracy**

This model evaluates positions using only 4 hand-crafted features:
Material Balance, Mobility, Center Control, and King Safety.

Because of this, the model has known limitations:
- It cannot detect tactical threats such as forks, pins, skewers, or checkmate threats.
- It cannot evaluate pawn structure, piece coordination, or long-term strategic advantages.
- Positions that appear roughly equal by material but contain decisive tactics (e.g. Fool's Mate setup)
  will not be accurately evaluated.

For example, in the Fool's Mate position (before Black delivers mate), material is equal and the model
may score it near 0 or even favour White — which does not reflect the true tactical danger.

This is a fundamental limitation of feature-based models. A stronger evaluation would require
full board encoding (e.g. 64-square bitboard representation) or integration with an engine like Stockfish.
"""


def render_board_section(model, model_name, key_prefix):
    """Shared interactive board UI for a given model."""

    st.write(f"""
    Set up any chess position using FEN notation and evaluate it using the {model_name}.
    Select a preset opening or enter a custom FEN string below.
    """)

    with st.expander("Model Limitation — Read Before Interpreting Results", expanded=False):
        st.markdown(MODEL_LIMITATION_NOTE)

    st.subheader("Position Setup")

    preset = st.selectbox(
        "Load a preset position",
        list(PRESET_FENS.keys()),
        key=f"{key_prefix}_preset"
    )

    if preset == "Custom FEN":
        fen_input = st.text_input(
            "Enter FEN string",
            placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            key=f"{key_prefix}_fen_input"
        )
    else:
        fen_input = PRESET_FENS[preset]
        st.text_input("Current FEN", value=fen_input, disabled=True, key=f"{key_prefix}_fen_display")

    side = st.radio(
        "Side to move",
        ["White", "Black"],
        horizontal=True,
        key=f"{key_prefix}_side"
    )

    def set_side(fen, side_str):
        parts = fen.strip().split()
        if len(parts) >= 2:
            parts[1] = 'w' if side_str == "White" else 'b'
        return ' '.join(parts)

    if fen_input:
        try:
            fen_to_use = set_side(fen_input, side)
            board = chess.Board(fen_to_use)

            svg = chess.svg.board(board, size=380)
            st.write("**Current Position**")
            st.components.v1.html(svg, height=400)

            col1, col2, col3 = st.columns(3)
            col1.metric("Legal Moves", board.legal_moves.count())
            col2.metric("In Check", "Yes" if board.is_check() else "No")
            col3.metric("Move Number", board.fullmove_number)

            if board.is_checkmate():
                winner = "Black" if board.turn == chess.WHITE else "White"
                st.error(f"Checkmate — {winner} wins. Model evaluation is not meaningful for terminal positions.")
            elif board.is_stalemate():
                st.warning("Stalemate — the game is drawn.")
            else:
                st.subheader(f"Evaluation — {model_name}")
                if st.button("Evaluate Position", key=f"{key_prefix}_eval_btn"):
                    features = extract_features(fen_to_use)
                    score = model.predict(features)[0]
                    st.success(f"Evaluation Score: {score:.2f}")
                    st.write(get_advantage_label(score))

                with st.expander("Show Feature Breakdown"):
                    features = extract_features(fen_to_use)
                    f = features[0]
                    st.write(f"**Material Balance:** {f[0]}  (positive = White has more material)")
                    st.write(f"**Mobility:** {f[1]}  (legal moves available to the current player)")
                    st.write(f"**Center Control:** {f[2]}  (positive = White attacks more center squares)")
                    st.write(f"**King Safety:** {f[3]}  (negative = White king is under more pressure)")

        except Exception as e:
            st.error(f"Invalid FEN string: {e}")
    else:
        st.info("Select a preset or enter a FEN string to set up a position.")


# ---------------- UI ----------------
st.title("Chess AI Evaluation System")

page = st.sidebar.selectbox(
    "Select Page",
    ["ML Model", "Neural Network", "Test ML", "Test NN", "Board (ML)", "Board (NN)"]
)

# ---------------- PAGE 1: ML Model ----------------
if page == "ML Model":
    st.header("Machine Learning Model (Ensemble)")

    st.subheader("1. Data Preparation")
    st.write("""
    This project uses two chess datasets:

    - **chessData.csv** — Chess positions in FEN format, each paired with an evaluation score computed by the Stockfish engine.
    - **chess_games.csv** — Historical chess game records downloaded from Kaggle.

    Preparation steps:
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

    Three base models used in the Voting Regressor:

    **Random Forest Regressor**
    Builds multiple Decision Trees using Bootstrap Sampling (random subsets of training data).
    Each tree independently predicts a score, and the final prediction is the mean across all trees.
    Effective at reducing overfitting due to the averaging of many uncorrelated trees.

    **Gradient Boosting Regressor**
    Builds models sequentially, where each new model focuses on correcting the residual errors of the previous one.
    Uses Gradient Descent to minimize the loss function at each step.
    Performs well on structured data with complex non-linear patterns.

    **Linear Regression**
    Models the relationship between features and target as a linear function.
    Acts as a simple, low-variance base estimator that stabilizes the overall ensemble predictions.
    """)

    st.subheader("3. Model Development Process")
    st.write("""
    1. Extracted 4 features from each FEN string:
       - **Material Balance** — Difference in total piece value between White and Black (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9).
       - **Mobility** — Number of legal moves available to the current player.
       - **Center Control** — Number of center squares (d4, e4, d5, e5) attacked by each side, expressed as a net difference.
       - **King Safety** — Whether each king is currently under attack (1 if attacked, 0 if not), expressed as a net difference.
    2. Split data into training and test sets using an 80:20 ratio via train_test_split.
    3. Constructed a VotingRegressor combining Random Forest, Gradient Boosting, and Linear Regression.
    4. Trained the model on the training set (X_train, y_train).
    5. Evaluated performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE) on the test set.
    6. Saved the trained model using joblib.
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

    Model Architecture:
    - Input Layer: 4 neurons (Material Balance, Mobility, Center Control, King Safety)
    - Hidden Layer 1: 64 neurons with ReLU activation
    - Hidden Layer 2: 32 neurons with ReLU activation
    - Output Layer: 1 neuron (predicted evaluation score)

    **ReLU Activation Function**
    Defined as f(x) = max(0, x). ReLU introduces non-linearity into the network, enabling it to learn
    complex patterns. It also mitigates the vanishing gradient problem commonly associated with Sigmoid
    and Tanh activations, which makes training deeper networks more stable.

    **Loss Function**
    Mean Squared Error (MSE) is used to measure the average squared difference between predicted and
    actual evaluation scores. Minimizing MSE during training drives the model to produce more accurate predictions.
    """)

    st.subheader("3. Model Development Process")
    st.write("""
    1. Used the same 4 features and target variable as the ML Model.
    2. Built an MLPRegressor using Scikit-learn with the following configuration:
       - hidden_layer_sizes = (64, 32)
       - activation = 'relu'
       - max_iter = 500
    3. Trained the model on X_train and y_train.
    4. Evaluated performance using MAE and MSE on X_test.
    5. Compared results against the Ensemble Model to assess relative accuracy.
    6. Saved the trained model using joblib.
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
            st.write(get_advantage_label(pred))
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
            st.write(get_advantage_label(pred_nn))
        except Exception as e:
            st.error(f"Invalid FEN or prediction error: {e}")

# ---------------- PAGE 5: Board (ML) ----------------
elif page == "Board (ML)":
    st.header("Interactive Board — Ensemble Model")
    render_board_section(ensemble, "Ensemble (ML)", key_prefix="ml")

# ---------------- PAGE 6: Board (NN) ----------------
elif page == "Board (NN)":
    st.header("Interactive Board — Neural Network")
    render_board_section(nn_model, "Neural Network", key_prefix="nn")
