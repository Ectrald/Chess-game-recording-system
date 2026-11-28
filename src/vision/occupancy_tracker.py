"""3D column-projection occupancy and board-transition scoring for skewed overhead cameras.

Addresses two physical failure modes of per-square 2D classifiers and naive temporal diffs:
1. Same-color piece camouflage (e.g., white piece on light square, dark piece on dark square):
   a piece standing at (r, c) exhibits interior edges/texture AND projects upward in image
   space into the bottom band of (r - 1, c), which ALWAYS has opposite checkerboard parity.
2. Vertical parallax spillover: tall pieces standing at (r + 1, c) lean into the bottom half
   of (r, c), creating large frame-to-frame deltas on empty square (r, c) while its top band
   remains empty and (r + 1, c) has a physically standing piece.

The model self-calibrates empty-square appearance (including a 2D illumination plane per
light/dark parity) from known-empty squares on the current authoritative board.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import chess
import cv2
import numpy as np

from src.core.datatypes import BoardOrientation, CHESS_SQUARES
from src.reasoning.candidate_generator import LegalMoveCandidateGenerator
from src.vision.roi_extractor import CellRoiExtractor


@dataclass(frozen=True)
class SquareBands:
    """Extracted gamma-0.5 appearance statistics across core, top, and bottom regions of one square."""

    square: str
    row: int
    col: int
    is_light: bool
    core_mean: float
    core_std: float
    core_grad: float
    top_std: float
    top_grad: float
    bot_std: float
    bot_grad: float


@dataclass(frozen=True)
class SquareOccupancy:
    """Normalized [0, 1] occupancy probabilities split by vertical sub-region."""

    top: float
    core: float
    bot: float

    @property
    def combined(self) -> float:
        return max(self.top, self.core * 0.85)


@dataclass
class _ParityBaseline:
    core_std: float
    top_std: float
    bot_std: float
    core_grad: float
    top_grad: float
    bot_grad: float
    plane_coeffs: np.ndarray  # (3,) -> [a, b, c] for L_empty(r, c) = a + b*r + c*c


class EmptySquareModel:
    """Self-calibrating appearance model for empty light and dark squares."""

    def __init__(self) -> None:
        self._baselines: Dict[bool, _ParityBaseline] = {}

    @property
    def is_calibrated(self) -> bool:
        return True in self._baselines and False in self._baselines

    def fit(self, bands: Dict[str, SquareBands], known_empty: Sequence[str], ema_alpha: float = 1.0) -> None:
        """Fits or smoothly updates empty-square light/dark statistics and 2D illumination planes."""
        for parity in (True, False):
            samples = [bands[sq] for sq in known_empty if sq in bands and bands[sq].is_light == parity]
            if not samples:
                continue

            # Trim extreme high-texture outliers (e.g. tall king leaning deeply into an empty neighbor)
            if len(samples) >= 5:
                med_tg = float(np.median([s.top_grad for s in samples]))
                med_ts = float(np.median([s.top_std for s in samples]))
                clean = [
                    s for s in samples
                    if s.top_grad <= max(med_tg * 2.2, 0.05) and s.top_std <= max(med_ts * 2.2, 12.0)
                ]
                if len(clean) >= 3:
                    samples = clean

            c_std = float(np.median([s.core_std for s in samples]))
            t_std = float(np.median([s.top_std for s in samples]))
            b_std = float(np.median([s.bot_std for s in samples]))
            c_grad = float(np.median([s.core_grad for s in samples]))
            t_grad = float(np.median([s.top_grad for s in samples]))
            b_grad = float(np.median([s.bot_grad for s in samples]))

            if len(samples) >= 6:
                design = np.array([[1.0, float(s.row), float(s.col)] for s in samples], dtype=np.float64)
                targets = np.array([s.core_mean for s in samples], dtype=np.float64)
                coeffs, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
            else:
                coeffs = np.array([float(np.median([s.core_mean for s in samples])), 0.0, 0.0], dtype=np.float64)

            if parity in self._baselines and ema_alpha < 1.0:
                prev = self._baselines[parity]
                k = float(max(0.01, min(1.0, ema_alpha)))
                self._baselines[parity] = _ParityBaseline(
                    core_std=(1.0 - k) * prev.core_std + k * c_std,
                    top_std=(1.0 - k) * prev.top_std + k * t_std,
                    bot_std=(1.0 - k) * prev.bot_std + k * b_std,
                    core_grad=(1.0 - k) * prev.core_grad + k * c_grad,
                    top_grad=(1.0 - k) * prev.top_grad + k * t_grad,
                    bot_grad=(1.0 - k) * prev.bot_grad + k * b_grad,
                    plane_coeffs=(1.0 - k) * prev.plane_coeffs + k * coeffs,
                )
            else:
                self._baselines[parity] = _ParityBaseline(
                    core_std=c_std,
                    top_std=t_std,
                    bot_std=b_std,
                    core_grad=c_grad,
                    top_grad=t_grad,
                    bot_grad=b_grad,
                    plane_coeffs=coeffs,
                )

    def predict_occupancy(self, bands: Dict[str, SquareBands]) -> Dict[str, SquareOccupancy]:
        """Scores top, core, and bot occupancy [0, 1] relative to empty light/dark baselines."""
        out: Dict[str, SquareOccupancy] = {}
        for sq, b in bands.items():
            ref = self._baselines.get(b.is_light)
            if ref is None:
                out[sq] = SquareOccupancy(top=0.0, core=0.0, bot=0.0)
                continue

            expected_mean = float(
                ref.plane_coeffs[0] + ref.plane_coeffs[1] * b.row + ref.plane_coeffs[2] * b.col
            )
            albedo_dev = abs(b.core_mean - expected_mean)

            top_score = _region_occupancy(
                std_val=b.top_std,
                empty_std=ref.top_std,
                grad_val=b.top_grad,
                empty_grad=ref.top_grad,
                albedo_dev=0.0,
            )
            core_score = _region_occupancy(
                std_val=b.core_std,
                empty_std=ref.core_std,
                grad_val=b.core_grad,
                empty_grad=ref.core_grad,
                albedo_dev=albedo_dev,
            )
            bot_score = _region_occupancy(
                std_val=b.bot_std,
                empty_std=ref.bot_std,
                grad_val=b.bot_grad,
                empty_grad=ref.bot_grad,
                albedo_dev=0.0,
            )
            out[sq] = SquareOccupancy(top=top_score, core=core_score, bot=bot_score)
        return out


def _region_occupancy(
    std_val: float,
    empty_std: float,
    grad_val: float,
    empty_grad: float,
    albedo_dev: float,
) -> float:
    std_excess = max(0.0, std_val - max(empty_std * 1.45, empty_std + 3.0)) / max(7.0, empty_std * 2.5)
    grad_excess = max(0.0, grad_val - max(empty_grad * 1.45, empty_grad + 0.012)) / max(0.04, empty_grad * 2.2)
    alb_excess = max(0.0, albedo_dev - 14.0) / 36.0
    raw = max(std_excess, grad_excess, alb_excess * 0.90) + 0.35 * min(std_excess, grad_excess)
    return float(max(0.0, min(1.0, raw)))


def _patch_topk_grad(patch_g: np.ndarray, frac: float = 0.28) -> float:
    if patch_g.size == 0:
        return 0.0
    flat = patch_g.reshape(-1)
    k = max(1, int(round(flat.size * frac)))
    if k >= flat.size:
        return float(np.mean(flat))
    topk = np.partition(flat, flat.size - k)[-k:]
    return float(np.mean(topk))


def _patch_ssim_distance(blur_b: np.ndarray, blur_a: np.ndarray) -> float:
    """Computes normalized structural dissimilarity (1 - SSIM) over a grayscale patch."""
    if blur_b.size == 0 or blur_a.size == 0:
        return 0.0
    a = blur_b.astype(np.float32)
    b = blur_a.astype(np.float32)
    if min(a.shape[:2]) >= 5:
        mu1 = cv2.GaussianBlur(a, (5, 5), 1.0)
        mu2 = cv2.GaussianBlur(b, (5, 5), 1.0)
        s1 = np.maximum(0.0, cv2.GaussianBlur(a * a, (5, 5), 1.0) - mu1 * mu1)
        s2 = np.maximum(0.0, cv2.GaussianBlur(b * b, (5, 5), 1.0) - mu2 * mu2)
        s12 = cv2.GaussianBlur(a * b, (5, 5), 1.0) - mu1 * mu2
    else:
        mu1 = float(np.mean(a))
        mu2 = float(np.mean(b))
        s1 = float(np.var(a))
        s2 = float(np.var(b))
        s12 = float(np.mean((a - mu1) * (b - mu2)))
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    num = (2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)
    den = (mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)
    ssim = float(np.mean(num / np.maximum(den, 1e-6)))
    return float(max(0.0, min(1.0, 1.0 - ssim)))


def _patch_change(
    blur_b: np.ndarray,
    blur_a: np.ndarray,
    diff_map: np.ndarray,
    edge_map: np.ndarray,
) -> float:
    if diff_map.size == 0:
        return 0.0
    l_diff = float(np.mean(diff_map)) / 255.0
    zm_diff = float(
        np.mean(np.abs((blur_b - float(np.mean(blur_b))) - (blur_a - float(np.mean(blur_a)))))
    ) / 255.0
    e_diff = float(np.mean(edge_map)) / 255.0
    std_b = float(np.std(blur_b)) / 255.0
    std_a = float(np.std(blur_a)) / 255.0
    ssim_dist = _patch_ssim_distance(blur_b, blur_a)
    return max(zm_diff, e_diff * 4.0, abs(std_a - std_b) * 1.6, l_diff * 0.25, ssim_dist * 0.45)


def _to_gamma_blur_grad(
    warped: np.ndarray,
    scale_ref_mean: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Converts BGR/gray warped board into exposure-normalized gamma-0.5 blur and gradient maps."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    lin = np.clip(gray.astype(np.float32), 0.0, 255.0)
    mean_lum = float(np.mean(lin))
    if scale_ref_mean is not None and mean_lum > 1e-3 and scale_ref_mean > 1e-3:
        scale = max(0.75, min(1.35, scale_ref_mean / mean_lum))
        lin = np.clip(lin * scale, 0.0, 255.0)
    gamma = np.sqrt(lin / 255.0) * 255.0
    blur = cv2.GaussianBlur(gamma, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy) / 255.0
    return blur, grad, mean_lum


class BoardOccupancyTracker:
    """Computes symmetric base-core deltas, vertical sub-band deltas, and 3D column occupancy scores."""

    def __init__(self) -> None:
        self.empty_model = EmptySquareModel()

    def reset(self) -> None:
        self.empty_model = EmptySquareModel()

    def extract_grid_mapping(
        self,
        warped_shape: Tuple[int, ...],
        orientation: BoardOrientation,
        roi_extractor: CellRoiExtractor,
    ) -> Tuple[Dict[str, Tuple[int, int]], Dict[Tuple[int, int], str]]:
        """Maps each algebraic square <-> image-space (row, col) grid indices [0..7]."""
        ah, aw = warped_shape[:2]
        cell_h = max(1.0, ah / 8.0)
        cell_w = max(1.0, aw / 8.0)
        sq_pos: Dict[str, Tuple[int, int]] = {}
        pos_to_sq: Dict[Tuple[int, int], str] = {}
        for sq in CHESS_SQUARES:
            x0, y0, _, _ = roi_extractor.get_square_coordinates(sq, orientation)
            rc = (int(round(y0 / cell_h)), int(round(x0 / cell_w)))
            sq_pos[sq] = rc
            pos_to_sq[rc] = sq
        return sq_pos, pos_to_sq

    def compute_pair_deltas(
        self,
        warped_after: np.ndarray,
        warped_before: np.ndarray,
        orientation: BoardOrientation,
        roi_extractor: CellRoiExtractor,
    ) -> Tuple[Dict[str, float], float, float]:
        """Computes normalized per-square base-core delta metrics."""
        blur_b, _, m_b = _to_gamma_blur_grad(warped_before, scale_ref_mean=None)
        blur_a, _, _ = _to_gamma_blur_grad(warped_after, scale_ref_mean=m_b)
        diff_map = np.abs(blur_b - blur_a)
        gxb = cv2.Sobel(blur_b, cv2.CV_32F, 1, 0, ksize=3)
        gyb = cv2.Sobel(blur_b, cv2.CV_32F, 0, 1, ksize=3)
        gxa = cv2.Sobel(blur_a, cv2.CV_32F, 1, 0, ksize=3)
        gya = cv2.Sobel(blur_a, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = np.hypot(gxa - gxb, gya - gyb) * 0.5
        h_img, w_img = diff_map.shape[:2]

        deltas: Dict[str, float] = {}
        for sq in CHESS_SQUARES:
            bx1, by1, bx2, by2 = roi_extractor.get_base_core_bbox(sq, orientation)
            cx1, cy1 = max(0, min(w_img, bx1)), max(0, min(h_img, by1))
            cx2, cy2 = max(0, min(w_img, bx2)), max(0, min(h_img, by2))
            deltas[sq] = _patch_change(
                blur_b[cy1:cy2, cx1:cx2],
                blur_a[cy1:cy2, cx1:cx2],
                diff_map[cy1:cy2, cx1:cx2],
                edge_map[cy1:cy2, cx1:cx2],
            )

        med_delta = float(np.median(list(deltas.values())))
        change_thresh = max(0.056, med_delta * 2.40)
        return deltas, med_delta, change_thresh

    def extract_bands(
        self,
        blur: np.ndarray,
        grad: np.ndarray,
        orientation: BoardOrientation,
        roi_extractor: CellRoiExtractor,
        sq_pos: Dict[str, Tuple[int, int]],
    ) -> Dict[str, SquareBands]:
        """Extracts core, upper, and lower appearance stats for all 64 squares."""
        h_img, w_img = blur.shape[:2]
        out: Dict[str, SquareBands] = {}
        for sq in CHESS_SQUARES:
            x0, y0, x1, y1 = roi_extractor.get_square_coordinates(sq, orientation)
            cw = x1 - x0
            ch = y1 - y0
            sq_idx = chess.parse_square(sq)
            is_light = (chess.square_file(sq_idx) + chess.square_rank(sq_idx)) % 2 == 1
            r, c = sq_pos[sq]

            bx1 = max(0, min(w_img, int(round(x0 + 0.22 * cw))))
            bx2 = max(0, min(w_img, int(round(x0 + 0.78 * cw))))
            by1 = max(0, min(h_img, int(round(y0 + 0.22 * ch))))
            by2 = max(0, min(h_img, int(round(y0 + 0.78 * ch))))

            tx1 = max(0, min(w_img, int(round(x0 + 0.18 * cw))))
            tx2 = max(0, min(w_img, int(round(x0 + 0.82 * cw))))
            ty1 = max(0, min(h_img, int(round(y0 + 0.14 * ch))))
            ty2 = max(0, min(h_img, int(round(y0 + 0.46 * ch))))

            lx1, lx2 = tx1, tx2
            ly1 = max(0, min(h_img, int(round(y0 + 0.54 * ch))))
            ly2 = max(0, min(h_img, int(round(y0 + 0.90 * ch))))

            core_p = blur[by1:by2, bx1:bx2]
            top_p = blur[ty1:ty2, tx1:tx2]
            bot_p = blur[ly1:ly2, lx1:lx2]

            out[sq] = SquareBands(
                square=sq,
                row=r,
                col=c,
                is_light=is_light,
                core_mean=float(np.mean(core_p)) if core_p.size > 0 else 0.0,
                core_std=float(np.std(core_p)) if core_p.size > 0 else 0.0,
                core_grad=_patch_topk_grad(grad[by1:by2, bx1:bx2]),
                top_std=float(np.std(top_p)) if top_p.size > 0 else 0.0,
                top_grad=_patch_topk_grad(grad[ty1:ty2, tx1:tx2]),
                bot_std=float(np.std(bot_p)) if bot_p.size > 0 else 0.0,
                bot_grad=_patch_topk_grad(grad[ly1:ly2, lx1:lx2]),
            )
        return out

    def compute_subband_deltas(
        self,
        blur_b: np.ndarray,
        blur_a: np.ndarray,
        orientation: BoardOrientation,
        roi_extractor: CellRoiExtractor,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Computes temporal change separately for each square's upper band and bottom spillover band."""
        diff_map = np.abs(blur_b - blur_a)
        gxb = cv2.Sobel(blur_b, cv2.CV_32F, 1, 0, ksize=3)
        gyb = cv2.Sobel(blur_b, cv2.CV_32F, 0, 1, ksize=3)
        gxa = cv2.Sobel(blur_a, cv2.CV_32F, 1, 0, ksize=3)
        gya = cv2.Sobel(blur_a, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = np.hypot(gxa - gxb, gya - gyb) * 0.5
        h_img, w_img = diff_map.shape[:2]

        delta_top: Dict[str, float] = {}
        delta_bot: Dict[str, float] = {}
        for sq in CHESS_SQUARES:
            x0, y0, x1, y1 = roi_extractor.get_square_coordinates(sq, orientation)
            cw = x1 - x0
            ch = y1 - y0
            tx1 = max(0, min(w_img, int(round(x0 + 0.18 * cw))))
            tx2 = max(0, min(w_img, int(round(x0 + 0.82 * cw))))
            ty1 = max(0, min(h_img, int(round(y0 + 0.14 * ch))))
            ty2 = max(0, min(h_img, int(round(y0 + 0.46 * ch))))
            ly1 = max(0, min(h_img, int(round(y0 + 0.54 * ch))))
            ly2 = max(0, min(h_img, int(round(y0 + 0.90 * ch))))

            delta_top[sq] = _patch_change(
                blur_b[ty1:ty2, tx1:tx2],
                blur_a[ty1:ty2, tx1:tx2],
                diff_map[ty1:ty2, tx1:tx2],
                edge_map[ty1:ty2, tx1:tx2],
            )
            delta_bot[sq] = _patch_change(
                blur_b[ly1:ly2, tx1:tx2],
                blur_a[ly1:ly2, tx1:tx2],
                diff_map[ly1:ly2, tx1:tx2],
                edge_map[ly1:ly2, tx1:tx2],
            )
        return delta_top, delta_bot

    def evaluate_transition(
        self,
        board: chess.Board,
        warped_after: np.ndarray,
        warped_before: np.ndarray,
        orientation: BoardOrientation,
        roi_extractor: CellRoiExtractor,
        candidate_generator: LegalMoveCandidateGenerator,
        base_deltas: Optional[Tuple[Dict[str, float], float, float]] = None,
        nn_occ_before: Optional[Dict[str, float]] = None,
        nn_occ_after: Optional[Dict[str, float]] = None,
    ) -> TransitionEvidence:
        """Evaluates 1-ply and 2-ply legal moves jointly via base deltas, 3D lean attribution, and occupancy."""
        if base_deltas is None:
            deltas, med_delta, change_thresh = self.compute_pair_deltas(
                warped_after, warped_before, orientation, roi_extractor
            )
        else:
            deltas, med_delta, change_thresh = base_deltas

        synthetic_precomputed = base_deltas is not None and np.array_equal(warped_after, warped_before)
        sq_pos, pos_to_sq = self.extract_grid_mapping(warped_after.shape, orientation, roi_extractor)
        blur_b, grad_b, m_b = _to_gamma_blur_grad(warped_before, scale_ref_mean=None)
        blur_a, grad_a, _ = _to_gamma_blur_grad(warped_after, scale_ref_mean=m_b)

        bands_b = self.extract_bands(blur_b, grad_b, orientation, roi_extractor, sq_pos)
        bands_a = self.extract_bands(blur_a, grad_a, orientation, roi_extractor, sq_pos)
        delta_top, delta_bot = self.compute_subband_deltas(blur_b, blur_a, orientation, roi_extractor)

        known_empty_before = [sq for sq in CHESS_SQUARES if board.piece_at(chess.parse_square(sq)) is None]
        if len(known_empty_before) >= 6:
            ema = 1.0 if not self.empty_model.is_calibrated else 0.25
            self.empty_model.fit(bands_b, known_empty_before, ema_alpha=ema)

        occ_b = self.empty_model.predict_occupancy(bands_b)
        occ_a = self.empty_model.predict_occupancy(bands_a)

        # Self-calibrated neural occupancy fusion (chesscog ResNet occupancy_classifier):
        # Measure NN reliability against known board occupancy before trusting its probabilities.
        if nn_occ_before and nn_occ_after and len(nn_occ_before) == 64 and len(nn_occ_after) == 64:
            correct = 0
            for sq in CHESS_SQUARES:
                exp_occ = board.piece_at(chess.parse_square(sq)) is not None
                if (nn_occ_before.get(sq, 0.0) >= 0.50) == exp_occ:
                    correct += 1
            nn_acc = correct / 64.0
            trust_w = 0.45 if nn_acc >= 0.85 else (0.25 if nn_acc >= 0.72 else 0.12)
            for occ_map, nn_map in ((occ_b, nn_occ_before), (occ_a, nn_occ_after)):
                for sq in CHESS_SQUARES:
                    p_nn = float(max(0.0, min(1.0, nn_map.get(sq, 0.0))))
                    cur = occ_map[sq]
                    blended_core = (1.0 - trust_w) * cur.core + trust_w * p_nn
                    blended_top = (1.0 - 0.5 * trust_w) * cur.top + (0.5 * trust_w) * p_nn
                    occ_map[sq] = SquareOccupancy(top=blended_top, core=blended_core, bot=cur.bot)

        secondary_thresh = max(med_delta * 1.85, change_thresh * 0.62)
        relaxed_sec = max(med_delta * 1.10, 0.020)

        def up_of(sq: str) -> Optional[str]:
            r, c = sq_pos[sq]
            return pos_to_sq.get((r - 1, c))

        def down_of(sq: str) -> Optional[str]:
            r, c = sq_pos[sq]
            return pos_to_sq.get((r + 1, c))

        # Per-square departure & arrival evidence combining temporal change, upward lean, and occupancy
        scored_1ply: List[Tuple[float, chess.Move, List[str], Set[str]]] = []

        def score_single_move(
            b_state: chess.Board,
            move: chess.Move,
            d_base: Dict[str, float],
            occ_prev: Dict[str, SquareOccupancy],
            occ_next: Dict[str, SquareOccupancy],
        ) -> Optional[Tuple[float, List[str], Set[str]]]:
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            aff = candidate_generator.get_affected_squares(b_state, move)
            aff_set = set(aff)
            up_f = up_of(from_sq)
            up_t = up_of(to_sq)

            explained: Set[str] = set(aff_set)
            if up_f is not None and (
                delta_bot[up_f] >= relaxed_sec
                or abs(occ_prev[up_f].bot - occ_next[up_f].bot) >= 0.14
                or abs(bands_a[up_f].core_mean - bands_b[up_f].core_mean) >= 20.0
            ):
                explained.add(up_f)

            # 1. Departure evidence at vacated square(s) (includes castle rook origin)
            vacated_sqs = [from_sq]
            arrived_sqs = [to_sq]
            if b_state.is_castling(move) and len(aff) >= 4:
                for s in aff:
                    if s not in (from_sq, to_sq):
                        if b_state.piece_at(chess.parse_square(s)) is not None:
                            vacated_sqs.append(s)
                        else:
                            arrived_sqs.append(s)
            elif b_state.is_en_passant(move):
                cap_sq = chess.square_name(
                    chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
                )
                if cap_sq not in vacated_sqs:
                    vacated_sqs.append(cap_sq)

            dep_scores: List[float] = []
            for v_sq in vacated_sqs:
                uv = up_of(v_sq)
                d_v = d_base[v_sq]
                occ_drop = max(0.0, occ_prev[v_sq].combined - occ_next[v_sq].combined)
                if (
                    uv is not None
                    and uv not in aff_set
                    and (occ_next[v_sq].top < 0.40 or occ_drop >= 0.15)
                    and (
                        (occ_prev[uv].bot - occ_next[uv].bot) >= 0.16
                        or abs(bands_a[uv].core_mean - bands_b[uv].core_mean) >= 24.0
                        or (delta_bot[uv] >= change_thresh and delta_top[uv] < delta_bot[uv] * 0.55)
                    )
                    and delta_top[uv] < max(change_thresh, delta_bot[uv] * 0.65)
                ):
                    d_v = max(d_v, 0.78 * delta_bot[uv])
                after_top = occ_next[v_sq].top
                after_core = occ_next[v_sq].core
                # Subtle same-color disappearance signal (e.g. e7 dark pawn on dark square)
                if after_top < 0.35 and occ_drop >= 0.16:
                    d_v = max(d_v, 0.10 * occ_drop + 0.025)
                # Still visibly standing on from_sq -> conservation veto only when both top & core stay occupied,
                # unless an adjacent piece below leans onto from_sq while top isn't solidly filled and albedo jumped
                stand_lvl = min(after_top, after_core)
                dv = down_of(v_sq)
                below_leans = (
                    dv is not None
                    and b_state.piece_at(chess.parse_square(dv)) is not None
                    and dv not in aff_set
                    and (occ_next[dv].top >= 0.35 or occ_next[dv].core >= 0.35 or d_base[dv] < relaxed_sec)
                )
                albedo_jump = abs(bands_a[v_sq].core_mean - bands_b[v_sq].core_mean)
                if (
                    not synthetic_precomputed
                    and stand_lvl > 0.50
                    and not (below_leans and albedo_jump >= 18.0)
                ):
                    empty_factor = max(0.02, 1.0 - (stand_lvl - 0.50) / 0.35)
                else:
                    empty_factor = 1.0
                dep_scores.append(d_v * empty_factor)

            # 2. Arrival evidence at destination square(s)
            arr_scores: List[float] = []
            for a_sq in arrived_sqs:
                ua = up_of(a_sq)
                da = down_of(a_sq)
                d_a = d_base[a_sq]
                lean_used = False
                occ_gain = occ_next[a_sq].combined - occ_prev[a_sq].combined
                alb_a = abs(bands_a[a_sq].core_mean - bands_b[a_sq].core_mean)
                if occ_gain >= 0.20:
                    d_a = max(d_a, 0.10 * occ_gain + 0.03)
                if not synthetic_precomputed and ua is not None and ua not in aff_set:
                    ua_occ_gain = max(
                        occ_next[ua].combined - occ_prev[ua].combined,
                        occ_next[ua].bot - occ_prev[ua].bot,
                    )
                    ua_alb = abs(bands_a[ua].core_mean - bands_b[ua].core_mean)
                    a_footprint = d_base[a_sq] >= 0.015 or occ_next[a_sq].bot >= 0.15 or occ_next[a_sq].core >= 0.15
                    if (
                        a_footprint
                        and (ua_occ_gain >= 0.18 or ua_alb >= 35.0)
                        and (b_state.piece_at(chess.parse_square(ua)) is None or delta_top[ua] < max(change_thresh, delta_bot[ua] * 0.65))
                    ):
                        lean_d = 0.85 * max(d_base[ua], delta_bot[ua])
                        if lean_d > d_a:
                            d_a = lean_d
                            lean_used = True

                # Require genuine physical arrival (not <secondary noise on an already-occupied neighbor)
                has_real_arrival = (
                    d_base[a_sq] >= secondary_thresh
                    or lean_used
                    or occ_gain >= 0.20
                    or alb_a >= 20.0
                    or (synthetic_precomputed and d_base[a_sq] >= change_thresh)
                )
                if not has_real_arrival:
                    d_a = 0.0

                if ua is not None and (
                    lean_used
                    or (occ_next[ua].bot - occ_prev[ua].bot >= 0.14)
                    or (
                        occ_next[ua].bot >= occ_prev[ua].bot - 0.05
                        and occ_next[ua].core >= occ_prev[ua].core - 0.15
                        and delta_bot[ua] >= change_thresh
                    )
                ):
                    explained.add(ua)

                # Mirage veto: if a_sq top+core look empty WHILE down(a_sq) below shows a newly arrived piece
                to_occ_factor = 1.0
                if not synthetic_precomputed and occ_next[a_sq].top < 0.28 and occ_next[a_sq].core < 0.35:
                    ua_occ = (
                        max(occ_next[ua].bot, occ_next[ua].core, d_base[ua] * 2.0, delta_bot[ua] * 2.0)
                        if ua is not None
                        else 0.0
                    )
                    if ua is None or ua_occ < 0.35:
                        to_occ_factor = 0.35
                    elif not lean_used and da is not None and occ_next[da].top >= 0.45 and d_base[da] >= relaxed_sec:
                        # a_sq is just the spillover of an actual piece moving onto down(a_sq)
                        to_occ_factor = 0.18
                    elif da is not None and da not in aff_set and occ_next[da].top >= 0.55 and d_base[da] >= relaxed_sec:
                        to_occ_factor = 0.22
                arr_scores.append(d_a * to_occ_factor)

            min_dep = min(dep_scores)
            min_arr = min(arr_scores)
            if max(min_dep, min_arr) < change_thresh * 0.85:
                return None
            if min(min_dep, min_arr) < relaxed_sec:
                return None

            geom_mean = (max(1e-4, min_dep) * max(1e-4, min_arr)) ** 0.5
            return geom_mean, aff, explained

        # Effective change set (squares with direct delta OR strong upward lean change below them)
        effective_active: Set[str] = {sq for sq, d in deltas.items() if d >= change_thresh}

        for move in board.legal_moves:
            res = score_single_move(board, move, deltas, occ_b, occ_a)
            if res is not None:
                sc, aff, exp = res
                scored_1ply.append((sc, move, aff, exp))

        # Only treat non-affected squares as significant changes when occupancy appeared/vanished,
        # or visual polarity flipped without staying identically occupied by a stationary non-capture piece.
        capture_targets = {chess.square_name(m.to_square) for m in board.legal_moves if board.is_capture(m)}
        significant_changed: Set[str] = set()
        for sq in CHESS_SQUARES:
            if deltas[sq] < max(change_thresh * 1.25, 0.085):
                continue
            if synthetic_precomputed:
                significant_changed.add(sq)
                continue
            occ_diff = abs(occ_a[sq].combined - occ_b[sq].combined)
            cm_b = bands_b[sq].core_mean
            cm_a = bands_a[sq].core_mean
            both_solid = min(occ_b[sq].top, occ_a[sq].top) >= 0.85 and min(occ_b[sq].core, occ_a[sq].core) >= 0.85
            polarity_flip = (
                max(cm_b, cm_a) >= 155.0
                and min(cm_b, cm_a) <= 135.0
                and abs(cm_a - cm_b) >= 38.0
                and (not both_solid or sq in capture_targets or min(occ_b[sq].core, occ_a[sq].core) < 0.85)
            )
            if occ_diff >= 0.22 or polarity_flip:
                significant_changed.add(sq)

        major_thresh = max(change_thresh * 1.15, 0.065)
        total_major_delta = sum(d for d in deltas.values() if d >= major_thresh)

        # Only attribute unexplained changes where square isn't merely leaning below-to-above spillover
        def filter_unexplained(cand_sqs: Sequence[str], explained_set: Set[str]) -> List[str]:
            out = []
            for sq in cand_sqs:
                if sq in explained_set:
                    continue
                da = down_of(sq)
                # Ignore square whose top stayed empty when square directly below explains it
                if occ_a[sq].top < 0.28 and da is not None and da in explained_set:
                    continue
                out.append(sq)
            return out

        best_1ply_score = max((item[0] for item in scored_1ply), default=-1.0)
        scored_1ply.sort(key=lambda item: item[0], reverse=True)

        competitive_1ply: List[Tuple[float, chess.Move, List[str], Set[str]]] = [
            item for item in scored_1ply if item[0] >= best_1ply_score * 0.72
        ]

        selected_changed: Set[str] = set()
        expected_dest_piece: Dict[str, str] = {}
        pending_m2: Optional[chess.Move] = None

        if competitive_1ply:
            best_sc, best_mv, best_aff, best_exp = competitive_1ply[0]
            explained_delta = sum(deltas[sq] for sq in best_exp if deltas[sq] >= major_thresh)
            unexplained_major_max = max(
                (d for sq, d in deltas.items() if sq not in best_exp and d >= major_thresh),
                default=0.0,
            )
            best_exp_max = max((deltas[sq] for sq in best_exp), default=0.0)
            low_share = (
                total_major_delta >= 0.25
                and (explained_delta / max(1e-4, total_major_delta)) < 0.36
                and unexplained_major_max >= max(change_thresh * 1.25, best_exp_max * 0.70)
            )
            unexplained_top = filter_unexplained(sorted(significant_changed), best_exp)
            if unexplained_top or low_share:
                # Prefer a joint 2-ply pair where BOTH m1 and m2 explain real occupancy changes
                two_ply = _search_joint_two_ply(
                    board=board,
                    candidate_generator=candidate_generator,
                    score_fn=lambda b_st, mv: score_single_move(b_st, mv, deltas, occ_b, occ_a),
                    significant_changed=significant_changed,
                    filter_unexplained=filter_unexplained,
                    seed_squares=effective_active | significant_changed,
                )
                if two_ply is not None:
                    m1_j, aff1_j, m2_j = two_ply
                    pending_m2 = m2_j
                    selected_changed = set(aff1_j)
                    to1 = chess.square_name(m1_j.to_square)
                    mp = board.piece_at(m1_j.from_square)
                    if mp is not None and m1_j.promotion is None:
                        expected_dest_piece[to1] = f"{'w' if mp.color == chess.WHITE else 'b'}{mp.symbol().upper()}"
                else:
                    b2 = board.copy()
                    b2.push(best_mv)
                    best_m2_res: Optional[Tuple[float, chess.Move, List[str], Set[str]]] = None
                    if not low_share and any(u in best_exp for u in significant_changed):
                        for m2 in b2.legal_moves:
                            res2 = score_single_move(b2, m2, deltas, occ_b, occ_a)
                            if res2 is None:
                                continue
                            sc2, aff2, exp2 = res2
                            if any(u in exp2 and u not in best_exp for u in unexplained_top):
                                if best_m2_res is None or sc2 > best_m2_res[0]:
                                    best_m2_res = (sc2, m2, aff2, exp2)
                    if best_m2_res is not None:
                        pending_m2 = best_m2_res[1]
                        selected_changed = set(best_aff)
                        to1 = chess.square_name(best_mv.to_square)
                        moved_piece = board.piece_at(best_mv.from_square)
                        if moved_piece is not None and best_mv.promotion is None:
                            p_prefix = "w" if moved_piece.color == chess.WHITE else "b"
                            expected_dest_piece[to1] = f"{p_prefix}{moved_piece.symbol().upper()}"
                    elif low_share:
                        # Suppress isolated low-share ripple commits (e.g. noise on h2/h4 when f5/g6 shifted)
                        competitive_1ply = []

            if competitive_1ply and not selected_changed:
                dest_cands: Dict[str, List[Tuple[float, str]]] = {}
                for sc, mv, aff, _ in competitive_1ply:
                    for sq in aff:
                        selected_changed.add(sq)
                    to_sq = chess.square_name(mv.to_square)
                    mp = board.piece_at(mv.from_square)
                    if mp is not None and mv.promotion is None:
                        cls_name = f"{'w' if mp.color == chess.WHITE else 'b'}{mp.symbol().upper()}"
                        dest_cands.setdefault(to_sq, []).append((sc, cls_name))
                for to_sq, items in dest_cands.items():
                    if len(items) == 1 or items[0][0] > items[1][0] * 1.12:
                        expected_dest_piece[to_sq] = items[0][1]
        elif len(significant_changed) >= 2:
            # No strong 1-ply: search out-of-order / fast 2-ply pair directly (e.g. Nf3 followed by Bg7)
            two_ply = _search_joint_two_ply(
                board=board,
                candidate_generator=candidate_generator,
                score_fn=lambda b_st, mv: score_single_move(b_st, mv, deltas, occ_b, occ_a),
                significant_changed=significant_changed,
                filter_unexplained=filter_unexplained,
                seed_squares=effective_active | significant_changed,
            )
            if two_ply is not None:
                m1_j, aff1_j, m2_j = two_ply
                pending_m2 = m2_j
                selected_changed = set(aff1_j)
                to1 = chess.square_name(m1_j.to_square)
                mp = board.piece_at(m1_j.from_square)
                if mp is not None and m1_j.promotion is None:
                    expected_dest_piece[to1] = f"{'w' if mp.color == chess.WHITE else 'b'}{mp.symbol().upper()}"

        anomaly_squares: List[str] = []
        if not selected_changed:
            sec_sqs = [sq for sq in CHESS_SQUARES if deltas[sq] >= secondary_thresh]
            if len(effective_active) >= 2 or (len(effective_active) == 1 and len(sec_sqs) >= 2):
                anomaly_squares = sorted(sec_sqs, key=lambda s: deltas[s], reverse=True)[:4]
            elif len(significant_changed) >= 2:
                anomaly_squares = sorted(significant_changed, key=lambda s: deltas[s], reverse=True)[:4]

        return TransitionEvidence(
            deltas=deltas,
            med_delta=med_delta,
            change_thresh=change_thresh,
            secondary_thresh=secondary_thresh,
            changed_sqs=selected_changed,
            expected_dest_piece=expected_dest_piece,
            pending_two_ply_second_move=pending_m2,
            anomaly_squares=anomaly_squares,
        )


def _search_joint_two_ply(
    board: chess.Board,
    candidate_generator: LegalMoveCandidateGenerator,
    score_fn,
    significant_changed: Set[str],
    filter_unexplained,
    seed_squares: Set[str],
) -> Optional[Tuple[chess.Move, List[str], chess.Move]]:
    """Finds the best legal 2-ply sequence (m1, m2) when two moves settle together or out of turn."""
    if not seed_squares:
        return None
    best_total = -1.0
    best_pair: Optional[Tuple[chess.Move, List[str], chess.Move]] = None

    # Pre-filter m1 candidates that touch at least one active/changed square
    m1_cands: List[Tuple[chess.Move, List[str]]] = []
    for m1 in board.legal_moves:
        aff1 = candidate_generator.get_affected_squares(board, m1)
        if any(sq in seed_squares for sq in aff1):
            m1_cands.append((m1, aff1))
    if len(m1_cands) > 40:
        return None

    for m1, aff1 in m1_cands:
        res1 = score_fn(board, m1)
        if res1 is None:
            continue
        sc1, _, exp1 = res1
        if not any(u in exp1 for u in significant_changed):
            continue
        b2 = board.copy()
        b2.push(m1)
        for m2 in b2.legal_moves:
            aff2 = candidate_generator.get_affected_squares(b2, m2)
            if not any(sq in seed_squares for sq in aff2):
                continue
            res2 = score_fn(b2, m2)
            if res2 is None:
                continue
            sc2, _, exp2 = res2
            # Require m2 to explain at least one square not already explained by m1 alone
            if not any(u in exp2 and u not in exp1 for u in significant_changed):
                continue
            combined_exp = exp1 | exp2
            if filter_unexplained(sorted(significant_changed), combined_exp):
                continue
            total = sc1 + sc2
            if total > best_total:
                best_total = total
                best_pair = (m1, aff1, m2)
    return best_pair


@dataclass
class TransitionEvidence:
    """Result of physical 3D transition & occupancy evaluation."""

    deltas: Dict[str, float]
    med_delta: float
    change_thresh: float
    secondary_thresh: float
    changed_sqs: Set[str]
    expected_dest_piece: Dict[str, str]
    pending_two_ply_second_move: Optional[chess.Move]
    anomaly_squares: List[str]
