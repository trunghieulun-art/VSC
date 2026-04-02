import difflib
import json
import math
from typing import Dict, List, Set, Tuple

from config import SpellCheckerConfig
from layout import get_keyboard_coordinates, keyboard_matrix
from text_utils import to_standard_telex


class MLSpellChecker:
    def __init__(
        self,
        config: SpellCheckerConfig,
        debug: bool = False,
        detail_log: bool = False,
    ) -> None:
        self.cfg = config
        self.debug = debug
        self.detail_log = detail_log

        print("Loading model...")
        with open(self.cfg.model_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab: Set[str] = set(data["vocab"])
        self.unigrams: Dict[str, int] = data["unigrams"]
        self.bigrams: Dict[str, int] = data["bigrams"]

        self.telex_to_vocab: Dict[str, List[str]] = {}
        for w in self.vocab:
            t = to_standard_telex(w)
            if t not in self.telex_to_vocab:
                self.telex_to_vocab[t] = []
            self.telex_to_vocab[t].append(w)

        self.telex_vocab_list: List[str] = list(self.telex_to_vocab.keys())

        self.kb_coords = get_keyboard_coordinates(keyboard_matrix=keyboard_matrix)
        print(f"Done! The dictionary has {len(self.vocab)} words.")

    # SINH CANDIDATE
    def get_candidates(
        self, error_word: str, prev_word: str | None = None
    ) -> List[str]:
        candidates: List[str] = []

        # Ép chữ lỗi về chuẩn Telex
        error_telex = to_standard_telex(error_word)

        # Lọc từ Bigram (Những từ từng đi liền sau prev_word)
        if prev_word:
            context_words: List[str] = [
                key.split()[1]
                for key in self.bigrams.keys()
                if key.startswith(f"{prev_word} ")
            ]

            if context_words:
                # Tạo mapping telex tạm thời cho tập context words
                context_telex_to_word: Dict[str, List[str]] = {}
                for cw in context_words:
                    ct = to_standard_telex(cw)
                    if ct not in context_telex_to_word:
                        context_telex_to_word[ct] = []
                    context_telex_to_word[ct].append(cw)

                context_telex_list = list(context_telex_to_word.keys())

                # Fuzzy match trên tập con TELEX
                context_telex_matches: List[str] = difflib.get_close_matches(
                    error_telex,
                    context_telex_list,
                    n=self.cfg.top_n,
                    cutoff=self.cfg.cutoff,
                )

                # Ánh xạ ngược từ Telex về chữ Tiếng Việt thật
                for ctm in context_telex_matches:
                    for real_word in context_telex_to_word[ctm]:
                        if real_word not in candidates:
                            candidates.append(real_word)

        # Bổ sung từ Unigram nếu chưa đủ số lượng top_n
        if len(candidates) < self.cfg.top_n:
            # Tìm trên toàn bộ từ điển TELEX
            general_telex_matches: List[str] = difflib.get_close_matches(
                error_telex,
                self.telex_vocab_list,
                n=self.cfg.top_n,
                cutoff=self.cfg.cutoff,
            )
            for gtm in general_telex_matches:
                for real_word in self.telex_to_vocab[gtm]:
                    if real_word not in candidates:
                        candidates.append(real_word)

        return candidates[: self.cfg.top_n]

    def get_kb_cost(self, char1: str, char2: str) -> float:
        # Tính phí phạt khi gõ nhầm char1 thành char2 dựa trên tọa độ bàn phím
        if char1 == char2:
            return 0.0

        if char1 not in self.kb_coords or char2 not in self.kb_coords:
            return self.cfg.unknown_char_penalty

        x1, y1 = self.kb_coords[char1]
        x2, y2 = self.kb_coords[char2]
        # Công thức tính khoảng cách học năm c2, c3 và calculus
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Chuẩn hóa khoảng cách.
        return min(dist / self.cfg.max_kb_distance, 1.0)

    def keyboard_aware_similarity(self, word1: str, word2: str) -> float:
        # Thuật toán Damerau-Levenshtein kết hợp Khoảng cách bàn phím

        m, n = len(word1), len(word2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        # Khởi tạo ma trận
        for i in range(m + 1):
            dp[i][0] = float(i)
        for j in range(n + 1):
            dp[0][j] = float(j)

        # Tính toán chi phí biến đổi
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = self.get_kb_cost(word1[i - 1], word2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,  # Deletion (Xóa 1 ký tự)
                    dp[i][j - 1] + 1.0,  # Insertion (Thêm 1 ký tự)
                    dp[i - 1][j - 1] + cost,  # Substitution (Gõ nhầm ký tự)
                )

                # Phép toán đảo vị trí (DAMERAU)
                if (
                    i > 1
                    and j > 1
                    and word1[i - 1] == word2[j - 2]
                    and word1[i - 2] == word2[j - 1]
                ):
                    dp[i][j] = min(
                        dp[i][j], dp[i - 2][j - 2] + self.cfg.transposition_cost
                    )

        max_len = max(m, n)
        if max_len == 0:
            return 1.0

        # Đổi từ 'Khoảng cách' (0.0 -> max_len) sang 'Độ giống nhau' (0.0 -> 1.0)
        distance = dp[m][n]
        sim = 1.0 - (distance / max_len)

        return max(0.0, sim)

    # TÍNH ĐIỂM
    def calculate_score(
        self, candidate: str, error_word: str, prev_word: str | None
    ) -> float:
        error_telex: str = to_standard_telex(error_word)
        cand_telex: str = to_standard_telex(candidate)

        # sim_score: float = difflib.SequenceMatcher(
        #     None, error_telex, cand_telex
        # ).ratio()
        sim_score: float = self.keyboard_aware_similarity(error_telex, cand_telex)

        # Frequency Score (Dùng Add-1 Smoothing)
        freq_score: int = self.unigrams.get(candidate, 1)

        # Context Score (Bigram)
        context_score: float = 1.0
        if prev_word:
            bigram_key = f"{prev_word} {candidate}"
            context_score = float(self.bigrams.get(bigram_key, 1))

            # Stutter Penalty (Phạt nếu lặp từ)
            if candidate == prev_word:
                context_score *= self.cfg.stutter_penalty

        # TỔNG ĐIỂM
        total_score: float = (
            (sim_score**self.cfg.sim_weight)
            * math.log2(freq_score + 1)
            * (context_score**self.cfg.context_weight)
        )

        if self.debug and self.detail_log:
            prev_str = prev_word if prev_word else "[Đầu câu]"
            print(
                f"      ➜ Xét bước nhảy: '{prev_str}' -> '{candidate}' (Gõ sai: '{error_word}', Telex: {cand_telex})"
            )
            print(
                f"         + Sim     = {sim_score:.2f}  (Mũ {self.sim_weight} -> {sim_score**self.sim_weight:.3f})"
            )
            print(
                f"         + Freq    = {freq_score:<4} (Log2 -> {math.log2(freq_score + 1):.2f})"
            )
            print(
                f"         + Context = {context_score:<4} (Mũ {self.context_weight} -> {context_score**self.context_weight:.2f})"
            )
            print(f"         = ĐIỂM BƯỚC NHẢY = {total_score:.4f}")

        return total_score

    # VITERBI DECODING (Sửa lỗi theo ngữ cảnh toàn câu)
    def correct_sentence(self, sentence: str) -> str:
        words: List[str] = sentence.lower().split()
        if not words:
            return ""

        # LƯU TRỮ CÁC ĐƯỜNG ĐI: {ứng_viên_hiện_tại: (tổng_điểm, [lịch_sử_các_từ_từ_đầu_câu])}
        paths: Dict[str, Tuple[float, List[str]]] = {}

        # KHỞI TẠO VỚI TỪ ĐẦU TIÊN
        first_word = words[0]
        self.get_candidates(first_word, prev_word=None)
        first_candidates = self.get_candidates(first_word, prev_word=None)

        if first_word in self.vocab and first_word not in first_candidates:
            first_candidates.append(first_word)  # Luôn giữ lại từ gốc nếu nó có nghĩa
        if not first_candidates:
            first_candidates = [first_word]

        if self.debug:
            print(f"\n[VITERBI] Từ thứ 1: '{first_word}'")

        for cand in first_candidates:
            score = self.calculate_score(cand, first_word, prev_word=None)
            paths[cand] = (score, [cand])
            if self.debug:
                print(f"  Khởi tạo nhánh: '{cand} (TÍCH LŨY = {score:.4f})'")

        # DUYỆT QUA CÁC TỪ CÒN LẠI
        for i in range(1, len(words)):
            current_word = words[i]
            new_paths: Dict[str, Tuple[float, List[str]]] = {}

            # Sinh ứng viên
            candidates_set = set()
            for prev_cand in paths.keys():
                cands = self.get_candidates(current_word, prev_word=prev_cand)
                candidates_set.update(cands)

            # Dự phòng thêm ứng viên chung (không ngữ cảnh)
            candidates_set.update(self.get_candidates(current_word, prev_word=None))

            candidates = list(candidates_set)
            if current_word in self.vocab and current_word not in candidates:
                candidates.append(current_word)
            if not candidates:
                candidates = [current_word]

            if self.debug:
                print(f"\n[VITERBI] Từ thứ {i + 1}: '{current_word}'")

            step_log_data: List[Dict] = []

            # TÌM ĐƯỜNG NỐI TỐT NHẤT TỪ BƯỚC TRƯỚC SANG BƯỚC HIỆN TẠI
            for curr_cand in candidates:
                best_score = -1.0
                best_path = []
                best_prev = None
                best_step_score = 0.0
                best_prev_score = 0.0

                # Thử nối curr_cand vào tất cả các nhánh (prev_cand) của bước trước
                for prev_cand, (prev_score, prev_path) in paths.items():
                    # Điểm bước nhảy (Bigram Context + Similarity + Frequency)
                    step_score = self.calculate_score(
                        curr_cand, current_word, prev_cand
                    )

                    # ĐIỂM TÍCH LŨY = Điểm lịch sử * Điểm bước nhảy
                    total_score = prev_score * step_score

                    # Lưu lại nhánh có điểm tích lũy cao nhất dẫn đến curr_cand
                    if total_score > best_score:
                        best_score = total_score
                        best_path = prev_path + [curr_cand]

                        if self.debug:
                            best_prev = prev_cand
                            best_step_score = step_score
                            best_prev_score = prev_score

                new_paths[curr_cand] = (best_score, best_path)

                if self.debug and best_prev:
                    step_log_data.append(
                        {
                            "cand": curr_cand,
                            "path": f"{best_prev} -> {curr_cand}",
                            "calc_str": f"({best_prev_score:.4f} * {best_step_score:.4f})",
                            "score": best_score,
                        }
                    )

            # Cập nhật các đường đi cho vòng lặp tiếp theo
            paths = new_paths

            if self.debug and step_log_data:
                # Sắp xếp theo score từ cao xuống thấp
                step_log_data.sort(key=lambda x: x["score"], reverse=True)

                print("-" * 82)
                print(
                    f"| {'ỨNG VIÊN':<12} | {'TUYẾN TỐT NHẤT':<20} | {'LỊCH SỬ * ĐIỂM BƯỚC NHẢY':<25} | {'TỔNG ĐIỂM':<12} |"
                )
                print("-" * 82)
                for item in step_log_data:
                    print(
                        f"| {item['cand']:<12} | {item['path']:<20} | {item['calc_str']:<25} | {item['score']:<12.4f} |"
                    )
                print("-" * 82)

        # Chọn ra tuyến đường cuối cùng có tổng điểm cao nhất
        best_final_candidate = max(paths.keys(), key=lambda k: paths[k][0])
        _, best_sentence = paths[best_final_candidate]

        return " ".join(best_sentence)

    """
    # PIPELINE cũ
    def correct_sentence(self, sentence: str, debug=False):
        # Tách từ từ câu gốc
        words: List[str] = sentence.lower().split()
        corrected_words: List[str] = []

        for i, word in enumerate(words):
            # 1. Nếu từ đã đúng (nằm trong từ điển) -> Giữ nguyên
            if word in self.vocab:
                corrected_words.append(word)
                if debug:
                    print(f"Bỏ qua: '{word}' (Đã có trong từ điển)")
                continue

            if debug:
                print(f"\n[DEBUG] PHÁT HIỆN TỪ SAI: '{word}'")

            prev_word: str | None = corrected_words[-1] if i > 0 else None

            # 2. Nếu từ nghi ngờ sai -> Sinh candidates
            candidates: List[str] = self.get_candidates(word, prev_word=prev_word)

            # Nếu không tìm được candidate nào hao hao -> Đành giữ nguyên từ gốc
            if not candidates:
                corrected_words.append(word)
                if debug:
                    print(f"Không tìm thấy ứng viên nào cho '{word}'. Giữ nguyên.")
                continue

            if debug:
                print(f"Ứng viên (Candidates): {candidates}")

            best_candidate: str = word
            max_score: float = -1

            for cand in candidates:
                score: float = self.calculate_score(cand, word, prev_word, debug=debug)

                # Cập nhật ứng viên điểm cao nhất
                if score > max_score:
                    max_score: float = score
                    best_candidate: str = cand

            if debug:
                print(
                    f"CHỐT THAY THẾ: '{word}' ➔ '{best_candidate}' (Điểm cao nhất: {max_score:.2f})\n"
                )
            # 4. Chốt ứng viên tốt nhất vào câu
            corrected_words.append(best_candidate)

        # Ghép lại thành câu hoàn chỉnh
        return " ".join(corrected_words)
    """
