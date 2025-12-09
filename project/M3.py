import pandas as pd
import numpy as np

class AbuseDetector:
    """
    규칙 기반 아동 학대/의심행동 탐지기

    핵심 개념
    ----------
    1) 학대(빨간 알람, type='abuse_report')
       - Adult 손/발이 Child에게 실제로 닿는 접촉(contact)
       - + Child의 명확한 반응(방어 자세, 구부림/넘어짐, 급격한 거리 증가, 반응 속도 증가)

    2) 의심행동(주황 알람, type='suspicious')
       - Adult 손/발 제스처가 Child 방향으로 일정 시간 이상 지속될 때
       - Child 반응 여부와 무관
       - 실제 접촉이 없어도 되는 “전조 동작”

    3) 기본 전제
       - Adult와 Child가 서로 상호작용하지 않고 개별로 움직일 때는
         학대/의심 어느 쪽에도 포함되지 않아야 함
    """
    def __init__(self):
        # === 의심행동(전조증상) 기준 ===
        self.SUSPICIOUS_VELOCITY_THRESH = 2.5       # 전조 제스처용 최소 속도
        self.SUSPICIOUS_PROXIMITY_THRESH = 260.0    # 전조 제스처용 거리 (Adult↔Child bbox 중심 거리)
        self.SUSPICIOUS_GESTURE_MIN_FRAMES = 10     # 전조 제스처 최소 지속 프레임수 (나중에 FPS보고 조정)

        # === 학대(빨간 알람) 기준 ===
        # 접촉 + 반응을 강하게 보기 위해 속도/거리 기준은 살짝만 둔다.
        self.ABUSE_VELOCITY_THRESH = 3.0            # 학대 상황으로 보기 위한 성인 최소 속도
        self.ABUSE_PROXIMITY_THRESH = 260.0         # 학대 상황에서 adult_prox가 이 값 안쪽이면 가중치↑
        self.ABUSE_MIN_FRAMES = 2                   # 학대 조건이 최소 2프레임 이상 유지될 때 알람 생성

        # === 키포인트 기반 접촉/반응 기준 ===
        self.CONTACT_KPT_DIST_THRESH = 80.0         # 손/발 ↔ 아동 머리/몸 최소 거리 (px)
        self.DISTANCE_INCREASE_THRESH = 60.0        # 프레임 간 Adult↔Child 거리 증가량 (px) (기존 20 → 60)
        self.CHILD_REACTION_VEL_MIN = 3.0           # 아동 반응 속도 (FeatureExtractor에서 계산한 값 기준)
        self.FACE_TOUCH_THRESH = 80.0               # 손-얼굴 거리 (얼굴 가리기/보호 자세)
        self.TWIST_RATIO_THRESH = 0.35              # 몸 비틀림 정도

        # === 성인 손/팔 방향 & 제스처 기준 ===
        self.ADULT_ARM_DIRECTION_THRESHOLD = 80.0   # 팔 벡터가 Child 방향과 이루는 최대 각도
        self.ADULT_HAND_MOVEMENT_THRESH = 2.0       # 손 움직임 크기 (제스처 강도)

        # === 상태 추적 ===
        self.suspicious_tracker = {}                # (adult_id, child_id)별 의심 제스처 프레임 카운트
        self.abuse_tracker = {}                     # (adult_id, child_id)별 학대 조건 지속 프레임 카운트
        self.hand_history = {}                      # (adult_id)별 손 위치 히스토리

    # ------------------------------------------------------------------
    #  키포인트 기반 보조 함수들
    # ------------------------------------------------------------------
    def _calc_child_center(self, child_kpts):
        valid = [kp[:2] for kp in child_kpts if kp[2] > 0.3]
        if not valid:
            return None
        return np.mean(valid, axis=0)

    def _calc_child_twist_ratio(self, child_kpts):
        """어깨/엉덩이 축으로 몸 비틀림 정도 계산"""
        try:
            ls = child_kpts[5][:2] if child_kpts[5][2] > 0.3 else None
            rs = child_kpts[6][:2] if child_kpts[6][2] > 0.3 else None
            lh = child_kpts[11][:2] if child_kpts[11][2] > 0.3 else None
            rh = child_kpts[12][:2] if child_kpts[12][2] > 0.3 else None

            if ls is None or rs is None or lh is None or rh is None:
                return 0.0

            shoulder_center = (ls + rs) / 2
            hip_center = (lh + rh) / 2

            horiz = abs(shoulder_center[0] - hip_center[0])
            vert = abs(shoulder_center[1] - hip_center[1])
            if vert <= 0:
                return 0.0
            return horiz / vert
        except Exception:
            return 0.0

    def _calc_child_face_touch_distance(self, child_kpts):
        """아동 손-얼굴(코 기준) 최소 거리"""
        try:
            nose = child_kpts[0][:2] if child_kpts[0][2] > 0.3 else None
            lw = child_kpts[9][:2] if child_kpts[9][2] > 0.3 else None
            rw = child_kpts[10][:2] if child_kpts[10][2] > 0.3 else None

            if nose is None:
                return 999.0

            d_min = float("inf")
            for w in (lw, rw):
                if w is None:
                    continue
                d = np.linalg.norm(nose - w)
                d_min = min(d_min, d)
            return d_min if d_min != float("inf") else 999.0
        except Exception:
            return 999.0

    def _check_contact_kpts(self, adult_kpts, child_kpts):
        """
        성인 손/발(손목, 무릎)과 아동 머리/어깨/엉덩이 사이 최소 거리로 접촉 여부 판단
        """
        try:
            adult_idx = [9, 10, 13, 14]    # 손목 + 무릎
            child_idx = [0, 1, 2, 3, 4, 5, 6, 11, 12]  # 머리/어깨/엉덩이

            min_dist = float("inf")
            for ai in adult_idx:
                if ai >= len(adult_kpts) or adult_kpts[ai][2] <= 0.3:
                    continue
                a_xy = adult_kpts[ai][:2]
                for ci in child_idx:
                    if ci >= len(child_kpts) or child_kpts[ci][2] <= 0.3:
                        continue
                    c_xy = child_kpts[ci][:2]
                    d = np.linalg.norm(a_xy - c_xy)
                    if d < min_dist:
                        min_dist = d

            if min_dist < self.CONTACT_KPT_DIST_THRESH:
                return True, min_dist
            return False, min_dist if min_dist != float("inf") else 999.0
        except Exception:
            return False, 999.0

    def _adult_hand_toward_child(self, adult_kpts, child_kpts, adult_prox):
        """
        성인 손/팔 방향이 Child 쪽을 향하는지 판단.
        - 팔 벡터 vs Child 방향 벡터 각도
        - + Child와 많이 가까워진 경우(보정)
        """
        try:
            child_center = self._calc_child_center(child_kpts)
            if child_center is None:
                return False, 0

            score = 0

            # 왼팔/오른팔 공통 처리
            for elbow_idx, shoulder_idx, wrist_idx in [
                (7, 5, 9),   # 왼팔
                (8, 6, 10),  # 오른팔
            ]:
                if wrist_idx >= len(adult_kpts):
                    continue
                wrist = adult_kpts[wrist_idx]
                if wrist[2] <= 0.3:
                    continue

                base = None
                if elbow_idx < len(adult_kpts) and adult_kpts[elbow_idx][2] > 0.3:
                    base = adult_kpts[elbow_idx][:2]
                elif shoulder_idx < len(adult_kpts) and adult_kpts[shoulder_idx][2] > 0.3:
                    base = adult_kpts[shoulder_idx][:2]

                if base is None:
                    continue

                hand_vec = wrist[:2] - base
                to_child_vec = child_center - base

                if np.linalg.norm(hand_vec) <= 1e-3 or np.linalg.norm(to_child_vec) <= 1e-3:
                    continue

                cos = np.dot(hand_vec, to_child_vec) / (
                    np.linalg.norm(hand_vec) * np.linalg.norm(to_child_vec)
                )
                cos = max(-1.0, min(1.0, cos))
                angle = np.degrees(np.arccos(cos))

                if angle < self.ADULT_ARM_DIRECTION_THRESHOLD:
                    score += 1

            # 손 방향이 정확하지 않더라도
            # Adult↔Child bbox 거리가 매우 가까우면 상호작용 가중치 부여
            if adult_prox < 200:
                score += 1

            return score > 0, score
        except Exception:
            return False, 0

    def _adult_hand_moving(self, adult_id, adult_kpts):
        """
        성인의 손 움직임(제스처 강도) 측정.
        - track_id별로 손목 좌표를 저장하고 프레임 간 이동량 계산
        """
        try:
            hands = []
            for idx in [9, 10]:
                if idx < len(adult_kpts) and adult_kpts[idx][2] > 0.3:
                    hands.append(adult_kpts[idx][:2])

            if adult_id not in self.hand_history:
                self.hand_history[adult_id] = []
            hist = self.hand_history[adult_id]
            hist.append(hands)

            if len(hist) < 2 or not hist[-2] or not hist[-1]:
                return False

            # 직전 프레임과 현재 프레임 사이 평균 손 이동거리 계산
            prev, cur = hist[-2], hist[-1]
            movements = []
            for i in range(min(len(prev), len(cur))):
                d = np.linalg.norm(cur[i] - prev[i])
                movements.append(d)

            if not movements:
                return False

            avg_move = float(np.mean(movements))
            return avg_move > self.ADULT_HAND_MOVEMENT_THRESH
        except Exception:
            return False

    # ------------------------------------------------------------------
    #  메인 탐지 로직
    # ------------------------------------------------------------------
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        # class: 1=adult, 0=child (기존 파이프라인 가정)
        adult_df = df[df["class"] == 1].sort_values("frame").copy()
        child_df = df[df["class"] == 0].sort_values("frame").copy()
        if adult_df.empty or child_df.empty:
            return pd.DataFrame()

        frames = sorted(adult_df["frame"].unique())

        # 상태 초기화
        self.suspicious_tracker.clear()
        self.abuse_tracker.clear()
        self.hand_history.clear()

        alerts = []

        # 프레임 간 Adult↔Child 거리 변화를 보기 위한 저장소
        prev_dist = {}

        for frame in frames:
            af = adult_df[adult_df["frame"] == frame]
            cf = child_df[child_df["frame"] == frame]
            if af.empty or cf.empty:
                continue

            for _, a_row in af.iterrows():
                adult_id = a_row["track_id"]
                adult_vel = float(a_row.get("limb_velocity", 0.0))
                adult_prox = float(a_row.get("min_dist_to_victim", 999.0))
                adult_kpts = np.array(a_row["keypoints"]).reshape(15, 3)

                # 제스처 강도
                hand_moving = self._adult_hand_moving(adult_id, adult_kpts)

                for _, c_row in cf.iterrows():
                    child_id = c_row["track_id"]
                    child_kpts = np.array(c_row["keypoints"]).reshape(15, 3)

                    pair_key = (adult_id, child_id)

                    # 거리 증가량 계산 
                    cur_dist = adult_prox
                    if pair_key in prev_dist:
                        dist_inc = max(0.0, cur_dist - prev_dist[pair_key])
                    else:
                        dist_inc = 0.0
                    prev_dist[pair_key] = cur_dist

                    # Child 반응 관련 피처
                    hands_above = int(c_row.get("child_hands_above_head", 0))
                    bending = int(c_row.get("child_bending_posture", 0))
                    child_fall = int(c_row.get("child_fall", 0))
                    child_react_vel = float(c_row.get("child_reaction_velocity", 0.0))

                    face_touch_dist = self._calc_child_face_touch_distance(child_kpts)
                    twist_ratio = self._calc_child_twist_ratio(child_kpts)

                    protective_pose = (
                        hands_above >= 1
                        or bending == 1
                        or child_fall == 1
                        or face_touch_dist < self.FACE_TOUCH_THRESH
                        or twist_ratio > self.TWIST_RATIO_THRESH
                    )

                    motion_reaction = (
                        child_react_vel > self.CHILD_REACTION_VEL_MIN
                        or dist_inc > self.DISTANCE_INCREASE_THRESH
                    )

                    child_reaction = protective_pose or motion_reaction

                    # Adult 손/팔 방향이 Child 쪽인지
                    toward_child, toward_score = self._adult_hand_toward_child(
                        adult_kpts, child_kpts, adult_prox
                    )

                    # 관절 기반 접촉 여부
                    contact, min_kpt_dist = self._check_contact_kpts(
                        adult_kpts, child_kpts
                    )

                    # --------------------------------------------------
                    # 1) 학대(빨간 알람) 후보 조건
                    #    - Adult와 Child가 실제로 상호작용하는 상황만 허용
                    # --------------------------------------------------
                    abuse_raw = False

                    # Case 1: 실제 접촉 + Child 반응
                    if contact and adult_vel > self.ABUSE_VELOCITY_THRESH and child_reaction:
                        abuse_raw = True

                    # Case 2: 비접촉 학대
                    #   - Adult 손/팔이 Child 방향이고
                    #   - 속도가 충분히 크며
                    #   - 아래 두 가지 중 하나 이상:
                    #       (1) 아동이 얼굴을 가리거나(손이 위로 올라감)
                    #       (2) 아동이 눈에 띄게 뒤로 물러남(거리 증가)
                    elif (
                        not contact
                        and toward_child
                        and adult_vel > self.ABUSE_VELOCITY_THRESH
                        and (
                            hands_above >= 1
                            or dist_inc > self.DISTANCE_INCREASE_THRESH
                        )
                    ):
                        abuse_raw = True

                    # Adult와 Child가 서로 전혀 상호작용하지 않는 경우(멀고, 방향도 안 맞고, 접촉도 없음)는
                    # abuse_raw가 False로 남게 됨.

                    # --------------------------------------------------
                    # 2) 의심행동(주황 알람) 후보 조건
                    #    - Adult 손/발 제스처가 Child 방향으로 일정 시간 지속
                    #    - Child 반응과 무관
                    # --------------------------------------------------
                    suspicious_raw = (
                        hand_moving
                        and adult_vel > self.SUSPICIOUS_VELOCITY_THRESH
                        and adult_prox < self.SUSPICIOUS_PROXIMITY_THRESH
                        and (toward_child or contact)  # 방향이 맞거나, 매우 근접/접촉 상태
                    )

                    # --------------------------------------------------
                    # Tracker 업데이트 & 알람 생성
                    # --------------------------------------------------
                    # 학대 tracker
                    if abuse_raw:
                        self.abuse_tracker[pair_key] = self.abuse_tracker.get(pair_key, 0) + 1
                    else:
                        if pair_key in self.abuse_tracker:
                            self.abuse_tracker[pair_key] = max(
                                0, self.abuse_tracker[pair_key] - 0.5
                            )

                    # 의심 tracker
                    if suspicious_raw and not abuse_raw:
                        # 학대와 동시에 의심으로 카운트하지 않음 
                        self.suspicious_tracker[pair_key] = (
                            self.suspicious_tracker.get(pair_key, 0) + 1
                        )
                    else:
                        if pair_key in self.suspicious_tracker:
                            # 제스처가 끊기면 리셋
                            self.suspicious_tracker[pair_key] = 0

                    # 실제 알람 생성
                    # 1) 학대 알람
                    abuse_frames = self.abuse_tracker.get(pair_key, 0)
                    if abuse_frames >= self.ABUSE_MIN_FRAMES:
                        base_conf = float(a_row.get("conf", 0.5))

                        # 심각도 점수
                        severity = 0.0
                        if contact:
                            severity += 0.4
                        if child_fall == 1:
                            severity += 0.3
                        if dist_inc > self.DISTANCE_INCREASE_THRESH:
                            severity += 0.2
                        if hands_above >= 1 or bending == 1:
                            severity += 0.2

                        vel_ratio = min(adult_vel / max(self.ABUSE_VELOCITY_THRESH, 1e-3), 3.0)
                        conf = min(base_conf * (0.6 + severity + 0.2 * vel_ratio), 0.99)

                        detail = (
                            f"🚨 학대신고: 접촉={'Y' if contact else 'N'}, "
                            f"관절거리 {min_kpt_dist:.0f}px, bbox거리 {adult_prox:.0f}px, "
                            f"성인속도 {adult_vel:.1f}, 아동반응(손:{hands_above},구부림:{bending},"
                            f"넘어짐:{child_fall},거리증가:{dist_inc:.0f}px,반응속도:{child_react_vel:.1f})"
                        )

                        alerts.append(
                            {
                                "start_frame": max(0, frame - int(abuse_frames) + 1),
                                "end_frame": frame + 3,
                                "perpetrator_id": adult_id,
                                "victim_id": child_id,
                                "type": "abuse_report",
                                "confidence": conf,
                                "details": detail,
                                "frame": frame,
                            }
                        )
                        # 같은 구간에서 중복 생성 방지
                        self.abuse_tracker[pair_key] = 0

                    # 2) 의심 알람
                    susp_frames = self.suspicious_tracker.get(pair_key, 0)
                    if susp_frames >= self.SUSPICIOUS_GESTURE_MIN_FRAMES:
                        base_conf = float(a_row.get("conf", 0.5))
                        dur_ratio = min(
                            susp_frames / max(self.SUSPICIOUS_GESTURE_MIN_FRAMES, 1e-3), 3.0
                        )
                        vel_ratio = min(
                            adult_vel / max(self.SUSPICIOUS_VELOCITY_THRESH, 1e-3), 3.0
                        )
                        conf = min(base_conf * (0.4 + 0.3 * dur_ratio + 0.2 * vel_ratio), 0.85)

                        detail = (
                            f"의심행동: 성인 손/발 제스처가 아동 방향으로 지속 | "
                            f"거리 {adult_prox:.0f}px, 제스처프레임 {susp_frames}"
                        )

                        alerts.append(
                            {
                                "start_frame": max(0, frame - int(susp_frames) + 1),
                                "end_frame": frame + 2,
                                "perpetrator_id": adult_id,
                                "victim_id": child_id,
                                "type": "suspicious",
                                "confidence": conf,
                                "details": detail,
                                "frame": frame,
                            }
                        )
                        # 연속 의심 구간이 한 번에만 찍히도록 리셋
                        self.suspicious_tracker[pair_key] = 0

        # --------------------------------------------------------------
        # 후처리: abuse와 suspicious 중첩 구간 정리
        # --------------------------------------------------------------
        if not alerts:
            return pd.DataFrame()

        result = pd.DataFrame(alerts).sort_values("start_frame")

        # abuse 구간과 겹치는 suspicious는 제거
        abuse_mask = result["type"] == "abuse_report"

        abuse_intervals = result[abuse_mask][["start_frame", "end_frame", "perpetrator_id", "victim_id"]].values

        keep_indices = []
        for idx, row in result.iterrows():
            if row["type"] == "abuse_report":
                keep_indices.append(idx)
                continue

            # suspicious일 때 abuse 구간과 겹치는지 체크
            s, e = row["start_frame"], row["end_frame"]
            pid, vid = row["perpetrator_id"], row["victim_id"]

            overlap = False
            for a_s, a_e, a_pid, a_vid in abuse_intervals:
                if pid == a_pid and vid == a_vid:
                    if not (e < a_s or s > a_e):
                        overlap = True
                        break
            if not overlap:
                keep_indices.append(idx)

        result = result.loc[keep_indices].reset_index(drop=True)
        return result
