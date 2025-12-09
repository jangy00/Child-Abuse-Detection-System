import os
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from M2 import FeatureExtractor
from M3 import AbuseDetector


class RealtimeAbuseDetector:
    """
    기존 배치 파이프라인(M1 → M2 → M3)의 규칙을 그대로 사용하면서
    프레임 단위로 결과를 스트리밍하는 실시간 분석용 클래스.
    """

    def __init__(self, model_path: str, conf: float = 0.5, iou: float = 0.5) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}")

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

        # M2 / M3 그대로 사용
        self.feature_extractor = FeatureExtractor()
        self.abuse_detector = AbuseDetector()

        # 누적 데이터 
        self._rows: List[Dict] = []

        # 최종 결과 캐시
        self.features_df: Optional[pd.DataFrame] = None
        self.alerts_df: Optional[p.DataFrame] = None  # type: ignore

        # 메타 정보
        self.fps: float = 0.0
        self.total_frames: int = 0

    # ----------------------------
    # 내부 유틸
    # ----------------------------
    def _init_video_meta(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if self.total_frames <= 0:
            raise ValueError(f"비디오 프레임 수를 읽을 수 없습니다: {video_path}")

    def _append_frame_rows(self, frame_idx: int, result) -> None:
        """YOLO 추론 결과에서 M1과 동일한 raw_data row 생성"""
        boxes = getattr(result, "boxes", None)
        kpts_obj = getattr(result, "keypoints", None)

        if boxes is None or kpts_obj is None or len(boxes) == 0:
            return

        # track id / class / conf
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else np.arange(len(boxes.cls))
        )
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        kpts = kpts_obj.data.cpu().numpy()  # (N, K, 3)
        num_det = min(len(ids), kpts.shape[0])

        for i in range(num_det):
            track_id = int(ids[i])

            # 🔐 데이터셋 규칙: ID 1, 2만 사용
            if track_id > 2:
                continue

            raw_cls = int(clss[i])
            conf = float(confs[i])
            keypoints_flat = kpts[i].reshape(-1).tolist()

            # 🔐 ID 기반 child/adult 고정
            #   - ID1 → ADULT(1)
            #   - ID2 → CHILD(0)
            if track_id == 1:
                fixed_cls = 1
            elif track_id == 2:
                fixed_cls = 0
            else:
                # 혹시 모를 다른 ID가 들어온 경우 YOLO 원본 클래스 사용
                fixed_cls = raw_cls

            self._rows.append(
                {
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class": fixed_cls,   # 0=child, 1=adult (파이프라인 공통)
                    "conf": conf,
                    "keypoints": keypoints_flat,
                }
            )

    def _run_rules_up_to(self, frame_idx: int) -> Tuple[int, str, float, float]:
        """
        지금까지 누적된 row를 가지고 M2/M3 전체 규칙을 적용한 뒤,
        현재 프레임 기준 상태코드/라벨 및 요약 통계를 계산.
        """
        if not self._rows:
            self.features_df = pd.DataFrame()
            self.alerts_df = pd.DataFrame()
            return 0, "정상", 0.0, float("inf")

        preds_df = pd.DataFrame(self._rows)

        # M2: 특징 추출
        features_df = self.feature_extractor.process(preds_df.copy())

        # M3: 규칙 기반 알림
        alerts_df = self.abuse_detector.detect(features_df.copy())

        self.features_df = features_df
        self.alerts_df = alerts_df

        status_code = 0
        status_label = "정상"

        if alerts_df is not None and not alerts_df.empty:
            # 현재 프레임에 해당하는 알림만 필터링
            current_alerts = alerts_df[
                (alerts_df["start_frame"] <= frame_idx)
                & (alerts_df["end_frame"] >= frame_idx)
            ]
            if not current_alerts.empty and "type" in current_alerts.columns:
                if (current_alerts["type"] == "abuse_report").any():
                    status_code = 2
                    status_label = "학대 신고 알람"
                elif (current_alerts["type"] == "suspicious").any():
                    status_code = 1
                    status_label = "의심 행동"

        # 현재 프레임 기준 통계 (성인 기준)
        max_adult_vel = 0.0
        min_adult_child_dist = float("inf")

        if self.features_df is not None and not self.features_df.empty:
            frame_df = self.features_df[self.features_df["frame"] == frame_idx]

            adults = frame_df[frame_df["class"] == 1]
            if not adults.empty:
                if "limb_velocity" in adults.columns:
                    max_adult_vel = float(adults["limb_velocity"].max())

                if "min_dist_to_victim" in adults.columns:
                    d = adults["min_dist_to_victim"]
                    if not d.empty:
                        min_adult_child_dist = float(d.min())

        return status_code, status_label, max_adult_vel, min_adult_child_dist

    # ----------------------------
    # Public API
    # ----------------------------
    def stream_video(
        self,
        video_path: str,
        progress_callback=None,
    ) -> Generator[Dict, None, None]:
        """
        비디오 전체를 한 번만 추론하면서,
        매 프레임마다 (annotated_frame, 상태, 통계)를 스트리밍.

        yield 값 구조:
        {
            "frame_idx": int,
            "frame_time": float,
            "annotated_frame": np.ndarray(H, W, 3),
            "status": {"code": int, "label": str},
            "stats": {
                "max_adult_velocity": float,
                "min_adult_child_dist": float,
            },
        }
        """
        self._init_video_meta(video_path)

        # YOLO tracking 스트림
        results = self.model.track(
            source=video_path,
            stream=True,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            persist=True,
        )

        for frame_idx, result in enumerate(results):
            # 진행률 콜백
            if self.total_frames > 0:
                progress = (frame_idx + 1) / self.total_frames
            else:
                progress = 0.0

            if progress_callback is not None:
                progress_callback(progress, "실시간 추론 및 규칙 적용 중...")

            # row 누적
            self._append_frame_rows(frame_idx, result)

            # M2/M3 규칙 적용 후 현재 프레임 상태 계산
            status_code, status_label, max_adult_vel, min_dist = self._run_rules_up_to(frame_idx)

            # 시각화용 프레임 (BGR → RGB)
            try:
                annotated = result.plot()
            except Exception:
                annotated = result.orig_img

            if annotated is None:
                continue

            if annotated.ndim == 3:
                annotated_rgb = annotated[..., ::-1]
            else:
                annotated_rgb = annotated

            frame_time = frame_idx / self.fps if self.fps > 0 else 0.0

            yield {
                "frame_idx": frame_idx,
                "frame_time": frame_time,
                "annotated_frame": annotated_rgb,
                "status": {"code": status_code, "label": status_label},
                "stats": {
                    "max_adult_velocity": float(max_adult_vel),
                    "min_adult_child_dist": float(min_dist),
                },
            }

        # 마지막 진행률 콜백
        if progress_callback is not None:
            progress_callback(1.0, "실시간 분석 완료")

    def get_final_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """실시간 스트리밍 이후 최종 features_df / alerts_df 반환"""
        if self.features_df is None:
            self.features_df = pd.DataFrame()
        if self.alerts_df is None:
            self.alerts_df = pd.DataFrame()
        return self.features_df, self.alerts_df
