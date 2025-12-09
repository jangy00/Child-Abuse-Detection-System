import math  # 진행률 계산용
import streamlit as st
import pandas as pd
import tempfile
import os
import sys
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import zipfile
from io import BytesIO
import uuid
import cv2
from typing import Dict, Generator, List, Optional, Tuple

# YOLO 및 프로젝트 모듈 임포트
from ultralytics import YOLO

# === 경로 설정 ======================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# project_root/project
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "project"))

for p in [CURRENT_DIR, PROJECT_DIR]:
    if p not in sys.path:
        sys.path.append(p)
# ===============================================================

from M2 import FeatureExtractor
from M3 import AbuseDetector

# ======================================================================================
#   Streamlit 페이지 설정
# ======================================================================================

st.set_page_config(
    page_title="아동 학대 자동 탐지 시스템",
    page_icon="🧒",
    layout="wide",
)

# 스타일링
st.markdown(
    """
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #ff4b4b; text-align: center; }
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("아동 학대 자동 탐지 시스템")
st.markdown("실시간 분석")

# 탭 구성
tab1, tab2 = st.tabs(["영상 분석", "결과 및 보고서"])

# 세션 상태 초기화 - 세션 상태 유지
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.video_path = None
    st.session_state.output_video_path = None
    st.session_state.alerts_df = None
    st.session_state.preds_df = None
    st.session_state.video_duration = 0.0
    st.session_state.video_fps = 0.0
    st.session_state.alert_clips = []
    st.session_state.analysis_complete = False
    st.session_state.speed_factor = 1
    st.session_state.abuse_clips = []

# 임포트 함수 (배치 모드용)
@st.cache_resource
def load_processing_functions():
    try:
        from main import process_video, create_annotated_video, create_alert_clips
        return process_video, create_annotated_video, create_alert_clips
    except ImportError as e:
        st.error(f"모듈 임포트 오류: {e}")
        return None, None, None


process_video, create_annotated_video, create_alert_clips = load_processing_functions()

# 임시 파일 정리 함수 
def cleanup_temp_files():
    """임시 파일 정리"""
    temp_dir = os.path.join(os.getcwd(), "temp_videos")
    if os.path.exists(temp_dir):
        current_time = time.time()
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if current_time - file_time > 3600:  # 1시간 이상 지난 파일
                        os.remove(file_path)
            except Exception:
                pass


def cleanup_system_temp():
    """시스템 임시 파일 정리"""
    temp_dir = tempfile.gettempdir()
    if os.path.exists(temp_dir):
        current_time = time.time()
        for filename in os.listdir(temp_dir):
            if filename.endswith(".mp4"):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        file_time = os.path.getmtime(file_path)
                        if current_time - file_time > 3600:
                            os.remove(file_path)
                except Exception:
                    pass


# =====================================================================
#  결과/타임라인 헬퍼 함수
# =====================================================================

def build_status_timeline_from_alerts(alerts_df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    alerts_df(start_frame/end_frame, type)을 기준으로
    프레임별 상태 타임라인 DataFrame 생성.
    time 컬럼은 (frame / fps) 기준의 '압축된 시간'으로 사용.
    (vid_stride를 적용한 논리 프레임 기준이라, 고속모드일수록 전체 시간이 짧아짐
    """
    if alerts_df is None or alerts_df.empty or fps <= 0:
        return pd.DataFrame()

    records: List[Dict] = []

    for _, row in alerts_df.iterrows():
        start_f = int(row.get("start_frame", 0))
        end_f = int(row.get("end_frame", start_f))
        t = row.get("type", "normal")

        if t == "normal":
            code = 0
            label = "정상"
        elif t == "suspicious":
            code = 1
            label = "의심"
        elif t == "abuse_report":
            code = 2
            label = "학대신고"
        else:
            code = 0
            label = "정상"

        for f in range(start_f, end_f + 1):
            records.append(
                {
                    "frame": f,
                    "time": f / fps,
                    "status_code": code,
                    "status_label": label,
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).drop_duplicates(subset=["frame"]).sort_values("frame")
    return df


def build_status_velocity_df(
    alerts_df: pd.DataFrame, features_df: pd.DataFrame, fps: float
) -> pd.DataFrame:
    """
    상태(정상/의심/학대) + 성인/아동 속도 + 근접거리까지 통합한 프레임 단위 DataFrame 생성.
    time 축은 vid_stride가 적용된 '압축 시간(초)' 기준.
    """
    if features_df is None or features_df.empty or fps <= 0:
        return pd.DataFrame()

    frames = sorted(features_df["frame"].unique().tolist())
    status_df = pd.DataFrame({"frame": frames})

    status_df["status_code"] = 0
    status_df["status_label"] = "정상"

    if alerts_df is not None and not alerts_df.empty:
        for _, row in alerts_df.iterrows():
            start_f = int(row.get("start_frame", 0))
            end_f = int(row.get("end_frame", start_f))
            t = row.get("type", "normal")

            mask = (status_df["frame"] >= start_f) & (status_df["frame"] <= end_f)
            if t == "abuse_report":
                status_df.loc[mask, "status_code"] = 2
                status_df.loc[mask, "status_label"] = "학대신고"
            elif t == "suspicious":
                status_df.loc[mask & (status_df["status_code"] < 2), "status_code"] = 1
                status_df.loc[mask & (status_df["status_code"] < 2), "status_label"] = "의심"

    stats = []
    for f in frames:
        frame_df = features_df[features_df["frame"] == f]

        adults = frame_df[frame_df["class"] == 1]
        max_adult_vel = float(adults["limb_velocity"].max()) \
            if "limb_velocity" in adults.columns and not adults.empty else 0.0

        childs = frame_df[frame_df["class"] == 0]
        max_child_vel = float(childs["limb_velocity"].max()) \
            if "limb_velocity" in childs.columns and not childs.empty else 0.0

        min_dist = float("inf")
        if "min_dist_to_victim" in adults.columns and not adults.empty:
            d = adults["min_dist_to_victim"]
            if not d.empty:
                min_dist = float(d.min())

        stats.append(
            {
                "frame": f,
                "max_adult_velocity": max_adult_vel,
                "max_child_velocity": max_child_vel,
                "min_adult_child_dist": min_dist,
            }
        )

    stats_df = pd.DataFrame(stats)
    merged = status_df.merge(stats_df, on="frame", how="left")
    merged["time"] = merged["frame"] / fps
    return merged


def compute_status_summary(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """
    alerts.csv 기반으로 상태별 구간(정상/의심/학대신고)을 요약하는 함수.
    start_time / end_time 컬럼이 없으면 start_frame / end_frame 기준으로 임시 생성.
    """
    if alerts_df is None or alerts_df.empty:
        return pd.DataFrame()

    alerts_df = alerts_df.copy()

    if ("start_time" not in alerts_df.columns) or ("end_time" not in alerts_df.columns):
        if ("start_frame" in alerts_df.columns) and ("end_frame" in alerts_df.columns):
            alerts_df["start_time"] = alerts_df["start_frame"].astype(float)
            alerts_df["end_time"] = alerts_df["end_frame"].astype(float)
        else:
            alerts_df["start_time"] = 0.0
            alerts_df["end_time"] = 0.0

    alerts_df["status_code"] = alerts_df["type"].map(
        {"normal": 0, "suspicious": 1, "abuse_report": 2}
    )

    alerts_df["duration"] = alerts_df["end_time"] - alerts_df["start_time"]

    status_summary = (
        alerts_df.groupby(["status_code", "type"])["duration"]
        .agg(["count", "sum"])
        .reset_index()
    )
    status_summary.rename(
        columns={"count": "구간 수", "sum": "총 지속 시간(초)"}, inplace=True
    )

    return status_summary


def create_abuse_clips(
    video_path: str,
    alerts_df: pd.DataFrame,
    fps: float,
    clip_duration: float = 3.0,
    pre_margin: float = 1.0,
) -> List[Dict]:
    """
    type == 'abuse_report' 구간을 2~3초 클립으로 잘라 개별 다운로드용 mp4 생성.
    반환: [{ 'label': '...', 'path': '/.../clip0.mp4', 'start': t0, 'end': t1 }, ...]
    """
    if alerts_df is None or alerts_df.empty or fps <= 0 or not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration = total_frames / fps if fps > 0 else 0.0

    temp_dir = os.path.join(os.getcwd(), "temp_clips")
    os.makedirs(temp_dir, exist_ok=True)

    clips: List[Dict] = []
    abuse_rows = alerts_df[alerts_df["type"] == "abuse_report"]

    clip_idx = 0
    for _, row in abuse_rows.iterrows():
        # 논리 프레임 → 원본 프레임으로 환산
        logical_start_f = int(row.get("start_frame", 0))

        # vid_stride 기반으로 원본 프레임 환산
        real_start_f = int(logical_start_f * st.session_state.speed_factor)

        # 원본 시간 기준으로 클립 시작
        start_time = max(0.0, (real_start_f / fps) - pre_margin)
        end_time = min(video_duration, start_time + clip_duration)

        start_frame_clip = int(start_time * fps)
        end_frame_clip = int(end_time * fps)

        if start_frame_clip >= end_frame_clip:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_clip)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if width == 0 or height == 0:
            continue

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        clip_filename = f"abuse_clip_{clip_idx}.mp4"
        clip_path = os.path.join(temp_dir, clip_filename)
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        current_f = start_frame_clip
        while current_f <= end_frame_clip:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_f += 1

        out.release()

        clips.append(
            {
                "label": f"학대 구간 클립 #{clip_idx + 1} ({start_time:.1f}s ~ {end_time:.1f}s)",
                "path": clip_path,
                "start": start_time,
                "end": end_time,
            }
        )
        clip_idx += 1

    cap.release()
    return clips

# ======================================================================================
#   review 학대 구간 클립 영상 생성
# ======================================================================================

def create_review_video(
    video_path: str,
    alerts_df: pd.DataFrame,
    fps: float,
    speed_factor: int,
    clip_output_dir: str = "temp_review",
) -> Optional[str]:
    """
    - 의심(suspicious) / 학대(abuse_report) 구간만 상단 색 바 + 텍스트로 표시한
      '전체 분석 완료 영상'을 생성하고 mp4 경로를 반환.
    - 라벨/키포인트/박스는 적용 x
    """

    if (
        alerts_df is None
        or alerts_df.empty
        or not os.path.exists(video_path)
        or fps <= 0
    ):
        return None

    os.makedirs(clip_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if total_frames <= 0 or width == 0 or height == 0:
        cap.release()
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_name = f"review_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(clip_output_dir, out_name)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # -----------------------------------------
    # logical_frame(=배속 적용된 프레임) 기준 상태 맵 생성
    # -----------------------------------------
    status_map: Dict[int, Dict[str, object]] = {}

    for _, row in alerts_df.iterrows():
        start_l = int(row.get("start_frame", 0))
        end_l = int(row.get("end_frame", start_l))
        t = row.get("type", "normal")

        if t == "abuse_report":
            code = 2
            label = "ABUSE DETECTED"
        elif t == "suspicious":
            code = 1
            label = "SUSPICIOUS BEHAVIOR"
        else:
            code = 0
            label = "NORMAL"

        for lf in range(start_l, end_l + 1):
            prev = status_map.get(lf)
            # 우선순위: 학대(2) > 의심(1) > 정상(0)
            if prev is None or code > prev["code"]:
                status_map[lf] = {"code": code, "label": label}

    if speed_factor <= 0:
        speed_factor = 1

    # -----------------------------------------
    # 원본 프레임 루프의 클립 영상 박스
    # -----------------------------------------
    for real_f in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        logical_idx = real_f // speed_factor
        info = status_map.get(logical_idx)

        if info is not None and info["code"] > 0:
            # 상단 오버레이 바
            overlay = frame.copy()
            h, w = frame.shape[:2]

            if info["code"] == 2:
                bar_color = (0, 0, 255)  # BGR: 빨간색
            else:
                bar_color = (0, 165, 255)  # 주황색

            cv2.rectangle(overlay, (0, 0), (w, 80), bar_color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # 텍스트
            main_text = info["label"]
            sub_text = (
                "ABUSE DETECTED"
                if info["code"] == 2
                else "SUSPICIOUS BEHAVIORD"
            )

            cv2.putText(
                frame,
                main_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                sub_text,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        out.write(frame)

    cap.release()
    out.release()

    return out_path


# ======================================================================================
#   실시간 스트리밍 파이프라인 (M1 → M2 → M3)
# ======================================================================================

class RealtimeAbuseDetector:
    """
    YOLO Pose 기반 실시간 분석기
    """

    def __init__(
        self, model_path: str, conf: float = 0.5, iou: float = 0.5, vid_stride: int = 1
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}")

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.vid_stride = max(1, int(vid_stride))

        self.fps: float = 0.0
        self.total_frames: int = 0

        self.track_class_counts: Dict[int, Dict[int, int]] = {}

        self.features_df: Optional[pd.DataFrame] = None
        self.alerts_df: Optional[pd.DataFrame] = None

        self.feature_extractor = FeatureExtractor()
        self.abuse_detector = AbuseDetector()

        self._rows: List[Dict] = []

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
        boxes = getattr(result, "boxes", None)
        kpts_obj = getattr(result, "keypoints", None)

        if boxes is None or kpts_obj is None or len(boxes) == 0:
            return

        ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else np.arange(len(boxes.cls))
        )
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        kpts = kpts_obj.data.cpu().numpy()

        num_det = min(len(ids), kpts.shape[0], xyxy.shape[0])

        for i in range(num_det):
            track_id = int(ids[i])
            raw_cls = int(clss[i])
            conf = float(confs[i])

            x1, y1, x2, y2 = xyxy[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            keypoints_flat = kpts[i].reshape(-1).tolist()

            counts = self.track_class_counts.setdefault(track_id, {0: 0, 1: 0})
            counts[raw_cls] = counts.get(raw_cls, 0) + 1
            stable_cls = 1 if counts.get(1, 0) > counts.get(0, 0) else 0

            if track_id == 1:
                stable_cls = 1
            elif track_id == 2:
                stable_cls = 0

            self._rows.append(
                {
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class": stable_cls,
                    "conf": conf,
                    "bbox_x1": float(x1),
                    "bbox_y1": float(y1),
                    "bbox_x2": float(x2),
                    "bbox_y2": float(y2),
                    "bbox_w": w,
                    "bbox_h": h,
                    "center_x": cx,
                    "center_y": cy,
                    "keypoints": keypoints_flat,
                }
            )

    def _run_rules_up_to(
        self, frame_idx: int
    ) -> Tuple[int, str, float, float, float]:
        if not self._rows:
            self.features_df = pd.DataFrame()
            self.alerts_df = pd.DataFrame()
            return 0, "정상", 0.0, float("inf"), 0.0

        preds_df = pd.DataFrame(self._rows)

        features_df = self.feature_extractor.process(preds_df.copy())
        alerts_df = self.abuse_detector.detect(features_df.copy())

        self.features_df = features_df
        self.alerts_df = alerts_df

        status_code = 0
        status_label = "정상"

        if alerts_df is not None and not alerts_df.empty:
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

        max_adult_vel = 0.0
        max_child_vel = 0.0
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

            childs = frame_df[frame_df["class"] == 0]
            if not childs.empty and "limb_velocity" in childs.columns:
                max_child_vel = float(childs["limb_velocity"].max())

        return (
            status_code,
            status_label,
            max_adult_vel,
            min_adult_child_dist,
            max_child_vel,
        )

    def stream_video(
        self,
        video_path: str,
        progress_callback=None,
    ) -> Generator[Dict, None, None]:
        self._init_video_meta(video_path)

        results = self.model.track(
            source=video_path,
            stream=True,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            persist=True,
        )

        SKELETON_CONNECTIONS = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 14),
            (12, 14),
        ]

        logical_frame_idx = 0
        processed_frames = 0
        total_logical_frames = (
            self.total_frames // self.vid_stride
            if self.total_frames > 0
            else max(1, math.ceil(self.total_frames / self.vid_stride))
        )

        for real_frame_idx, result in enumerate(results):
            if real_frame_idx % self.vid_stride != 0:
                continue

            logical_frame_idx = real_frame_idx // self.vid_stride

            processed_frames += 1
            progress = (
                processed_frames / total_logical_frames
                if total_logical_frames > 0
                else 0.0
            )
            progress = max(0.0, min(1.0, float(progress)))

            if progress_callback is not None:
                progress_callback(progress, "실시간 추론 및 규칙 적용 중...")

            self._append_frame_rows(logical_frame_idx, result)

            (
                status_code,
                status_label,
                max_adult_vel,
                min_dist,
                max_child_vel,
            ) = self._run_rules_up_to(logical_frame_idx)

            frame_bgr = result.orig_img
            if frame_bgr is None:
                continue

            annotated = frame_bgr.copy()

            frame_feat = (
                self.features_df[self.features_df["frame"] == logical_frame_idx]
                if self.features_df is not None and not self.features_df.empty
                else pd.DataFrame()
            )
            id_class_mapping: Dict[int, int] = {}
            if not frame_feat.empty:
                for tid in frame_feat["track_id"].unique():
                    sub = frame_feat[frame_feat["track_id"] == tid]
                    if "class" in sub.columns:
                        mode_class = sub["class"].mode()
                        if not mode_class.empty:
                            id_class_mapping[int(tid)] = int(mode_class.iloc[0])

            boxes = getattr(result, "boxes", None)
            kpts_obj = getattr(result, "keypoints", None)

            if boxes is not None and kpts_obj is not None and len(boxes) > 0:
                ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else np.arange(len(boxes.cls))
                )

                if 1 in ids:
                    id_class_mapping[1] = 1
                if 2 in ids:
                    id_class_mapping[2] = 0

                kpts = kpts_obj.data.cpu().numpy()
                num_det = min(len(ids), kpts.shape[0])

                for i in range(num_det):
                    track_id = int(ids[i])
                    if track_id > 2:
                        continue

                    raw_cls = int(boxes.cls[i])
                    class_id = id_class_mapping.get(track_id, raw_cls)

                    if class_id == 1:
                        color = (255, 0, 0)
                        cls_name = "Adult"
                        text_color = (255, 255, 255)
                    else:
                        color = (0, 255, 0)
                        cls_name = "Child"
                        text_color = (0, 0, 0)

                    k = kpts[i][:15]
                    valid_k = k[k[:, 2] > 0.3]
                    if len(valid_k) > 0:
                        x1, y1 = valid_k[:, 0].min(), valid_k[:, 1].min()
                        x2, y2 = valid_k[:, 0].max(), valid_k[:, 1].max()

                        h_img, w_img = annotated.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img, x2), min(h_img, y2)

                        p1 = (int(x1), int(y1))
                        p2 = (int(x2), int(y2))

                        cv2.rectangle(annotated, p1, p2, color, 3)

                        label = f"{cls_name} ID:{track_id}"

                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )

                        bg_x1 = p1[0]
                        bg_y1 = max(p1[1] - label_height - 10, 0)
                        bg_x2 = p1[0] + label_width + 10
                        bg_y2 = bg_y1 + label_height + 10

                        cv2.rectangle(
                            annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1
                        )

                        text_x = bg_x1 + 5
                        text_y = bg_y2 - 5

                        cv2.putText(
                            annotated,
                            label,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2,
                        )

                    for (s_idx, e_idx) in SKELETON_CONNECTIONS:
                        if (
                            s_idx < len(k)
                            and e_idx < len(k)
                            and k[s_idx][2] > 0.3
                            and k[e_idx][2] > 0.3
                        ):
                            p_start = (int(k[s_idx][0]), int(k[s_idx][1]))
                            p_end = (int(k[e_idx][0]), int(k[e_idx][1]))
                            cv2.line(annotated, p_start, p_end, color, 2)

                    for j in range(min(len(k), 15)):
                        if k[j][2] > 0.3:
                            center = (int(k[j][0]), int(k[j][1]))
                            cv2.circle(annotated, center, 6, color, -1)
                            cv2.circle(annotated, center, 6, text_color, 1)

            h_img, w_img = annotated.shape[:2]
            if status_code == 2:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w_img, 100), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

                cv2.putText(
                    annotated,
                    "ABUSE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    "Immediate action required",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
            elif status_code == 1:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w_img, 80), (0, 165, 255), -1)
                cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)

                cv2.putText(
                    annotated,
                    "SUSPICIOUS BEHAVIOR",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    annotated,
                    "Review current interaction carefully",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

            frame_time = logical_frame_idx / self.fps if self.fps > 0 else 0.0

            yield {
                "frame_idx": logical_frame_idx,
                "frame_time": frame_time,
                "annotated_frame": annotated[..., ::-1],
                "status": {"code": status_code, "label": status_label},
                "stats": {
                    "max_adult_velocity": float(max_adult_vel),
                    "max_child_velocity": float(max_child_vel),
                    "min_adult_child_dist": float(min_dist),
                },
            }

        if progress_callback is not None:
            progress_callback(1.0, "실시간 분석 완료")

    def get_final_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.features_df is None:
            self.features_df = pd.DataFrame()
        if self.alerts_df is None:
            self.alerts_df = pd.DataFrame()
        return self.features_df, self.alerts_df


MODEL_PATH = os.path.join(PROJECT_DIR, "models", "detect", "best.pt")

# ======================================================================================
#   TAB 1: 실시간 영상 분석
# ======================================================================================

with tab1:
    st.header("영상 업로드 및 분석")

    st.info(
        "⚠️ 메모리 사용량 안내\n"
        "고해상도 영상 분석 시 메모리 사용량이 많을 수 있습니다.\n"
        "긴 영상의 경우 분석 시간이 오래 걸릴 수 있으니 3분 이내 영상을 권장합니다."
    )

    uploaded_file = st.file_uploader(
        "MP4 영상 파일 업로드 (권장: 3분 이내)", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        temp_dir = os.path.join(os.getcwd(), "temp_videos")
        os.makedirs(temp_dir, exist_ok=True)

        unique_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        video_path = os.path.join(temp_dir, unique_filename)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.video_path = video_path
        st.success("영상이 업로드되었습니다. 아래에서 분석을 시작할 수 있습니다.")

    if st.session_state.video_path:
        st.video(st.session_state.video_path)

    if st.session_state.video_path:
        st.markdown("---")

        # ✅ 고속 모드 체크박스 제거, 배속 선택만 유지
        speed_factor = st.select_slider(
            "분석 배속 선택",
            options=[1, 8, 16],
            value=8,
            help="배속이 높을수록 프레임을 더 많이 건너뛰면서 빠르게 분석합니다.",
        )

        st.session_state.speed_factor = int(speed_factor)
        vid_stride = int(speed_factor)

        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            start_realtime = st.button(
                "실시간 분석 시작",
                key="realtime_start",
                use_container_width=True,  
            )

        if start_realtime:
            # progress_slot 변수 제거 → NameError 방지
            progress_bar = st.progress(0.0)
            status_slot = st.empty()

            col_live_frame, col_live_graph = st.columns([3, 2])
            with col_live_frame:
                frame_slot = st.empty()
                log_slot = st.empty()
            with col_live_graph:
                graph_slot = st.empty()

            timeline_data: List[Dict] = []

            def live_progress(p: float, msg: str = ""):
                p_clamped = max(0.0, min(1.0, float(p)))
                progress_bar.progress(
                    p_clamped, text=f"{msg} ({p_clamped*100:.1f}%)"
                )

            try:
                rt_detector = RealtimeAbuseDetector(
                    model_path=MODEL_PATH,
                    conf=0.5,
                    iou=0.5,
                    vid_stride=vid_stride,
                )

                step_stream = rt_detector.stream_video(
                    st.session_state.video_path,
                    progress_callback=live_progress,
                )

                for step in step_stream:
                    frame = step["annotated_frame"]
                    status = step["status"]
                    stats = step["stats"]

                    frame_slot.image(
                        frame,
                        channels="RGB",
                    )

                    timeline_data.append(
                        {
                            "time": step["frame_time"],
                            "status_code": status["code"],
                            "status_label": status["label"],
                            "max_adult_velocity": stats["max_adult_velocity"],
                            "max_child_velocity": stats["max_child_velocity"],
                            "min_adult_child_dist": stats["min_adult_child_dist"],
                        }
                    )

                    if status["code"] == 2:
                        log_text = (
                            f"🚨 [학대 신고 레벨] {step['frame_time']:.2f}s - "
                            f"성인/아동 상호작용에서 고위험 행동 감지"
                        )
                    elif status["code"] == 1:
                        log_text = (
                            f"⚠️ [의심 행동] {step['frame_time']:.2f}s - "
                            f"이상한 상호작용 패턴 감지"
                        )
                    else:
                        log_text = (
                            f"✅ [정상] {step['frame_time']:.2f}s - "
                            f"위험 행동 없음"
                        )

                    log_slot.markdown(log_text)

                    if len(timeline_data) > 1:
                        live_df = pd.DataFrame(timeline_data)

                        fig_live = make_subplots(
                            rows=2,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            row_heights=[0.4, 0.6],
                        )

                        fig_live.add_trace(
                            go.Scatter(
                                x=live_df["time"],
                                y=[
                                    0 if c == 0 else 1 if c == 1 else 2
                                    for c in live_df["status_code"]
                                ],
                                mode="lines",
                                line=dict(
                                    color="white",
                                    width=2,
                                ),
                                name="상태 코드 (0=정상,1=의심,2=학대)",
                            ),
                            row=1,
                            col=1,
                        )

                        abuse_mask = live_df["status_code"] == 2
                        if abuse_mask.any():
                            fig_live.add_trace(
                                go.Scatter(
                                    x=live_df.loc[abuse_mask, "time"],
                                    y=[2] * abuse_mask.sum(),
                                    mode="lines",
                                    line=dict(
                                        color="red",
                                        width=4,
                                    ),
                                    name="학대 감지",
                                ),
                                row=1,
                                col=1,
                            )

                        susp_mask = live_df["status_code"] == 1
                        if susp_mask.any():
                            fig_live.add_trace(
                                go.Scatter(
                                    x=live_df.loc[susp_mask, "time"],
                                    y=[1] * susp_mask.sum(),
                                    mode="lines",
                                    line=dict(
                                        color="orange",
                                        width=3,
                                    ),
                                    name="의심 감지",
                                ),
                                row=1,
                                col=1,
                            )

                        fig_live.add_trace(
                            go.Scatter(
                                x=live_df["time"],
                                y=live_df["max_adult_velocity"],
                                mode="lines",
                                name="성인 움직임",
                                line=dict(
                                    color="blue",
                                    width=2,
                                ),
                            ),
                            row=2,
                            col=1,
                        )

                        fig_live.add_trace(
                            go.Scatter(
                                x=live_df["time"],
                                y=live_df["max_child_velocity"],
                                mode="lines",
                                name="아동 움직임",
                                line=dict(
                                    color="green",
                                    width=2,
                                ),
                            ),
                            row=2,
                            col=1,
                        )

                        fig_live.add_trace(
                            go.Scatter(
                                x=live_df["time"],
                                y=live_df["min_adult_child_dist"],
                                mode="lines",
                                name="성인-아동 거리",
                                line=dict(
                                    color="white",
                                    width=1,
                                ),
                            ),
                            row=2,
                            col=1,
                        )

                        fig_live.update_xaxes(title_text="시간 (초)")

                        fig_live.update_layout(
                            height=500,
                            margin=dict(l=40, r=20, t=40, b=40),
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                font=dict(size=10),
                            ),
                        )

                        graph_slot.plotly_chart(fig_live, width='stretch')

                features_df, alerts_df = rt_detector.get_final_results()

                st.session_state.processed = True
                st.session_state.analysis_complete = True
                st.session_state.preds_df = features_df
                st.session_state.alerts_df = alerts_df

                st.session_state.video_fps = rt_detector.fps
                st.session_state.video_duration = (
                    rt_detector.total_frames / rt_detector.fps
                    if rt_detector.fps > 0
                    else 0.0
                )

                st.session_state.abuse_clips = create_abuse_clips(
                    st.session_state.video_path,
                    alerts_df,
                    rt_detector.fps if rt_detector.fps > 0 else 30.0,
                )

                # 전체 분석 완료 원본 영상 생성
                st.session_state.review_video_path = create_review_video(
                    st.session_state.video_path,
                    alerts_df,
                    rt_detector.fps if rt_detector.fps > 0 else 30.0,
                    st.session_state.speed_factor,
                )

                st.success(
                    "실시간 분석이 완료되었습니다. '결과 및 보고서' 탭에서 상세 결과를 확인하세요."
                )

            except Exception as e:
                st.error(f"실시간 분석 중 오류 발생: {e}")

# ======================================================================================
#   TAB 2: 결과 및 보고서
# ======================================================================================

with tab2:
    st.header("분석 결과 및 보고서")

    if st.session_state.processed and st.session_state.analysis_complete:
        alerts_df = st.session_state.alerts_df
        features_df = st.session_state.preds_df

        if alerts_df is not None and not alerts_df.empty:
            fps = st.session_state.video_fps or 30.0

            col1, col2 = st.columns([2, 3])

            # 1) 상태 요약
            with col1:
                st.subheader("상태 요약")

                status_summary = compute_status_summary(alerts_df)

                if not status_summary.empty:
                    fig = go.Figure()

                    colors = {0: "#10b981", 1: "#f97316", 2: "#ef4444"}
                    labels = {0: "정상", 1: "의심", 2: "학대 신고"}

                    for code in status_summary["status_code"].unique():
                        sub = status_summary[status_summary["status_code"] == code]
                        fig.add_trace(
                            go.Bar(
                                x=sub["type"],
                                y=sub["총 지속 시간(초)"],
                                name=labels.get(code, "알 수 없음"),
                                marker=dict(color=colors.get(code, "#6b7280")),
                            )
                        )

                    fig.update_layout(
                        height=260,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=10),
                        ),
                        xaxis=dict(title="상태"),
                        yaxis=dict(title="총 지속 시간(프레임 단위)"),
                        plot_bgcolor="#020617",
                        paper_bgcolor="#020617",
                        font=dict(color="#e5e7eb"),
                    )

                    st.plotly_chart(fig, width='stretch')

                abuse_report_count = (alerts_df["type"] == "abuse_report").sum()
                suspicious_count = (alerts_df["type"] == "suspicious").sum()

                if abuse_report_count > 0:
                    st.markdown(
                        f"🚨 **긴급: {abuse_report_count}건의 학대 신고 알람이 감지되었습니다!**  \n"
                        f"**즉각적인 전문가 모니터링이 필요합니다.**  \n"
                        f"**관할 기관에 신고가 필요합니다.**"
                    )

                if suspicious_count > 0:
                    st.markdown(
                        f"⚠️ **주의: {suspicious_count}건의 의심 행동 구간이 감지되었습니다.**  \n"
                        f"**해당 구간을 우선적으로 검토하는 것을 권장합니다.**"
                    )

            # 2) 상태 타임라인
            with col2:
                st.subheader("상태 타임라인")

                status_timeline_df = build_status_timeline_from_alerts(alerts_df, fps)

                if not status_timeline_df.empty:
                    max_time = float(status_timeline_df["time"].max())
                    default_time = st.session_state.get("timeline_selected_time", 0.0)
                    default_time = max(0.0, min(default_time, max_time))

                    selected_time = st.slider(
                        "타임라인 위치 선택 (초)",
                        min_value=0.0,
                        max_value=max_time,
                        value=default_time,
                        step=max_time / 200 if max_time > 0 else 0.1,
                    )
                    st.session_state.timeline_selected_time = selected_time

                    fig_timeline = go.Figure()

                    fig_timeline.add_trace(
                        go.Scatter(
                            x=status_timeline_df["time"],
                            y=[
                                0 if c == 0 else 1 if c == 1 else 2
                                for c in status_timeline_df["status_code"]
                            ],
                            mode="lines",
                            line=dict(color="white", width=2),
                            name="상태 (0=정상,1=의심,2=학대)",
                        )
                    )

                    abuse_mask = status_timeline_df["status_code"] == 2
                    if abuse_mask.any():
                        fig_timeline.add_trace(
                            go.Scatter(
                                x=status_timeline_df.loc[abuse_mask, "time"],
                                y=[2] * abuse_mask.sum(),
                                mode="lines",
                                line=dict(color="red", width=4),
                                name="학대 감지",
                            )
                        )

                    susp_mask = status_timeline_df["status_code"] == 1
                    if susp_mask.any():
                        fig_timeline.add_trace(
                            go.Scatter(
                                x=status_timeline_df.loc[susp_mask, "time"],
                                y=[1] * susp_mask.sum(),
                                mode="lines",
                                line=dict(color="orange", width=3),
                                name="의심 감지",
                            )
                        )

                    fig_timeline.add_vline(
                        x=selected_time,
                        line=dict(color="white", width=2, dash="dash"),
                        annotation_text="현재 위치",
                        annotation_position="top",
                    )

                    fig_timeline.update_yaxes(
                        tickmode="array",
                        tickvals=[0, 1, 2],
                        ticktext=["정상", "의심", "학대"],
                    )

                    fig_timeline.update_xaxes(title_text="시간 (초)")
                    fig_timeline.update_layout(
                        height=260,
                        margin=dict(l=10, r=10, t=10, b=10),
                        plot_bgcolor="#020617",
                        paper_bgcolor="#020617",
                        font=dict(color="#e5e7eb"),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=10),
                        ),
                    )

                    st.plotly_chart(fig_timeline, width='stretch')
                else:
                    st.info("상태 타임라인을 생성할 수 있는 데이터가 부족합니다.")

        st.markdown("---")
        st.subheader("분석 통합 그래프")

        if (
            st.session_state.alerts_df is not None
            and not st.session_state.alerts_df.empty
            and st.session_state.preds_df is not None
            and not st.session_state.preds_df.empty
        ):
            fps = st.session_state.video_fps or 30.0
            merged_df = build_status_velocity_df(
                st.session_state.alerts_df,
                st.session_state.preds_df,
                fps,
            )

            if not merged_df.empty:
                max_time = float(merged_df["time"].max())
                current_time = st.session_state.get(
                    "timeline_selected_time",
                    max_time / 2 if max_time > 0 else 0.0,
                )
                current_time = max(0.0, min(current_time, max_time))

                fig_state_vel = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.4, 0.6],
                )

                fig_state_vel.add_trace(
                    go.Scatter(
                        x=merged_df["time"],
                        y=[
                            0 if c == 0 else 1 if c == 1 else 2
                            for c in merged_df["status_code"]
                        ],
                        mode="lines",
                        line=dict(color="white", width=2),
                        name="상태 (0=정상,1=의심,2=학대)",
                    ),
                    row=1,
                    col=1,
                )

                abuse_mask2 = merged_df["status_code"] == 2
                if abuse_mask2.any():
                    fig_state_vel.add_trace(
                        go.Scatter(
                            x=merged_df.loc[abuse_mask2, "time"],
                            y=[2] * abuse_mask2.sum(),
                            mode="lines",
                            line=dict(color="red", width=4),
                            name="학대 감지",
                        ),
                        row=1,
                        col=1,
                    )

                susp_mask2 = merged_df["status_code"] == 1
                if susp_mask2.any():
                    fig_state_vel.add_trace(
                        go.Scatter(
                            x=merged_df.loc[susp_mask2, "time"],
                            y=[1] * susp_mask2.sum(),
                            mode="lines",
                            line=dict(color="orange", width=3),
                            name="의심 감지",
                        ),
                        row=1,
                        col=1,
                    )

                # 현재 위치
                fig_state_vel.add_vline(
                    x=current_time,
                    line=dict(color="white", width=2, dash="dash"),
                    row=1,
                    col=1,
                )
                fig_state_vel.add_vline(
                    x=current_time,
                    line=dict(color="white", width=1, dash="dash"),
                    row=2,
                    col=1,
                )

                fig_state_vel.add_trace(
                    go.Scatter(
                        x=merged_df["time"],
                        y=merged_df["max_adult_velocity"],
                        mode="lines",
                        name="성인 움직임",
                        line=dict(color="blue", width=2),
                    ),
                    row=2,
                    col=1,
                )

                fig_state_vel.add_trace(
                    go.Scatter(
                        x=merged_df["time"],
                        y=merged_df["max_child_velocity"],
                        mode="lines",
                        name="아동 움직임",
                        line=dict(color="green", width=2),
                    ),
                    row=2,
                    col=1,
                )

                fig_state_vel.add_trace(
                    go.Scatter(
                        x=merged_df["time"],
                        y=merged_df["min_adult_child_dist"],
                        mode="lines",
                        name="성인-아동 거리",
                        line=dict(color="white", width=1),
                    ),
                    row=2,
                    col=1,
                )

                fig_state_vel.update_yaxes(
                    tickmode="array",
                    tickvals=[0, 1, 2],
                    ticktext=["정상", "의심", "학대"],
                    row=1,
                    col=1,
                )
                fig_state_vel.update_xaxes(title_text="시간 (초)", row=2, col=1)

                fig_state_vel.update_layout(
                    height=500,
                    margin=dict(l=40, r=20, t=40, b=40),
                    plot_bgcolor="#020617",
                    paper_bgcolor="#020617",
                    font=dict(color="#e5e7eb"),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=10),
                    ),
                )

                st.plotly_chart(fig_state_vel, width='stretch')
            else:
                st.info("상태 + 속도 통합 그래프를 생성할 수 있는 데이터가 부족합니다.")
        else:
            st.info("그래프를 그릴 수 있는 분석 데이터가 부족합니다.")

            
        # 전체 분석 완료 영상 다운로드
        st.markdown("---")
        st.subheader("전체 분석된 영상 다운로드")

        review_path = getattr(st.session_state, "review_video_path", None)
        if review_path and os.path.exists(review_path):
            with open(review_path, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                label="전체 분석 완료 영상 다운로드",
                data=video_bytes,
                file_name=os.path.basename(review_path),
                mime="video/mp4",
            )
        else:
            st.info("분석된 전체 영상을 찾을 수 없습니다. 먼저 실시간 분석을 실행해 주세요.")

        # 학대 감지 구간 클립 + 개별 다운로드
        st.markdown("---")
        st.subheader("학대 감지 구간 클립 다운로드")

        if st.session_state.abuse_clips:
            clip_labels = [c["label"] for c in st.session_state.abuse_clips]
            selected_label = st.selectbox(
                "다운로드할 학대 감지 클립 선택",
                options=clip_labels,
            )

            selected_clip = next(
                (c for c in st.session_state.abuse_clips if c["label"] == selected_label),
                None,
            )

            if selected_clip and os.path.exists(selected_clip["path"]):
                with open(selected_clip["path"], "rb") as f:
                    clip_bytes = f.read()

                st.download_button(
                    label="선택한 클립 다운로드",
                    data=clip_bytes,
                    file_name=os.path.basename(selected_clip["path"]),
                    mime="video/mp4",
                )
            else:
                st.info("선택한 클립 파일을 찾을 수 없습니다. 다시 분석을 실행해 주세요.")
        else:
            st.info("학대 감지 구간이 없어 생성된 클립이 없습니다.")

    elif st.session_state.processed and not st.session_state.analysis_complete:
        st.info("분석이 진행 중이거나 결과를 불러오는 중입니다.")
        st.button("결과 새로고침", on_click=lambda: st.experimental_rerun())
    else:
        st.info("영상을 업로드하고 분석을 시작하세요.")
