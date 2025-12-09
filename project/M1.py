import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import sys

class ModelInference:
    def __init__(self, model_path):
        # 모델 로드 (track 기능 포함)
        self.model = YOLO(model_path)
        
        # 15개 키포인트 인덱스
        self.TORSO_INDICES = [5, 6, 11, 12]
        self.HEAD_INDICES = [0, 1, 2, 3, 4]
        self.WRIST_INDICES = [9, 10]
        self.KNEE_INDICES = [13, 14]
        
        # 진행률 콜백
        self.progress_callback = None
        
        # ID별 클래스 카운트 
        self.id_class_counts = defaultdict(lambda: defaultdict(int))

    def _update_progress(self, current, total):
        """진행률 콜백 호출"""
        if self.progress_callback and total > 0:
            progress = current / total
            self.progress_callback(progress)

    def classify_by_height(self, track_heights):
        """
        ID별 평균 키(박스 높이)로 child/adult를 분류.
        - 가장 큰 ID를 기준으로 adult,
        - 상대적 비율로 child 여부 판단.
        """
        if not track_heights:
            return {}
        
        avg_heights = {tid: np.mean(hs) for tid, hs in track_heights.items()}
        if not avg_heights:
            return {}
        
        # 가장 키가 큰 사람을 adult 기준으로
        max_tid = max(avg_heights, key=avg_heights.get)
        max_h = avg_heights[max_tid]

        classifications = {}
        for tid, h in avg_heights.items():
            # 상대 키 비율
            ratio = h / max_h if max_h > 0 else 1.0
            # 0.8 이상이면 adult, 그 외는 child
            if ratio >= 0.8:
                classifications[tid] = 'adult'
            else:
                classifications[tid] = 'child'
        return classifications

    def run_inference(self, video_path, progress_callback=None):
        """비디오 추론 및 데이터 추출 - ID 기반 고정 분류"""
        if not os.path.exists(video_path):
            raise ValueError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

        # 비디오 정보 얻기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"비디오 프레임 수를 읽을 수 없습니다: {video_path}")
            
        cap.release()

        # 진행률 콜백 설정
        if progress_callback:
            self.progress_callback = progress_callback
        
        # YOLO Tracking 실행 
        try:
            results = self.model.track(
                source=video_path, 
                stream=True, 
                persist=True, 
                verbose=False,
                conf=0.5,
                iou=0.5
            )
        except Exception as e:
            raise RuntimeError(f"YOLO 추론 중 오류: {e}")

        raw_data = []
        frame_idx = 0
        last_reported_percent = -1

        # ID별 키(높이) 기록
        track_heights = defaultdict(list)

        # 전체 프레임 순회하며 ID별 키(박스 높이) 수집
        print("=== STEP 1: ID별 키(박스 높이) 수집 ===")
        for result in results:
            if result is None or result.boxes is None or len(result.boxes) == 0:
                frame_idx += 1
                self._update_progress(frame_idx, total_frames)
                continue

            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            
            # pose 키포인트
            if result.keypoints is None:
                frame_idx += 1
                self._update_progress(frame_idx, total_frames)
                continue
            kpts = result.keypoints.data.cpu().numpy()  # (N, 15, 3)

            if ids is None:
                frame_idx += 1
                self._update_progress(frame_idx, total_frames)
                continue

            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = xyxy[i]
                h = float(y2 - y1)
                if h > 0:
                    track_heights[int(tid)].append(h)

            frame_idx += 1
            self._update_progress(frame_idx, total_frames)

        # ID별 child/adult 고정 분류 (기본: 키 비율 기반)
        classifications = self.classify_by_height(track_heights)

        # 데이터셋 규칙
        #   - ID 1  → ADULT
        #   - ID 2  → CHILD
        if 1 in track_heights:
            classifications[1] = 'adult'
        if 2 in track_heights:
            classifications[2] = 'child'

        # 2차 패스: 다시 Tracking 수행하여 최종 raw_data 생성
        try:
            results = self.model.track(
                source=video_path, 
                stream=True, 
                persist=True, 
                verbose=False,
                conf=0.5,
                iou=0.5
            )
        except Exception as e:
            raise RuntimeError(f"[2차 패스] YOLO 추론 중 오류: {e}")

        print("\n=== STEP 2: 최종 raw_data 생성 ===")
        frame_idx = 0
        last_reported_percent = -1

        for result in results:
            if result is None or result.boxes is None or len(result.boxes) == 0:
                frame_idx += 1
                self._update_progress(frame_idx, total_frames)
                continue

            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            
            if result.keypoints is None:
                frame_idx += 1
                self._update_progress(frame_idx, total_frames)
                continue
            kpts = result.keypoints.data.cpu().numpy()  # (N, 15, 3)

            if ids is None:
                frame_idx += 1
                self._update_progress(frame_idx, total_frames)
                continue

            frame_objects = []
            for i in range(len(ids)):
                tid = int(ids[i])
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])
                cls = int(clss[i])
                keypoints = kpts[i]  # (15, 3)

                obj = {
                    'frame': frame_idx,
                    'track_id': tid,
                    'cls': cls,
                    'conf': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'kpts': keypoints
                }
                frame_objects.append(obj)

            # ID별 global child/adult 라벨 적용
            for obj in frame_objects:
                tid = obj['track_id']
                if tid in classifications:
                    if classifications[tid] == 'child':
                        obj['cls'] = 0   # CHILD
                    else:
                        obj['cls'] = 1   # ADULT

            for obj in frame_objects:
                if obj['track_id'] <= 2:
                    raw_data.append({
                        'frame': frame_idx,
                        'track_id': obj['track_id'],
                        'class': obj['cls'],
                        'conf': obj['conf'],
                        'keypoints': obj['kpts'].flatten().tolist()
                    })

            frame_idx += 1
            self._update_progress(frame_idx, total_frames)

        print("=== STEP 3: DataFrame 생성 ===")
        df = pd.DataFrame(raw_data)
        return df
