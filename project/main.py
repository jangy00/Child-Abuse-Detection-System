import os
import cv2
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

try:
    from project.M1 import ModelInference
    from project.M2 import FeatureExtractor
    from project.M3 import AbuseDetector
except ImportError:
    from M1 import ModelInference
    from M2 import FeatureExtractor
    from M3 import AbuseDetector

def process_video(video_path, model_path, output_dir, progress_callback=None):
    """전체 파이프라인 실행 함수 - 진행률 콜백 지원"""
    
    # 1. 초기화
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 경로 절대 경로 변환
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
        
    m1 = ModelInference(model_path)
    m2 = FeatureExtractor()
    m3 = AbuseDetector()
    
    # Step 1: 추론 및 트래킹
    if progress_callback:
        progress_callback(0, "Step 1: 추론 및 트래킹 준비중...")
    
    print("\n" + "="*50)
    print("Step 1: 추론 및 트래킹 중...")
    print("="*50)
    
    start_time = time.time()
    
    # 진행률 콜백 설정
    if progress_callback:
        def step1_progress(progress, status):
            # Step 1의 진행률을 0-70%로 매핑
            mapped_progress = progress * 0.7
            progress_callback(mapped_progress, status)
        m1.set_progress_callback(step1_progress)
    
    preds_df = m1.run_inference(video_path)
    inference_time = time.time() - start_time
    print(f"\n✅ 추론 완료: {len(preds_df)}개의 감지 결과 ({inference_time:.2f}초 소요)")
    
    if preds_df.empty:
        print("경고: 추론 결과가 없습니다.")
        return None, None

    # Step 2: 피처 추출
    if progress_callback:
        progress_callback(0.7, "Step 2: 피처 추출 중...")
    
    print("\n" + "="*50)
    print("Step 2: 피처 추출 중...")
    print("="*50)
    
    start_time = time.time()
    features_df = m2.process(preds_df)
    feature_time = time.time() - start_time
    print(f"\n✅ 피처 추출 완료 ({feature_time:.2f}초 소요)")
    
    # Step 3: 규칙 기반 탐지
    if progress_callback:
        progress_callback(0.85, "Step 3: 규칙 기반 탐지 중...")
    
    print("\n" + "="*50)
    print("Step 3: 규칙 기반 탐지 중...")
    print("="*50)
    
    start_time = time.time()
    alerts_df = m3.detect(features_df)
    detection_time = time.time() - start_time
    
    # 탐지 결과 요약
    if alerts_df is not None and not alerts_df.empty:
        suspicious_count = len(alerts_df[alerts_df['type'] == 'suspicious'])
        abuse_count = len(alerts_df[alerts_df['type'] == 'abuse_report'])
        print(f"\n✅ 규칙 기반 탐지 완료 ({detection_time:.2f}초 소요)")
        print(f"   • 의심 행동: {suspicious_count}건")
        print(f"   • 학대 신고 알람: {abuse_count}건")
    else:
        print(f"\n✅ 규칙 기반 탐지 완료 ({detection_time:.2f}초 소요)")
        print("   • 감지된 알림이 없습니다.")
    
    # 결과 저장
    preds_path = os.path.join(output_dir, 'preds.csv')
    alerts_path = os.path.join(output_dir, 'alerts.csv')
    
    features_df.to_csv(preds_path, index=False)
    alerts_df.to_csv(alerts_path, index=False)

    print(f"\n📁 결과 저장 완료: {preds_path}, {alerts_path}")
    
    # 최종 완료
    if progress_callback:
        progress_callback(1.0, "분석 완료!")
    
    return features_df, alerts_df

def create_annotated_video(video_path, features_df, alerts_df, output_path, progress_callback=None):
    """결과 시각화 비디오 생성 - 메모리 최적화 및 영상 품질 개선"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("오류: 비디오를 열 수 없습니다.")
        return None

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 비디오 정보 확인
    if total_frames <= 0:
        print("오류: 비디오 프레임 수를 읽을 수 없습니다.")
        cap.release()
        return None
    
    # 해상도 유지 (품질 저하 방지)
    width = original_width
    height = original_height
    
    # 비디오 코덱 설정 (호환성 개선)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v 코덱 사용
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"오류: 비디오 라이터를 열 수 없습니다. 코덱: mp4v, 해상도: {width}x{height}, FPS: {fps}")
        cap.release()
        return None
    
    # 15개 키포인트 연결선 정의
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),          # 얼굴
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # 팔
        (5, 11), (6, 12), (11, 12),               # 몸통
        (11, 13), (13, 14), (12, 14)              # 다리
    ]
    
    # ID-클래스 매핑 생성 (track_id -> class)
    print("\n🔍 ID-클래스 매핑 확인 중...")
    id_class_mapping = {}
    if not features_df.empty:
        for track_id in features_df['track_id'].unique():
            track_data = features_df[features_df['track_id'] == track_id]
            if not track_data.empty:
                # 가장 많이 나타나는 클래스 사용
                mode_class = track_data['class'].mode()
                if not mode_class.empty:
                    id_class_mapping[int(track_id)] = int(mode_class.iloc[0])
                    class_name = "성인" if mode_class.iloc[0] == 1 else "아동"
                    print(f"   ID {track_id} -> 클래스 {mode_class.iloc[0]} ({class_name})")
    
    print("\n" + "="*50)
    print("🎬 주석 비디오 생성 중...")
    print("="*50)
    
    frame_idx = 0
    
    # ✅ tqdm 제거: 직접 진행률 표시
    print(f"📊 총 프레임 수: {total_frames}")
    last_reported_percent = -1
    
    # 프레임별 색상 캐시
    frame_cache = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            # 현재 프레임 데이터
            f_data = features_df[features_df['frame'] == frame_idx]
            
            # 현재 프레임이 Alert 구간인지 확인
            is_alert = False
            alert_type = ""
            alert_details = ""
            confidence_percent = 0
            
            if alerts_df is not None and not alerts_df.empty:
                active_alerts = alerts_df[(alerts_df['start_frame'] <= frame_idx) & 
                                          (alerts_df['end_frame'] >= frame_idx)]
                if not active_alerts.empty:
                    is_alert = True
                    # 가장 높은 신뢰도의 경고 정보 사용
                    highest_alert = active_alerts.loc[active_alerts['confidence'].idxmax()]
                    alert_type = highest_alert['type']
                    confidence_percent = int(highest_alert['confidence'] * 100)

            # 시각화: 경고 메시지
            if is_alert:
                if alert_type == 'abuse_report':
                    # 학대 신고 알람 - 빨간색
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    # 수정: 'CHILD ABUSE REPORT' 텍스트를 단순한 경고로 변경
                    alert_text = "ABUSE DETECTED"
                    alert_details = f"Immediate action required - Confidence: {confidence_percent}%"
                    
                    # 주 경고 문구 (물음표 없이)
                    cv2.putText(frame, alert_text, 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                    # 상세 정보
                    cv2.putText(frame, alert_details, 
                               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 경고 아이콘 추가 (선택사항)
                    # cv2.putText(frame, "WARNING", (width - 150, 50), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                else:
                    # 의심 행동 - 주황색
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 165, 255), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    alert_text = "Suspicious Behavior"
                    alert_details = f"Suspicion Level: {confidence_percent}%"
                    
                    # 주 경고 문구
                    cv2.putText(frame, alert_text, 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                    # 상세 정보
                    cv2.putText(frame, alert_details, 
                               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 시각화: 스켈레톤 및 박스
            for _, row in f_data.iterrows():
                kpts = row['keypoints']
                if isinstance(kpts, str):
                    kpts = eval(kpts)
                
                # 키포인트 개수에 따라 reshape (15개로 고정)
                num_kpts = 15
                kpts = np.array(kpts).reshape(num_kpts, 3)
                
                track_id = int(row['track_id'])
                class_id = int(row['class'])
                
                if track_id > 2:  
                    continue
                
                # ID에 따른 고정 색상 및 클래스 이름
                if class_id == 1:  
                    color = (255, 0, 0)      # 빨간색
                    cls_name = "Adult"
                    text_color = (255, 255, 255)
                else:  
                    color = (0, 255, 0)      # 초록색
                    cls_name = "Child"
                    text_color = (0, 0, 0)
                
                # BBox 그리기 (신뢰도 0.3 이상인 키포인트 기준)
                valid_kpts = kpts[kpts[:, 2] > 0.3]
                if len(valid_kpts) > 0:
                    x1, y1 = valid_kpts[:, 0].min(), valid_kpts[:, 1].min()
                    x2, y2 = valid_kpts[:, 0].max(), valid_kpts[:, 1].max()
                    
                    # 화면 밖으로 나가는 것 방지
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    # BBox 그리기 (둥근 모서리)
                    thickness = 3
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # 라벨 배경
                    label = f"{cls_name} ID:{track_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    label_bg_end = (int(x1) + label_size[0] + 10, int(y1))
                    label_bg_start = (int(x1), int(y1) - label_size[1] - 10)
                    
                    # 배경이 화면 위쪽을 넘지 않도록 조정
                    if label_bg_start[1] < 0:
                        label_bg_start = (int(x1), 0)
                        label_bg_end = (int(x1) + label_size[0] + 10, label_size[1] + 5)
                    
                    cv2.rectangle(frame, label_bg_start, label_bg_end, color, -1)
                    
                    # 라벨 텍스트
                    text_pos = (int(x1) + 5, int(y1) - 5 if y1 - 5 > 0 else label_size[1])
                    cv2.putText(frame, label, text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                # 스켈레톤 키포인트 시각화
                # 연결선 그리기
                for connection in SKELETON_CONNECTIONS:
                    start_idx, end_idx = connection
                    if (start_idx < len(kpts) and end_idx < len(kpts) and 
                        kpts[start_idx][2] > 0.3 and kpts[end_idx][2] > 0.3):
                        
                        start_point = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                        end_point = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                        
                        # 연결선 그리기
                        cv2.line(frame, start_point, end_point, color, 3)
                
                # 키포인트 점 그리기
                for i in range(min(len(kpts), 15)):
                    if kpts[i][2] > 0.3:
                        center = (int(kpts[i][0]), int(kpts[i][1]))
                        # 키포인트 점
                        cv2.circle(frame, center, 6, color, -1)
                        # 키포인트 점 테두리
                        cv2.circle(frame, center, 6, text_color, 1)
            
            # 프레임 캐시에 저장 (메모리 최적화)
            frame_cache[frame_idx] = frame.copy()
            
            # 주기적으로 프레임 쓰기 (메모리 관리)
            if frame_idx % 10 == 0:
                for cached_idx in sorted(frame_cache.keys()):
                    out.write(frame_cache[cached_idx])
                frame_cache.clear()
            
            frame_idx += 1
            
            # 진행률 표시 (5% 단위)
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                progress = frame_idx / total_frames
                percent = int(progress * 100)
                if percent != last_reported_percent and percent % 5 == 0:
                    print(f"   진행률: {percent}% ({frame_idx}/{total_frames})")
                    last_reported_percent = percent
                
                # 진행률 콜백 호출
                if progress_callback:
                    progress_callback(progress, f"비디오 생성 중... ({frame_idx}/{total_frames})")
            
            # 메모리 관리: 매 100프레임마다 가비지 컬렉션
            if frame_idx % 100 == 0:
                import gc
                gc.collect()
    
    except Exception as e:
        print(f"비디오 생성 중 오류 발생: {e}")
    
    finally:
        # 남은 프레임 쓰기
        for cached_idx in sorted(frame_cache.keys()):
            out.write(frame_cache[cached_idx])
        
        cap.release()
        out.release()
    
    print("   진행률: 100% (완료)")
    
    # 비디오 파일 확인
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB 단위
        print(f"\n✅ 주석 비디오 생성 완료: {output_path}")
        print(f"   파일 크기: {file_size:.1f} MB")
        print(f"   해상도: {width}x{height}, FPS: {fps}")
    else:
        print(f"\n❌ 비디오 생성 실패: {output_path}")
    
    if progress_callback:
        progress_callback(1.0, "비디오 생성 완료!")
    
    return output_path

def create_alert_clips(video_path, alerts_df, output_dir):
    """학대 신고 알람만 따로 클립으로 생성 (1.5~3초)"""
    if alerts_df is None or alerts_df.empty:
        print("\n⚠️ 생성할 클립이 없습니다.")
        return []
    
    # 학대 신고 알람만 필터링
    abuse_reports = alerts_df[alerts_df['type'] == 'abuse_report']
    if abuse_reports.empty:
        print("\n⚠️ 학대 신고 알람 클립이 없습니다.")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("오류: 비디오를 열 수 없습니다.")
        return []
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 1.5~3초 내외 클립 생성 (fps 기준)
    min_clip_frames = int(fps * 1.5)  # 최소 1.5초
    max_clip_frames = int(fps * 3)    # 최대 3초
    
    # 해상도 유지 (품질 저하 방지)
    width = original_width
    height = original_height
    
    clip_paths = []
    
    print("\n" + "="*50)
    print("🚨 학대 신고 알람 클립 생성 중...")
    print("="*50)
    
    for i, alert in abuse_reports.iterrows():
        start_frame = int(alert['start_frame'])
        confidence = alert['confidence']
        
        # 클립 길이 결정 (1.5~3초)
        clip_duration_frames = min(
            max(min_clip_frames, int(fps * 2.0)),  # 기본 2.0초
            max_clip_frames
        )
        end_frame = start_frame + clip_duration_frames
        
        # 클립 파일명 생성
        clip_filename = f"abuse_report_{i+1}_conf_{confidence:.2f}.mp4"
        clip_path = os.path.join(output_dir, clip_filename)
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"오류: 클립 라이터를 열 수 없습니다: {clip_path}")
            continue
        
        # 해당 프레임 범위로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames_to_capture = clip_duration_frames
        
        for frame_idx in range(frames_to_capture):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 빨간색 알림 배경 (투명도 적용)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # 영어 알림 문구
            cv2.putText(frame, "CHILD ABUSE REPORT", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "IMMEDIATE ACTION REQUIRED", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.1%}", 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 경고 아이콘
            cv2.putText(frame, "⚠️", (width - 60, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            out.write(frame)
        
        out.release()
        
        if os.path.exists(clip_path):
            file_size = os.path.getsize(clip_path) / (1024 * 1024)
            clip_paths.append(clip_path)
            print(f"✅ 학대 신고 클립 {i+1} 생성 완료: {clip_filename} ({clip_duration_frames/fps:.1f}초, {file_size:.1f}MB)")
        else:
            print(f"❌ 클립 생성 실패: {clip_filename}")
        
        # 메모리 관리
        import gc
        gc.collect()
    
    cap.release()
    
    if clip_paths:
        print(f"\n🚨 총 {len(clip_paths)}개의 학대 신고 클립 생성됨")
    else:
        print("\n⚠️ 생성된 클립이 없습니다.")
    
    return clip_paths