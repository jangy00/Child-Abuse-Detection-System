import pandas as pd
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # 15개 키포인트 인덱스 
        self.WRISTS = [9, 10]  # Left, Right Wrist
        self.KNEES = [13, 14]  # Left, Right Knee (발목 대신 무릎 사용)
        self.HEAD_PARTS = [0, 1, 2, 3, 4]  # Nose, Eyes, Ears
        self.SHOULDERS = [5, 6]  # 어깨
        self.HIPS = [11, 12]  # 엉덩이

    def extract_coordinates(self, kpts_list, idx):
        """flatten된 리스트에서 x, y 좌표 추출"""
        # kpts 구조: [x1, y1, conf1, x2, y2, conf2, ...]
        base = idx * 3
        if base + 1 >= len(kpts_list):
            return np.array([0, 0])
        return np.array([kpts_list[base], kpts_list[base+1]])

    def calculate_child_pose_features(self, kpts):
        """아동의 자세 특징 계산 - 손 위로 들기, 구부정한 자세, 넘어짐 감지"""
        if len(kpts) < 15:
            return 0, 0, 0  # hands_above_head, bending_posture, child_fall
        
        hands_above_head = 0
        bending_posture = 0
        child_fall = 0
        
        try:
            # 1. 손이 머리 위로 올라갔는지 확인 (방어 자세 감지)
            nose = kpts[0] if kpts[0][2] > 0.3 else None
            left_wrist = kpts[9] if kpts[9][2] > 0.3 else None
            right_wrist = kpts[10] if kpts[10][2] > 0.3 else None
            
            if nose is not None:
                # 코보다 30px 이상 위에 있고, 어깨보다 위에 있는 경우 (방어 자세)
                if left_wrist is not None and left_wrist[1] < nose[1] - 30:
                    left_shoulder = kpts[5] if kpts[5][2] > 0.3 else None
                    if left_shoulder is None or left_wrist[1] < left_shoulder[1]:
                        hands_above_head += 1
                
                if right_wrist is not None and right_wrist[1] < nose[1] - 30:
                    right_shoulder = kpts[6] if kpts[6][2] > 0.3 else None
                    if right_shoulder is None or right_wrist[1] < right_shoulder[1]:
                        hands_above_head += 1
            
            # 2. 구부정하거나 몸을 비틀거나 방어하는 자세 감지
            left_shoulder = kpts[5] if kpts[5][2] > 0.3 else None
            right_shoulder = kpts[6] if kpts[6][2] > 0.3 else None
            left_hip = kpts[11] if kpts[11][2] > 0.3 else None
            right_hip = kpts[12] if kpts[12][2] > 0.3 else None
            
            if (left_shoulder is not None and right_shoulder is not None and
                left_hip is not None and right_hip is not None):
                
                shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_avg_y = (left_hip[1] + right_hip[1]) / 2
                
                # 어깨보다 엉덩이가 30px 이상 아래에 있으면 구부정한 자세
                if hip_avg_y > shoulder_avg_y + 30:
                    bending_posture = 1
                
                # 어깨와 엉덩이의 수평선이 많이 틀어졌을 때 (몸을 비튼 자세)
                shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                hip_center_x = (left_hip[0] + right_hip[0]) / 2
                if abs(shoulder_center_x - hip_center_x) > 50:  # 50px 이상 틀어짐
                    bending_posture = 1
            
            # 3. 넘어짐 감지 (개선된 알고리즘)
            child_fall = self.detect_child_fall(kpts)
            
        except Exception as e:
            print(f"아동 자세 분석 중 오류: {e}")
        
        return hands_above_head, bending_posture, child_fall

    def detect_child_fall(self, kpts):
        """아동 넘어짐 감지: 키포인트 간 거리가 급격히 짧아지고, 키포인트가 압축되는 경우"""
        if len(kpts) < 15:
            return 0
        
        try:
            # 유효한 키포인트만 필터링
            valid_kpts = [kp for kp in kpts if kp[2] > 0.3]
            if len(valid_kpts) < 8:
                return 0
            
            # 키포인트 좌표 추출
            kpts_coords = np.array([kp[:2] for kp in valid_kpts])
            
            # 1. 키포인트 간 평균 거리 계산
            num_kpts = len(kpts_coords)
            if num_kpts < 2:
                return 0
            
            total_dist = 0
            count = 0
            for i in range(num_kpts):
                for j in range(i+1, num_kpts):
                    total_dist += np.linalg.norm(kpts_coords[i] - kpts_coords[j])
                    count += 1
            
            if count == 0:
                return 0
            
            avg_distance = total_dist / count
            
            # 2. 키포인트 분포 범위 계산 (bounding box 크기)
            min_x = np.min(kpts_coords[:, 0])
            max_x = np.max(kpts_coords[:, 0])
            min_y = np.min(kpts_coords[:, 1])
            max_y = np.max(kpts_coords[:, 1])
            
            width = max_x - min_x
            height = max_y - min_y
            bbox_area = width * height
            
            # 3. 넘어짐 조건: 키포인트 간 평균 거리가 짧고, bounding box 크기가 작은 경우
            # (기존 70, 20000 → 110, 50000 으로 완화)
            if avg_distance < 110 and bbox_area < 50000:
                return 1
                
            # 4. 머리(0)와 무릎(13,14)의 높이 차이가 급격히 줄어든 경우
            nose = kpts[0] if kpts[0][2] > 0.3 else None
            left_knee = kpts[13] if kpts[13][2] > 0.3 else None
            right_knee = kpts[14] if kpts[14][2] > 0.3 else None
            
            if nose is not None and (left_knee is not None or right_knee is not None):
                if left_knee is not None and right_knee is not None:
                    knee_y = (left_knee[1] + right_knee[1]) / 2
                elif left_knee is not None:
                    knee_y = left_knee[1]
                else:
                    knee_y = right_knee[1]
                
                # 머리와 무릎의 높이 차이가 90px 이하이면 넘어진 것으로 판단 (기존 50)
                if abs(nose[1] - knee_y) < 90:
                    return 1
            
            # 5. 어깨와 엉덩이의 높이 차이가 작고, 수평 자세인 경우
            left_shoulder = kpts[5] if kpts[5][2] > 0.3 else None
            right_shoulder = kpts[6] if kpts[6][2] > 0.3 else None
            left_hip = kpts[11] if kpts[11][2] > 0.3 else None
            right_hip = kpts[12] if kpts[12][2] > 0.3 else None
            
            if (left_shoulder is not None and right_shoulder is not None and
                left_hip is not None and right_hip is not None):
                
                shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_avg_y = (left_hip[1] + right_hip[1]) / 2
                
                # 어깨와 엉덩이의 높이 차이가 70px 이하이면 수평 자세 (넘어짐, 기존 40)
                if abs(shoulder_avg_y - hip_avg_y) < 70:
                    return 1
            
            return 0
            
        except Exception as e:
            print(f"넘어짐 감지 중 오류: {e}")
            return 0

    def calculate_reaction_velocity(self, kpts):
        """아동의 반응 속도 계산 - 팔, 다리의 움직임 속도 (학대 신고용으로만 사용)"""
        if len(kpts) < 15:
            return 0
        
        try:
            # 손목(9,10)과 무릎(13,14)의 속도 계산
            reaction_points = [9, 10, 13, 14]
            velocities = []
            
            for point in reaction_points:
                if kpts[point][2] > 0.3:
                    velocities.append(abs(kpts[point][0]) + abs(kpts[point][1]))
            
            if velocities:
                return np.mean(velocities)
            return 0
            
        except Exception as e:
            print(f"반응 속도 계산 중 오류: {e}")
            return 0

    def process(self, preds_df):
        if preds_df.empty: 
            return preds_df

        # 문자열로 저장된 리스트 복원 
        if isinstance(preds_df['keypoints'].iloc[0], str):
            preds_df['keypoints'] = preds_df['keypoints'].apply(eval)

        # 키포인트 개수 확인 (15개로 고정)
        num_kpts = 15

        # 1. 속도(Velocity) 계산
        preds_df = preds_df.sort_values(['track_id', 'frame'])
        preds_df['limb_velocity'] = 0.0
        
        for tid in preds_df['track_id'].unique():
            idx = preds_df[preds_df['track_id'] == tid].index
            kpts = np.stack(preds_df.loc[idx, 'keypoints'].values)
            
            # 키포인트 reshape (N, 15, 3)
            kpts = kpts.reshape(-1, num_kpts, 3)[:, :, :2]
            
            # 속도 계산
            if len(kpts) > 1:
                diff = np.diff(kpts, axis=0)
                dist = np.linalg.norm(diff, axis=2)
                
                # 주요 타격 부위(손, 무릎)의 속도
                limb_indices = self.WRISTS + self.KNEES
                valid_limb_indices = [idx for idx in limb_indices if idx < num_kpts]
                
                if valid_limb_indices:
                    vel = np.mean(dist[:, valid_limb_indices], axis=1)
                    vel = np.insert(vel, 0, 0)
                    preds_df.loc[idx, 'limb_velocity'] = vel
                else:
                    preds_df.loc[idx, 'limb_velocity'] = 0
            else:
                preds_df.loc[idx, 'limb_velocity'] = 0

        # 2. 근접성(Proximity) 및 상호작용 계산
        preds_df['min_dist_to_victim'] = 9999.0
        
        frames = preds_df['frame'].unique()
        for f in frames:
            frame_data = preds_df[preds_df['frame'] == f]
            adults = frame_data[frame_data['class'] == 1]
            children = frame_data[frame_data['class'] == 0]

            if adults.empty or children.empty:
                continue
            
            # 모든 성인 - 모든 아동 쌍에 대해 거리 계산
            for idx_a, adult in adults.iterrows():
                a_kpts = np.array(adult['keypoints']).reshape(num_kpts, 3)
                
                min_dist = 9999.0
                for idx_c, child in children.iterrows():
                    c_kpts = np.array(child['keypoints']).reshape(num_kpts, 3)
                    
                    attacker_indices = self.WRISTS + self.KNEES
                    victim_indices = self.HEAD_PARTS + self.SHOULDERS
                    
                    valid_attacker = [idx for idx in attacker_indices if idx < num_kpts]
                    valid_victim = [idx for idx in victim_indices if idx < num_kpts]
                    
                    attacker_pts = a_kpts[valid_attacker, :2]
                    victim_pts = c_kpts[valid_victim, :2]
                    
                    dists = []
                    for ap in attacker_pts:
                        if ap[0] == 0 and ap[1] == 0: 
                            continue
                        for vp in victim_pts:
                            if vp[0] == 0 and vp[1] == 0: 
                                continue
                            dists.append(np.linalg.norm(ap - vp))
                    
                    if dists:
                        current_pair_min = min(dists)
                        if current_pair_min < min_dist:
                            min_dist = current_pair_min
                
                if min_dist < 9999.0:
                    preds_df.loc[idx_a, 'min_dist_to_victim'] = min_dist

        # 3. 아동 자세 특징 계산 (학대 신고용)
        preds_df['child_hands_above_head'] = 0
        preds_df['child_bending_posture'] = 0
        preds_df['child_fall'] = 0
        preds_df['child_reaction_velocity'] = 0.0  # 아동 반응 속도 추가 (학대 신고용)
        
        child_data = preds_df[preds_df['class'] == 0]
        for idx, row in child_data.iterrows():
            kpts = np.array(row['keypoints']).reshape(num_kpts, 3)
            hands_above, bending, child_fall = self.calculate_child_pose_features(kpts)
            # limb_velocity(팔/다리 속도)를 그대로 아동 반응 속도로 사용
            reaction_velocity = float(row.get('limb_velocity', 0.0))
            
            preds_df.loc[idx, 'child_hands_above_head'] = hands_above
            preds_df.loc[idx, 'child_bending_posture'] = bending
            preds_df.loc[idx, 'child_fall'] = child_fall
            preds_df.loc[idx, 'child_reaction_velocity'] = reaction_velocity

        return preds_df