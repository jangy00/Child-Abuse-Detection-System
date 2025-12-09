## 아동 학대 자동 탐지 시스템 MVP


---
## R&D Team

<table align="center" width="100%">
  <tr>
    <!-- 김장유 -->
    <td align="center" width="240">
      <img src="https://github.com/jangy00.png" alt="jangy00 avatar" width="100" style="border-radius:12px"/><br/>
      <b>김장유</b><br/>
      Team Leader
      <div style="font-size:12px; margin-top:6px;">
        <div>프로젝트 총괄 </div>
        <div>모델 학습 및 개발</div>
      </div>
      <div style="margin-top:10px;">
      <br>
        <a href="https://github.com/jangy00" target="_blank">
          <img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub Link"/>
        </a>
      </div>
    </td>
    <!-- 송정민 -->
    <td align="center" width="240">
      <img src="https://github.com/SJM-source.png" alt="SJM-source avatar" width="100" style="border-radius:12px"/><br/>
      <b>송정민</b><br/>
      Full-Stack
      <div style="font-size:12px; margin-top:6px;">
        <div>추론 데이터 세팅</div>
        <div>규칙 데이터 세팅</div>
      </div>
      <div style="margin-top:10px;">
      <br>
        <a href="https://github.com/SJM-source" target="_blank">
          <img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub Link"/>
        </a>
      </div>
    </td>
    <!-- 박유진 -->
    <td align="center" width="240">
      <img src="https://github.com/lhj8-8.png" alt="lhj8-8 avatar" width="100" style="border-radius:12px"/><br/>
      <b>박유진</b><br/>
      Full-Stack
      <div style="font-size:12px; margin-top:6px;">
        <div>데이터 수집</div>
        <div>streamlit 보조</div>
      </div>
      <div style="margin-top:10px;">
      <br>
        <a href="https://github.com/pyj1110" target="_blank">
          <img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub Link"/>
        </a>
      </div>
    </td>
  </tr>
</table>

---

## <a id="overview"></a>Overview
**아동학대** 자동 탐지를 위한 시스템 개발 MVP

> 🎯 **목표:** 파이프라인 정교화 및 모델 개선  

---
## 주요 기능
**1. 실시간 분석 및 배치 분석**

성인과 아동 객체를 키포인트와 시계열 감지 시스템으로 학습한 YOLOv8 모델을 사용하여 feature 데이터로 의심/학대를 증상을 감지합니다.

> **실시간 분석** <br>
├─M2.py <br> 
├─M3.py <br> 
├─app.py <br><br>
**배치 분석** <br>
├─M1.py <br> 
├─M2.py <br> 
├─M3.py <br> 
├─main.py

**2. 상세 지표 타임라인**

학대 구간 감지 확인 시 해당 구간에 대한 좌표와 학대 구간을 클립을 생성합니다.<br><br>
프레임을 빠른 단위로 분석 후 원본의 자료로 의심/학대 구간를 확인할 수 있습니다.


---

## Tech Stack

### AI TRAINING
<img src="https://img.shields.io/badge/YOLOv8-000000?style=for-the-badge&logo=https://raw.githubusercontent.com/ultralytics/assets/main/logo/ultralytics.svg&logoColor=white">
<img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white">

### FRONT-END
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">

### TOOLING
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
<img src="https://img.shields.io/badge/HuggingFace-F7931A?style=for-the-badge&logo=huggingface&logoColor=white">

---

## <a id="installation--run"></a>Installation

### ① 프로젝트 클론
```js
# 가상 환경 설정
conda create -n 가상환경이름 python=3.11
conda activate 가상환경이름

# pip upgrade
pip install --upgrade pip

# 패키지 설치
cd app_streamlit
pip install -r requirements.txt

# 실행
streamlit run app.py
```
---
## <a id="updatelog"></a>업데이트 로그
| 버전         | 변경 사항               | 날짜      |
|------------|---------------------|---------|
| **v1.0.0** | 초기 파이프라인 설계 및 모델 실행 | 2025-12 |
| **v1.1.0** | -                   | -       |
| **v1.2.0** | -                   | -       |



