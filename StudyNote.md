# Legged Locomotion Study Notes

## 프로젝트 개요
- **프로젝트명**: legged-loco
- **주요 로봇**: Go1, G1, H1, Go2
- **학습 프레임워크**: Isaac Lab + RSL-RL
- **목표**: 시각 정보를 활용한 로봇 보행 학습

## 주요 학습 내용

### 1. 학습 명령어 분석

#### 기본 학습 명령어
```bash
python scripts/train.py --task=go1_vision --history_length=9 --run_name=XXX --max_iterations=2000 --save_interval=200 --headless --video --use_cnn --enable_cameras
```

#### 인자별 역할
- `--task=go1_vision`: Go1 로봇의 시각 기반 태스크
- `--history_length=9`: 9개의 이전 관측값을 히스토리 버퍼에 저장
- `--use_cnn`: 깊이 이미지를 CNN으로 처리
- `--use_rnn`: RNN을 사용한 시계열 메모리
- `--enable_cameras`: 카메라 센서 활성화

### 2. `--history_length=9` 인자 분석

#### 전체 스택 트레이스
1. **인자 파싱**: `train.py`에서 `args_cli.history_length = 9` 설정
2. **환경 래퍼**: `RslRlVecEnvHistoryWrapper` 적용
3. **히스토리 버퍼**: `(num_envs, 9, proprio_obs_dim)` 크기 생성
4. **관측값 처리**: Proprioceptive 관측값만 히스토리 버퍼에 저장
5. **정책 네트워크**: 히스토리 정보를 포함한 확장된 관측값 처리

#### 영향
- **메모리 사용량**: 각 환경마다 추가 메모리 사용
- **관측값 확장**: 현재 관측값 + 9개 이전 관측값 연결
- **학습 효과**: 시계열 정보를 활용한 더 안정적인 정책 학습

### 3. `--use_rnn`, `--use_cnn`, `--history_length` 상호작용

#### 정책 클래스 결정 로직
- `--use_rnn` 없음 + `--use_cnn` 없음 → `ActorCritic`
- `--use_rnn` 없음 + `--use_cnn` 있음 → `ActorCriticDepthCNN`
- `--use_rnn` 있음 + `--use_cnn` 없음 → `ActorCriticRecurrent`
- `--use_rnn` 있음 + `--use_cnn` 있음 → `ActorCriticDepthCNNRecurrent`

#### 조합별 특징
| 조합 | 정책 클래스 | Proprioceptive 처리 | 깊이 이미지 처리 | 시계열 메모리 | 히스토리 버퍼 |
|------|-------------|---------------------|------------------|---------------|---------------|
| 기본 | ActorCritic | MLP | 없음 | 없음 | 없음 |
| +CNN | ActorCriticDepthCNN | MLP | CNN | 없음 | 히스토리 래퍼 |
| +RNN | ActorCriticRecurrent | RNN | 없음 | LSTM/GRU | 없음 |
| +CNN+RNN | ActorCriticDepthCNNRecurrent | MLP→RNN | CNN→RNN | LSTM/GRU | 히스토리 래퍼 |

### 4. Depth Sensor 데이터 처리 과정

#### 센서 설정
```python
depth_sensor = RayCasterCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/trunk",
    offset=RayCasterCameraCfg.OffsetCfg(pos=(0.272, 0.0075, 0.092), rot=(0.389, 0.0, 0.921, 0.0)),
    pattern_cfg=patterns.PinholeCameraPatternCfg(
        focal_length=1.93, horizontal_aperture=3.8,
        height=24, width=32,
    ),
    max_distance=3,
)
```

#### 데이터 처리 파이프라인
1. **레이캐스팅**: 각 픽셀에서 거리 측정
2. **전처리**: 클리핑 (0.3m ~ 2.0m), 정규화
3. **평면화**: `(24, 32)` → `768차원`
4. **CNN 처리**: `DepthOnlyFCBackbone`을 통한 특징 추출
5. **정책 입력**: Proprioceptive + Depth Image 결합

#### CNN 백본 구조
```python
# 입력: (1, 24, 32)
# Conv2d + MaxPool: (1, 24, 32) → (16, 10, 14)
# Conv2d + MaxPool: (16, 10, 14) → (32, 4, 6)
# Flatten: (32, 4, 6) → 768차원
# MLP: 768 → hidden_dim → output_dim(128)
```

### 5. Go1 3발 보행 문제 분석

#### 문제 원인
1. **보상 함수 부족**: Hip 관절 제약, 보행 리듬, 속도 추적 보상 없음
2. **관측값 누락**: `base_lin_vel`, `projected_gravity` 등 중요 관측값 주석 처리
3. **몸통 안정성 부족**: `flat_orientation_l2` 보상 비활성화

#### 해결 방법
```python
# 보상 함수 수정
@configclass 
class CustomGo1RewardsCfg(RewardsCfg):
    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,  # 강한 페널티
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,  # 약한 페널티
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.25},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    # flat_orientation_l2는 원본 Isaac Lab에서도 0.0으로 설정됨
```

### 6. Sim2Real 구현 가이드

#### 관측값 구현 가능성
- **base_lin_vel**: 가속도 적분으로 구현 가능 (드리프트 문제 있음)
- **projected_gravity**: IMU 가속도계로 구현 가능
- **depth_image**: 실제 깊이 카메라로 구현 가능

#### 구현 예시
```python
def compute_projected_gravity(imu_data):
    accel = imu_data.accelerometer
    gravity_magnitude = torch.norm(accel)
    projected_gravity = accel / gravity_magnitude
    return projected_gravity
```

### 7. 주요 보상 함수 설명

#### Isaac Lab 원본 보상 가중치
| Index | Name | Weight |
|-------|------|--------|
| 0 | track_lin_vel_xy_exp | 1.5 |
| 1 | track_ang_vel_z_exp | 0.75 |
| 2 | lin_vel_z_l2 | -2.0 |
| 3 | ang_vel_xy_l2 | -0.05 |
| 4 | dof_torques_l2 | -0.0002 |
| 5 | dof_acc_l2 | -2.5e-07 |
| 6 | action_rate_l2 | -0.01 |
| 7 | feet_air_time | 0.01 |
| 8 | flat_orientation_l2 | 0.0 |
| 9 | dof_pos_limits | 0.0 |

#### 핵심 보상들
- **hip_deviation**: Hip 관절 제약 (자연스러운 보행 유도)
- **feet_air_time**: 발 공중 시간 (보행 리듬 유도)
- **track_lin_vel_xy_exp**: 선형 속도 추적
- **track_ang_vel_z_exp**: 각속도 추적
- **flat_orientation_l2**: 몸통 수평 유지 (원본에서도 0.0으로 비활성화)
- **base_height**: 기본 높이 유지

#### 권장 가중치
```python
self.rewards.feet_air_time.weight = 0.25  # 원본보다 강화
self.rewards.track_lin_vel_xy_exp.weight = 1.5
self.rewards.track_ang_vel_z_exp.weight = 0.75
self.rewards.flat_orientation_l2.weight = 0.0  # 원본과 동일하게 비활성화
self.rewards.dof_torques_l2.weight = -0.0002
```

### 8. 에러 해결

#### ActorCriticDepthCNNRecurrent 에러
**문제**: `self.critic.encode()` 메서드 없음
**원인**: `critic`이 `nn.Sequential`로 정의되어 `encode` 메서드 없음
**해결**: `self.actor.encode()` 사용하도록 수정

```python
def evaluate(self, critic_observations, masks=None, hidden_states=None):
    critic_observations = self.actor.encode(critic_observations)  # 수정
    input_c = self.memory_c(critic_observations, masks, hidden_states)
    return super().evaluate_hidden(input_c.squeeze(0))
```

### 9. 학습 모범 사례

#### 설정 권장사항
1. **보상 함수**: Hip 관절 제약, 보행 리듬, 속도 추적 보상 필수
2. **관측값**: `base_lin_vel`, `projected_gravity` 활성화
3. **히스토리 길이**: 9개 정도가 적절
4. **CNN + RNN**: 시각 정보와 시계열 정보 모두 활용

#### 디버깅 팁
1. **보상 모니터링**: 각 보상 항목의 가중치와 값 확인
2. **관측값 시각화**: 깊이 이미지, 관절 상태 등 확인
3. **보행 패턴**: 발 접촉 패턴, 보행 리듬 분석

## 업데이트 로그
- **1734566400**: 초기 노트 생성, 기본 학습 내용 정리
- **1734566400**: Depth Sensor 처리 과정 상세 분석 추가
- **1734566400**: 3발 보행 문제 해결 방법 추가
- **1734566400**: Sim2Real 구현 가이드 추가
- **1734566400**: Isaac Lab 원본 보상 가중치 정보 수정, flat_orientation_l2가 원본에서도 0.0으로 비활성화되어 있음을 확인
- **1761381270**: Go1 Vision 설정에 보상 함수 수정사항 실제 반영 완료
  - hip_deviation 보상 추가 (weight=-0.4)
  - joint_deviation 보상 세분화 (hip 제외, weight=-0.04)
  - base_height 보상 추가 (weight=-5.0)
  - feet_air_time 보상 추가 (weight=0.25)
  - base_lin_vel, projected_gravity 관측값 활성화
  - feet_air_time 가중치 강화 (0.01 → 0.25)
- **1761381639**: 원본 Isaac Lab 코드와 비교 분석 완료
  - Height Scanner (160포인트) vs Depth Sensor (768포인트) 정보량 비교
  - 보상 함수 차이점 분석: 원본에는 hip_deviation, base_height 없음
  - 권장사항: 보상 가중치 완화 고려, Height Scanner 하이브리드 접근법 검토
