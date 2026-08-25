# 부재 증거 확인이 가능한 데이터셋 — 조사 2026-08-26

기준: (a) 물건이 떠난 자리를 **다시 방문**한 기록이 있고 (b) 부재를 GT 로 확인할
수 있는가. (c) 방/자리 수준 위치 GT 가 있으면 가산.

## 판정 요약

| 데이터셋 | 부재 GT | 재방문 | 판정 |
|---|---|---|---|
| **3RScan** (보유했었음, 재다운 가능) | ✅ rescan 간 removed/rigid 라벨 | ✅ 스캔 쌍 | **여전히 최적.** 단일 방 한계 (§우리 결과 0.625) |
| RIO10 | 3RScan 파생 (재국소화 벤치) | ✅ | 카메라 재국소화용 — 부재 라벨은 3RScan 것 |
| Living Scenes (2023) | ✅ 다시점 스캔, 물체 제거 시나리오 | ✅ | 3RScan 계열 소규모. 재구성 중심 |
| **EgoTracks** | ⚠️ 사라짐/재검출 주석 22k 개 | 시야 수준 | **조건③ 그대로** — "시야에서 사라짐" 이지 "자리에서 사라짐" 이 아니다 |
| **Ego4D VQ2D/VQ3D** | ⚠️ "마지막으로 본 위치" GT 22k 질의·433h | 부분 | 검색 GT 는 최대 규모. 부재는 유도만 가능(위치 GT 없음) |
| SuperMemory-VQA (보유) | ⚠️ 물체·위치 기억 질의 | ✅ | 이미 사용(비율 2.4~2.5). 방 GT 우리가 유도 |
| X-LeBench (2025) | ❌ 초장시간 이해 벤치 | — | 질의 형식이 VQA — 부재 GT 없음 |
| LongEgoRefer (2026) | ⚠️ **absent-object 지칭** 포함 | — | "없는 물체를 지칭했을 때 없다고 답하기" — 우리 (c) 상태와 흡사. 소규모(1.5k) |
| HD-EPIC (2025) | ❌ 요리 41h, 3D 그라운딩 | 부엌 단일 | 부재 시나리오 아님 |

## 결론

1. **실세계 부재 GT 의 결정판은 여전히 3RScan** — 새 조사에서도 이를 넘는 것이
   없다. 우리의 방·자리 수준 결과(0.625)가 이미 그 위에 있다.
2. **새로 확보할 가치가 있는 것 둘:**
   - **Ego4D VQ2D** — "어디서 마지막으로 봤나" 검색 GT 22k 개. 검색·국소화의
     실사 검증 규모를 몇 배로 키운다. 부재는 약하게만.
   - **LongEgoRefer** — absent-object 지칭이 명시적으로 든 첫 벤치. (c) "없다"
     답변 능력의 실사 채점에 직접 쓸 수 있다. 2026 공개라 입수 가능성 확인 필요.
3. EgoTracks 는 조건③(시야 이탈 ≠ 자리 이탈) 때문에 우리 목적에 안 맞는다 —
   과거에 자리 수준 부재가 우연으로 나온 것과 같은 구조.
4. 보유분(EgoLife·IT3DEgo·SuperMemory·ADT)으로 시작하고, Ego4D 는 실사 검증
   단계에서 내려받는다(라이선스 동의 필요, ~수백 GB 는 부분 다운 가능).

Sources: [EgoTracks](https://arxiv.org/pdf/2301.03213) · [Ego4D episodic-memory](https://github.com/EGO4D/episodic-memory) · [SuperMemory-VQA](https://arxiv.org/html/2606.00825) · [X-LeBench](https://arxiv.org/html/2501.06835) · [LongEgoRefer/최근 목록](https://www.labellerr.com/blog/egocentric-datasets-robotics/) · [3RScan/RIO10](https://github.com/WaldJohannaU/RIO10) · [Living Scenes](https://arxiv.org/pdf/2312.09138)
