# 퍼징 테스트 에러 분석 리포트

## 1 에러 기본 요약
- **에러 유형** AssertionError
- **에러 메시지** error.log 실시간 파싱 중 이상 에러 검출 - [ERROR] get_entity_by_id - ID 80를 찾을 수 없습니다.

## 2 에러 발생 시점 게임 상태 스냅샷
- **현재 진행 턴** 6
- **현재 턴 플레이어** player2
- **플레이어 1 체력** 20 (PP 0)
- **플레이어 2 체력** 20 (PP 0)

### 플레이어 1 손패 카드 목록
['빙계의 사슴왕', '폭풍의 천업 그림니르', '무지갯빛 궁수 쿠피탄', '충풍화 미로쿠', '적염의 무희 안스리아']

### 플레이어 2 손패 카드 목록
['진소화 이마리', '종연의 현현 리셰나', '거침없는 캐스터', '파괴의 단결자', '무뚝뚝한 마스터']

### 플레이어 1 필드 카드 목록
[]

### 플레이어 2 필드 카드 목록
['비탄에 맞서는 자', '포격의 고양이 수인', '어택 아티팩트']

## 3 상세 트레이스백 정보
```text
Traceback (most recent call last):
  File "C:\Users\sys91\Documents\개발 프로젝트\SVsim\fuzz_runner.py", line 311, in run_fuzzing
    raise AssertionError(f"error.log 실시간 파싱 중 이상 에러 검출 - {logged_error}")
AssertionError: error.log 실시간 파싱 중 이상 에러 검출 - [ERROR] get_entity_by_id - ID 80를 찾을 수 없습니다.

```
