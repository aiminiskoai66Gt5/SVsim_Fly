# 퍼징 테스트 에러 분석 리포트

## 1 에러 기본 요약
- **에러 유형** AttributeError
- **에러 메시지** 'Process' object has no attribute 'value'

## 2 에러 발생 시점 게임 상태 스냅샷
- **현재 진행 턴** 10
- **현재 턴 플레이어** player2
- **플레이어 1 체력** 20 (PP 2)
- **플레이어 2 체력** 20 (PP 1)

### 플레이어 1 손패 카드 목록
['캣 네이비', '스타라이트 가디스', '리터 슈나이트', '호사스러운 여금화', '평범한 기사 라킬', '미몽의 사자 모드레드', '정통성의 왕관']

### 플레이어 2 손패 카드 목록
['혼융의 기도자', '조련사 스켈레톤', '용기로 가득한 자', '사경의 아나테마 도희', '교지의 타천사 벨리알', '쌍륜야행 긴세츠&유즈키', '다리를 잇는 청귀']

### 플레이어 1 필드 카드 목록
[]

### 플레이어 2 필드 카드 목록
['군배의 대장부', '홍단의 혼백술사']

## 3 상세 트레이스백 정보
```text
Traceback (most recent call last):
  File "C:\Users\sys91\Documents\개발 프로젝트\SVsim\fuzz_runner.py", line 337, in run_fuzzing
    game.evolve_follower(action["card_id"], current_player)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sys91\Documents\개발 프로젝트\SVsim\src\engine\main_game_logic.py", line 742, in evolve_follower
    self.process_events()
    ~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\sys91\Documents\개발 프로젝트\SVsim\src\engine\main_game_logic.py", line 143, in process_events
    self.event_manager.process_events()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\sys91\Documents\개발 프로젝트\SVsim\src\engine\event_manager.py", line 45, in process_events
    listener.callback(event)
    ~~~~~~~~~~~~~~~~~^^^^^^^
  File "C:\Users\sys91\Documents\개발 프로젝트\SVsim\src\engine\main_game_logic.py", line 204, in _handle_card_effect
    self.effect_processor.resolve_effect(effect_to_resolve, card_id, self.game_state_manager, target_id)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sys91\Documents\개발 프로젝트\SVsim\src\engine\effect_processor.py", line 1324, in resolve_effect
    handler(process, target, game_state_manager)
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sys91\Documents\개발 프로젝트\SVsim\src\engine\effect_processor.py", line 1512, in _process_conditional_effect
    val = effect_data.value
          ^^^^^^^^^^^^^^^^^
  File "C:\Users\sys91\Documents\개발 프로젝트\SVsim\src\common\effect.py", line 53, in __getattr__
    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
AttributeError: 'Process' object has no attribute 'value'

```
