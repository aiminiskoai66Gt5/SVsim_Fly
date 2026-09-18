"""User-facing strings (Traditional Chinese) and action labels.

Only dialog / menu text lives here. Engine log lines are not user-facing and
are left as they are. Card names come from the card database and follow
``src.common.card_data.DISPLAY_LANGUAGE``.
"""

# Labels returned by Game.get_available_actions(); also used as menu keys.
ACTION_ATTACK = "攻擊隨從"
ACTION_EVOLVE = "進化"
ACTION_SUPER_EVOLVE = "超進化"
ACTION_ENGAGE = "啟動護符（Engage）"

# Main menu (HumanDecider)
MENU_TITLE = "--- 選擇行動 ---"
MENU_PLAY_CARD = "從手牌出牌"
MENU_FIELD = "場上操作（攻擊 / 進化 / 超進化 / 啟動護符）"
MENU_END_TURN = "結束回合"
MENU_BACK = "返回"
MENU_CANCEL = "取消"
MENU_OK = "確認"
MENU_DONE = "完成"
USE_EXTRA_PP_PROMPT = "要使用額外 PP 嗎？"
USE_EXTRA_PP_YES = "使用"
USE_EXTRA_PP_NO = "不使用"
NO_PLAYABLE_CARD = "手牌中沒有可以打出的卡。"
HAND_TITLE = "--- 目前玩家的手牌 ---"
NO_FIELD_CARD = "場上沒有可以操作的隨從或護符。"
FIELD_TITLE = "--- 選擇要操作的卡 ---"
NO_ACTION_FOR_CARD = "這張卡目前沒有可以執行的行動。"
ACTION_FOR_CARD_TITLE = "[{name}] 要執行的行動 ---"
NO_ATTACK_TARGET = "沒有可以攻擊的目標。"
TARGET_TITLE = "--- 選擇攻擊目標 ---"
END_TURN_CONFIRM = "{player_id} 結束回合。"
INVALID_CHOICE = "無效的選擇，請重新選擇。"
EVOLVED_CONFIRM = "[{name}] 已進化！"
SUPER_EVOLVED_CONFIRM = "[{name}] 已超進化！"
ENGAGED_CONFIRM = "[{name}] 已啟動！"
DISCARD_PROMPT = "{player_id}：選擇要捨棄的卡"
FUSE_PROMPT = "{player_id}：選擇要融合進 {name} 的卡"
CHOOSE_AGAIN = "請重新選擇。"

# Engine-raised choices
EFFECT_OPTION = "效果 {n}"
CHOOSE_EFFECT = "{player_id}，請選擇效果："
CHOOSE_ALLY_FOLLOWER = "請選擇我方隨從："
CHOOSE_ALLY_CARD = "請選擇我方的卡："
CHOOSE_ENEMY_FOLLOWER = "請選擇對方隨從："
CHOOSE_ENEMY_FOLLOWER_NTH = "請選擇第 {n} 個對方隨從："
CHOOSE_HAND_CARD = "請選擇手牌中的卡（{i}/{count}）。"
CHOOSE_UNEVOLVED_ALLY = "請選擇尚未進化的我方隨從："

# GUI chrome
WINDOW_TITLE = "Shadowverse 對戰檢視器"
TURN_INFO = "第 {turn} 回合 - 目前玩家：{player}"
STATS_FRAME = "{player_id} 狀態"
HAND_FRAME = "手牌"
FIELD_FRAME = "場上"
STATS_TEXT = "血量 - {hp}/{max_hp}\nPP - {pp}/{max_pp}\nEP - {ep}/{max_ep}\nSEP - {sep}/{max_sep}\n牌庫 - {deck}\n墓地 - {grave}"
CARD_COST = "費用 - {cost}"
CARD_COUNTDOWN = "倒數 - {n}"
MULLIGAN_WINDOW_TITLE = "{player_id} 的換牌"
MULLIGAN_PROMPT = "勾選要換掉的卡（勾選的卡會被換掉）"
MULLIGAN_CARD = "{name} (ID - {card_id})（費用 - {cost}）"
MULLIGAN_CONFIRM = "確認換牌"
DECK_SELECT_TITLE = "SVsim 選擇牌組"
DECK_SELECT_P1 = "玩家 1 牌組"
DECK_SELECT_P2 = "玩家 2 牌組"
DECK_DEFAULT = "預設範例牌組"
DECK_LOAD_FAILED_TITLE = "牌組載入失敗"
DECK_LOAD_FAILED = "牌組檔案載入失敗，改用預設牌組。{error}"
DECK_START = "開始對戰"
OPPONENT_LABEL = "玩家 2 由誰操作"
OPPONENT_HUMAN = "人類（本機雙人）"
OPPONENT_GREEDY = "電腦：GreedyAgent"
DECK_START_DEFAULT = "用預設牌組開始"
GAME_OVER_DEMO = "--- 範例對局結束 ---"
GAME_OVER = "對局結束：{winner} 獲勝"

# Keyword names as the official site shows them (shadowverse-wb.com, lang=cht, 2026-09-19).
# Keys are EffectType member names; unlisted members fall back to the member name.
KEYWORD_ZH_TW = {
    "FANFARE": "入場曲", "LAST_WORDS": "謝幕曲", "ON_EVOLVE": "進化時", "EVOLVED": "進化時",
    "ON_SUPER_EVOLVE": "超進化時", "SUPER_EVOLVED": "超進化時", "STRIKE": "攻擊時", "CLASH": "交戰時",
    "WARD": "守護", "STORM": "疾馳", "RUSH": "突進", "AMBUSH": "潛行", "BANE": "必殺", "DRAIN": "吸血",
    "INTIMIDATE": "威懾", "AURA": "光紋", "BARRIER": "障壁", "OVERFLOW": "覺醒", "SPELLBOOST": "魔力增幅時",
    "COUNTDOWN": "倒數", "NECROMANCY": "死靈術", "REANIMATE": "亡者召還", "EARTH_RITE": "土之秘術",
    "EARTH_SIGIL": "土之印", "ENHANCE": "爆能強化", "INVOKE": "瞬念召喚", "RALLY": "協作", "COMBO": "連擊",
    "MODE": "模式", "ENGAGE": "策動", "SKYBOUND_ART": "奧義", "SUPER_SKYBOUND_ART": "解放奧義",
    "SPELL": "效果", "ON_MY_TURN_END": "我方回合結束時", "ON_OPPONENTS_TURN_END": "對方回合結束時",
    "ON_MY_TURN_START": "我方回合開始時", "ON_FOLLOWER_ENTER_FIELD": "從者進場時", "ON_LEAVE_FIELD": "離場時",
    "DISABLE": "無法攻擊", "ON_DISCARD": "被捨棄時",
}


def keyword_name(effect_type) -> str:
    """Display name of an EffectType (official zh-TW where known)."""
    return KEYWORD_ZH_TW.get(getattr(effect_type, "name", str(effect_type)), getattr(effect_type, "name", str(effect_type)))
