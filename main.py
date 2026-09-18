# 역할 정의. 게임을 실행하고 전체 흐름 및 플레이어 조작 루프를 데모하는 메인 엔트리 스크립트입니다.

from src.models.card import Card
from src.common.enums import Zone, CardType, EffectType, TargetType
from src.engine.main_game_logic import Game
from src.models.player import Player
from src.common import card_data
from svai.actions import apply_action, END_TURN
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox


def load_deck_file(filename, deck_dir="decks"):
    """선택한 덱 파일을 로드하여 CardData 객체 목록으로 변환합니다."""
    if not filename or filename == "기본 예시 덱":
        return None
    filepath = os.path.join(deck_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    cards_list = []
    for item in data.get("cards", []):
        card_id = item.get("card_id")
        count = item.get("count", 1)
        c_data = card_data.get_card_data_by_id(card_id)
        if c_data:
            for _ in range(count):
                cards_list.append(c_data)
    return cards_list


def select_decks_gui():
    """Tkinter를 사용하여 플레이어별 덱을 선택하는 다이얼로그를 띄웁니다."""
    deck_dir = "decks"
    deck_files = []
    if os.path.exists(deck_dir):
        deck_files = [f for f in os.listdir(deck_dir) if f.endswith(".json")]

    # 덱이 아예 없는 경우 콤보박스에 표시할 텍스트입니다.
    choices = deck_files if deck_files else ["기본 예시 덱"]

    root = tk.Tk()
    root.title("SVsim 덱 선택")
    root.geometry("400x250")
    
    # 다크 테마 느낌으로 스타일을 통일합니다.
    bg_dark = "#1e1e2e"
    bg_panel = "#313244"
    fg_light = "#cdd6f4"
    accent_blue = "#89b4fa"
    
    root.configure(bg=bg_dark)
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background=bg_dark)
    style.configure("TLabel", background=bg_dark, foreground=fg_light, font=("맑은 고딕", 10))
    style.configure("Header.TLabel", background=bg_dark, foreground=accent_blue, font=("맑은 고딕", 12, "bold"))
    style.configure("TCombobox", fieldbackground=bg_panel, background=bg_dark, foreground=fg_light)

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    ttk.Label(frame, text="플레이어 1 덱 선택", style="Header.TLabel").pack(anchor=tk.W, pady=5)
    p1_var = tk.StringVar(value=choices[0])
    p1_combo = ttk.Combobox(frame, textvariable=p1_var, values=choices, state="readonly", width=30)
    p1_combo.pack(fill=tk.X, pady=5)

    ttk.Label(frame, text="플레이어 2 덱 선택", style="Header.TLabel").pack(anchor=tk.W, pady=5)
    p2_var = tk.StringVar(value=choices[0])
    p2_combo = ttk.Combobox(frame, textvariable=p2_var, values=choices, state="readonly", width=30)
    p2_combo.pack(fill=tk.X, pady=5)

    def load_deck_file_with_gui(filename):
        """다이얼로그에서 예외가 발생하면 경고 메시지를 띄우고 기본 덱으로 작동시킵니다."""
        try:
            return load_deck_file(filename, deck_dir)
        except Exception as e:
            messagebox.showwarning("덱 로드 실패", f"덱 파일 로드 실패로 기본 덱을 사용합니다. {str(e)}")
            return None

    result = {"p1": None, "p2": None}

    def start_game():
        """선택한 덱 정보로 게임을 시작합니다."""
        result["p1"] = load_deck_file_with_gui(p1_var.get())
        result["p2"] = load_deck_file_with_gui(p2_var.get())
        root.destroy()

    def start_fallback():
        """기본 덱 설정을 적용하여 시작합니다."""
        result["p1"] = None
        result["p2"] = None
        root.destroy()

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill=tk.X, pady=20)

    start_btn = tk.Button(btn_frame, text="게임 시작", command=start_game, bg=accent_blue, fg=bg_dark, font=("맑은 고딕", 10, "bold"), relief=tk.FLAT, padx=10)
    start_btn.pack(side=tk.LEFT, padx=5)

    fallback_btn = tk.Button(btn_frame, text="기본 덱으로 시작", command=start_fallback, bg=bg_panel, fg=fg_light, font=("맑은 고딕", 10), relief=tk.FLAT, padx=10)
    fallback_btn.pack(side=tk.RIGHT, padx=5)

    root.mainloop()
    return result["p1"], result["p2"]


# 게임 실행 예시입니다.
if __name__ == "__main__":
    card_data.load_card_databases('card_database/3_parsed_database/card_database_parsed.json')
    p1_deck, p2_deck = select_decks_gui()
    # Default view is the tkinter GameGUI and the default decider is a HumanDecider
    # bound to its dialogs, i.e. human vs human exactly as before.
    game = Game("player1", "player2", p1_deck, p2_deck)
    current_player = "player1"

    for turn_num in range(1, 21):  # 20턴까지 진행하는 예시입니다.

        while True:
            decider = game.decider_for(current_player)
            # The human menu flow lives in svai.deciders.HumanDecider.choose_action.
            action = decider.choose_action(game, current_player, None)
            apply_action(game, current_player, action)
            decider.notify_action_applied(game, current_player, action)
            if action["type"] == END_TURN:
                break  # 턴 종료 후 다음 플레이어로 넘어갑니다.

        # 턴 플레이어를 전환합니다.
        current_player = game.opponent_id[current_player]

    game.gui.get_user_choice("--- 게임 종료 예시 ---", {"확인": None})