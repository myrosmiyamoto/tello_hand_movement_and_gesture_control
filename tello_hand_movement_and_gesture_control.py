from djitellopy import Tello, TelloException    # DJITelloPyのTelloクラスをインポート
import sys                                      # sys.exitを使うため
import time                                     # time.sleepを使うため
import cv2                                      # OpenCVを使うため
from threading import Thread                    # 並列処理をするため
from queue import Queue                         # 最新の映像にするため
import numpy as np                              # ラベリングにNumPyが必要なため
import mediapipe as mp                          # 手の検出に必要なため



# Telloを制御するクラス
class TelloControl:
    def __init__(self, ip, port):
        """ コンストラクタ """
        # Telloの設定
        Tello.RETRY_COUNT = 1          # retry_countは応答が来ないときのリトライ回数
        Tello.RESPONSE_TIMEOUT = 0.01  # 応答が来ないときのタイムアウト時間
        # Telloクラスを使って，telloというインスタンス(実体)を作る
        self.tello = Tello()
        self._connect_tello()

        # フラグ関係
        self.is_running = True         # Telloが動作中か
        self.is_automode = False       # 自動制御を行うかどうか
        self.is_flip = False

        # カメラストリームを取得
        self.cap = cv2.VideoCapture(f'udp://{ip}:{port}')

        # バッファを最小に設定
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 画像の幅と高さを設定
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # ウィンドウの名前
        self.frame_name = f'Tello {ip}'
        # 最新フレームを保持するキュー
        self.frame_queue = Queue(maxsize=1)

        # フレームをキャプチャするスレッドを開始
        self.capture_thread = Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()

        # Telloが10秒で自動的に制御停止するのを防ぐ処理のための変数
        self.last_command_time = time.time()    # 最後にコマンドを送信した時間

        # 手検出のセットアップ
        self._setup_hand_detection()
        self.prev_position = None
        self.velocity = (0, 0)

        self.target_x = int(self.width / 2)   # 画面の中心X座標
        self.target_y = int(self.height / 2)  # 画面の中心Y座標


    def _connect_tello(self):
        """ Telloとの接続を開始 """
        try:
            # Telloへ接続
            self.tello.connect()

            # 画像転送を有効にする
            self.tello.streamoff()   # 誤動作防止の為、最初にOFFする
            self.tello.streamon()    # 画像転送をONに
        except KeyboardInterrupt:
            print('\n[Finish] Press Ctrl+C to exit')
            sys.exit()
        except TelloException:
            print('\n[Finish] Connection timeout')
            sys.exit()


    def _setup_hand_detection(self):
        """ Mediapipeを用いた手の検出のセットアップ """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # 検出する手の最大数
            model_complexity=0,  # モデルの複雑さ
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )


    def _capture_frames(self):
        """ Telloからの映像ストリームを取得し、最新のフレームをキューに保持 """
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # 最新フレームのみを保持
                if not self.frame_queue.empty():
                    # キューからフレームを取り出す
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)


    def _detect_hand_movement_and_gesture(self, frame):
        """ Mediapipeを用いて手の動きを検出し、方向を判定 """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # ランドマークを描画
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = float('-inf'), float('-inf')

                for lm in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if self.is_automode:
                    # ドローンを中心にするための移動方向を計算
                    a = b = c = d = 0   # rcコマンドの初期値は0
                    width = x_max - x_min
                    height = y_max - y_min
                    center_x = int(x_min + width / 2)
                    center_y = int(y_min + height / 2)

                    dx = 0.3 * (self.target_x - center_x)
                    dy = 0.3 * (self.target_y - center_y)
                    dw = 0.4 * (80 - width)        # 基準顔サイズ100pxとの差分

                    # 旋回方向の不感帯を設定
                    d = 0.0 if abs(dx) < 20.0 else dx   # ±20未満ならゼロにする
                    # 旋回方向のソフトウェアリミッタ(±100を超えないように)
                    d =  100 if d >  100.0 else d
                    d = -100 if d < -100.0 else d

                    # 前後方向の不感帯を設定
                    b = 0.0 if abs(dw) < 10.0 else dw   # ±10未満ならゼロにする
                    # # 前後方向のソフトウェアリミッタ
                    b =  100 if b >  100.0 else b
                    b = -100 if b < -100.0 else b

                    # 上下方向の不感帯を設定
                    c = 0.0 if abs(dy) < 30.0 else dy   # ±30未満ならゼロにする
                    # 上下方向のソフトウェアリミッタ
                    c =  100 if c >  100.0 else c
                    c = -100 if c < -100.0 else c

                    self._send_movement_command(a, b, c, d)

                    # 手のジェスチャーを認識する処理
                    fingers = []
                    # 親指
                    # 人差し指～小指
                    for tip_id in [8, 12, 16, 20]:
                        fingers.append(hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y)

                    # ジェスチャー認識
                    if all(fingers[1:]):  # すべての指が開いている
                        gesture = "paper"
                    elif not any(fingers):  # すべての指が閉じている
                        gesture =  "rock"
                    elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                        gesture =  "Scissors"  # 人差し指と中指が開いていて、薬指と小指が閉じている
                        if not self.is_flip:
                            self.is_flip = True
                            Thread(target=self._tello_control, args=('b_flip', )).start()
                    else:
                        gesture = "nothing"

                    # 結果を表示
                    cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    def _send_movement_command(self, a, b, c, d):
        """ Telloへ移動コマンドを送信 """
        x, y = self.velocity
        self.tello.send_rc_control(int(a), int(b), int(c), int(d))


    def _send_periodic_command(self):
        """ 10秒ごとに'command'を送信し、Telloの制御を維持 """
        if time.time() - self.last_command_time > 10:
            Thread(target=self._tello_control, args=('command',)).start()
            self.last_command_time = time.time()


    def run(self):
        """ Tello制御のメインの関数 """
        while self.is_running:
            if not self.frame_queue.empty():  # キューの中が空でなければ処理を実行する
                frame = self.frame_queue.get()  # キューの値を取得
                frame = cv2.flip(cv2.resize(frame, (self.width, self.height)), 1)  # フレームのリサイズと反転

                self._detect_hand_movement_and_gesture(frame)

                cv2.imshow(self.frame_name, frame)  # Telloからの映像を表示

                self._handle_key_input()  # キー入力を処理
                self._send_periodic_command()  # 10秒ごとに'command'を送信し、Telloの制御を維持


    def _handle_key_input(self):
        """ キー入力を処理 """
        key = cv2.waitKey(1) & 0xFF
        # ESCキーが押されたら、また、ウィンドウが終了したらストップ
        if key == 27 or cv2.getWindowProperty(self.frame_name, cv2.WND_PROP_VISIBLE) < 1:
            print('[Finish] Press ESC key or close window to exit')
            self.stop()
        elif key == ord('t'):  # 離陸
            self.tello.takeoff()
        elif key == ord('l'):  # 着陸
            self.tello.land()
        elif key == ord('w'):  # 前進 30cm
            Thread(target=self._tello_control, args=('w', )).start()
        elif key == ord('s'):  # 後進 30cm
            Thread(target=self._tello_control, args=('s', )).start()
        elif key == ord('a'):  # 左移動 30cm
            Thread(target=self._tello_control, args=('a', )).start()
        elif key == ord('d'):  # 右移動30cm
            Thread(target=self._tello_control, args=('d', )).start()
        elif key == ord('e'):  # 旋回-時計回り 30度
            Thread(target=self._tello_control, args=('e', )).start()
        elif key == ord('q'):  # 旋回-反時計回り 30度
            Thread(target=self._tello_control, args=('q', )).start()
        elif key == ord('r'):  # 上昇 30cm
            Thread(target=self._tello_control, args=('r', )).start()
        elif key == ord('f'):  # 下降 30cm
            Thread(target=self._tello_control, args=('f', )).start()
        elif key == ord('p'):  # ステータスをprintする
            print(self.tello.get_current_state())
        elif key == ord('1'):  # 自動モードON
            self._toggle_automode(True)
        elif key == ord('0'):  # 自動モードOFF
            self.tello.send_rc_control(0, 0, 0, 0)
            self._toggle_automode(False)


    def _toggle_automode(self, state):
        """ 自動モードの切り替え """
        self.is_automode = state
        print(f'Auto mode {"ON" if state else "OFF"}')


    def stop(self):
        """ 終了処理 """
        self.tello.emergency()  # Telloの動作を完全に停止
        print(f'[Battery] {self.tello.get_battery()}%')  # バッテリー残量を表示
        self.is_running = False  # ストリームを停止
        self.capture_thread.join()  # スレッドを終了
        cv2.destroyAllWindows()  # ウィンドウを閉じる
        self.tello.streamoff()
        if self.cap.isOpened():
            self.cap.release()  # カメラを開放
        self.tello.end()


    def _tello_control(self, control_flag):
        """ Telloを制御する関数 """
        if control_flag == 'command':
            self.tello.send_command_without_return('command')    # 'command'送信
        elif control_flag == 'w':  # 前進 30cm
            self.tello.move_forward(30)
            time.sleep(1)
        elif control_flag == 's':  # 後進 30cm
            self.tello.move_back(30)
            time.sleep(1)
        elif control_flag == 'a':  # 左移動 30cm
            self.tello.move_left(30)
            time.sleep(1)
        elif control_flag == 'd':  # 右移動30cm
            self.tello.move_right(30)
            time.sleep(1)
        elif control_flag == 'e':  # 時計回りに旋回 30度
            self.tello.rotate_clockwise(30)
            time.sleep(1)
        elif control_flag == 'q':  # 反時計回りに旋回 30度
            self.tello.rotate_counter_clockwise(30)
            time.sleep(1)
        elif control_flag == 'r':  # 上昇 30cm
            self.tello.move_up(30)
            time.sleep(1)
        elif control_flag == 'f':  # 下降 30cm
            self.tello.move_down(30)
            time.sleep(1)
        elif control_flag == 'b_flip':
            try:
                self.tello.flip_back()
            except TelloException:
                print('[ERROR] Failed flip')
            time.sleep(1)
            self.is_flip = False



# メイン関数
def main():
    ip = '192.168.10.1'
    port = '11111'

    tello = TelloControl(ip, port)

    try:
        tello.run()
    except(KeyboardInterrupt, SystemExit):    # Ctrl+cが押されたらループ脱出
        print('[Finish] Press Ctrl+C to exit')
        tello.stop()
        sys.exit()


# "python3 color_tracking.py"として実行された時だけ動く様にするおまじない処理
if __name__ == "__main__":      # importされると__name_に"__main__"は入らないので，pyファイルが実行されたのかimportされたのかを判断できる．
    main()    # メイン関数を実行
